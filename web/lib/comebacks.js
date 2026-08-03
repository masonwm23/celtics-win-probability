/**
 * Comeback wins, ranked by how unlikely the model thought they were.
 *
 * WHAT THE RANKING IS
 * -------------------
 * Wins, ordered by the LOWEST out-of-fold win probability the game ever
 * reached. Top of the list is the game where the model was most convinced
 * Boston were going to lose, and they won anyway.
 *
 * Every probability in the series is out of fold: it comes from a model fitted
 * on the other seven seasons, so it never saw the game it is judging. That is
 * what makes "the model gave them 0.2%" a claim worth making rather than a
 * model reciting an outcome it was trained on.
 *
 * WHY PROBABILITY AND NOT POINTS
 * ------------------------------
 * A twenty point hole in the second quarter is a bigger deficit than a four
 * point hole with a minute left, and the second is the more improbable win.
 * Ranking by deficit answers "how big was the hole", ranking by probability
 * answers "how close were they to gone". This list answers the second.
 *
 * The deficit is still shown on every row when the serving index carries it,
 * because "down 32, model had them at 0.9%" is the whole story in one line. It
 * does not affect the order.
 *
 * THE 50% CUT
 * -----------
 * A win in which Boston were never below even money is not a comeback: there
 * was no point at which the model had them losing. Those are excluded and
 * counted in the header rather than padding the tail of the list. The cut is
 * on the model's probability, so it is a statement about the model, and the
 * interface says so.
 */

import { formatClock, percent, periodLabel } from "./format.js";

/** Below this, the model had Boston losing. */
export const UNDERDOG = 0.5;

/**
 * Whether the index carries the points deficit as well.
 *
 * Optional. The ranking does not need it; the row shows it when it is there.
 * An index built before scripts/42_comeback_index.py ran does not have it, and
 * that is not a reason to show nothing.
 */
export function hasDeficitData(games) {
  if (!Array.isArray(games) || games.length === 0) return false;
  return games.every((g) => Number.isFinite(g?.largest_deficit));
}

/** "Q2 3:58", or null when the game carries no deficit moment. */
export function deficitMoment(game) {
  if (!game?.deficit_period) return null;
  return `${periodLabel(game.deficit_period)} ${formatClock(game.deficit_clock)}`;
}

/**
 * How long the odds were, in words, for the row's own label.
 *
 * Deliberately plain. "1 in 500" is a restatement of the probability, not an
 * extra claim about it.
 */
export function longShot(lowestWp) {
  const p = Number(lowestWp);
  if (!Number.isFinite(p) || p <= 0) return null;
  if (p >= UNDERDOG) return null;
  // "1 in 2" is a coin flip restated at more length, so it is not said.
  const odds = Math.round(1 / p);
  return odds >= 3 ? `about 1 in ${odds}` : null;
}

/**
 * The comeback wins of one season, longest odds first.
 *
 * Ties break on the larger deficit, then on date, so the order is total and
 * the same list comes back on every render.
 */
export function comebackWins(games, season, { limit = 0, threshold = UNDERDOG } = {}) {
  const rows = (games || [])
    .filter((g) => !season || g.season === season)
    .filter((g) => g.celtics_won)
    .filter((g) => Number.isFinite(g.lowest_wp) && g.lowest_wp < threshold)
    .map((g) => ({
      ...g,
      moment: deficitMoment(g),
      odds: longShot(g.lowest_wp),
      lowestLabel: percent(g.lowest_wp, g.lowest_wp < 0.01 ? 2 : 1),
    }));

  rows.sort((a, b) => {
    if (a.lowest_wp !== b.lowest_wp) return a.lowest_wp - b.lowest_wp;
    const da = Number.isFinite(a.largest_deficit) ? a.largest_deficit : -1;
    const db = Number.isFinite(b.largest_deficit) ? b.largest_deficit : -1;
    if (da !== db) return db - da;
    return a.date.localeCompare(b.date);
  });

  return limit > 0 ? rows.slice(0, limit) : rows;
}

/**
 * What the season looked like, so the list can say what it left out.
 *
 * `neverBehind` is wins in which the model never had Boston below even money.
 * Those are real wins and the wording should not imply otherwise.
 */
export function comebackSummary(games, season, { threshold = UNDERDOG } = {}) {
  const inSeason = (games || []).filter((g) => !season || g.season === season);
  const wins = inSeason.filter((g) => g.celtics_won);
  const ranked = comebackWins(inSeason, season, { threshold });
  return {
    games: inSeason.length,
    wins: wins.length,
    losses: inSeason.length - wins.length,
    comebacks: ranked.length,
    neverBehind: wins.length - ranked.length,
    lowest: ranked.length ? ranked[0].lowest_wp : null,
    largestDeficit: ranked.reduce(
      (best, g) => (Number.isFinite(g.largest_deficit) && g.largest_deficit > best
        ? g.largest_deficit
        : best),
      0
    ),
  };
}
