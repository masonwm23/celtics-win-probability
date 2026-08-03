/**
 * Reading a game rather than decorating it.
 *
 * Four things, all computed from the out-of-fold probability series and the
 * event table that are already on screen. Nothing here is modelled, estimated
 * or inferred.
 *
 * WHAT THIS DELIBERATELY DOES NOT DO
 * ----------------------------------
 * It does not explain WHY the probability moved. The model is a gradient
 * boosting ensemble over thirteen features; it does not expose a reason, and
 * writing one would be inventing a causal story to fit a number. What
 * `changeSummary` returns is a statement of what CHANGED in two of the model's
 * inputs, side by side with what the probability did. Those are two facts, not
 * a cause and an effect, and the interface says so.
 *
 * It also does not compute leverage in the usual sense. Leverage is the swing
 * a moment COULD produce, which needs an estimator over game states built out
 * of fold, and that does not exist yet. What is here is the swing that
 * actually happened, ranked within this game, and it is labelled as realised
 * swing rather than borrowing a word that means something else.
 */

import { formatClock, periodLabel } from "./format.js";

/** Probability change on one event, in percentage points. */
export function deltaAt(events, index) {
  if (!events?.wp || index <= 0 || index >= events.wp.length) return 0;
  return (events.wp[index] - events.wp[index - 1]) * 100;
}

/**
 * The biggest probability moves in this game, largest first.
 *
 * Real events, real probabilities, sorted. The interface makes each row seek
 * to that play, so the list is a way into the game rather than a summary of it.
 */
export function biggestSwings(events, { limit = 6 } = {}) {
  const n = events?.wp?.length || 0;
  const rows = [];
  for (let i = 1; i < n; i += 1) {
    const points = deltaAt(events, i);
    if (Math.abs(points) < 0.05) continue;
    rows.push({
      index: i,
      points,
      towards: points > 0 ? "bos" : "opp",
      period: events.period[i],
      clock: events.clock[i],
      label: `${periodLabel(events.period[i])} ${formatClock(events.clock[i])}`,
      description: events.description[i] || "",
    });
  }
  rows.sort((a, b) => Math.abs(b.points) - Math.abs(a.points));
  return rows.slice(0, limit);
}

/** Bands for how big a realised swing was, relative to this game. */
export const SWING_BANDS = [
  { at: 0.9, label: "Very large" },
  { at: 0.75, label: "Large" },
  { at: 0.5, label: "Moderate" },
  { at: 0, label: "Small" },
];

/**
 * How this play's swing compares with every other play in the same game.
 *
 * A percentile over the absolute changes actually observed. Scoped to one game
 * on purpose: a 3-point move means something different in a blowout and in a
 * one-possession fourth quarter, and this says which game it is being measured
 * against rather than implying a league-wide scale.
 */
export function swingSize(events, index) {
  const n = events?.wp?.length || 0;
  if (n < 2) return null;
  const points = deltaAt(events, index);
  const size = Math.abs(points);

  let below = 0;
  let counted = 0;
  for (let i = 1; i < n; i += 1) {
    const other = Math.abs(deltaAt(events, i));
    counted += 1;
    if (other < size) below += 1;
  }
  const percentile = counted ? below / counted : 0;
  const band = SWING_BANDS.find((b) => percentile >= b.at) || SWING_BANDS[SWING_BANDS.length - 1];
  return { points, size, percentile, label: band.label, of: counted };
}

/**
 * The current scoring run, counted backwards from this play.
 *
 * A run is consecutive points by one team with none by the other. Both scores
 * are columns in the event table, so this is arithmetic on recorded numbers.
 * Returns null when the last scoring event was a shared possession or the game
 * has not started.
 */
export function scoringRun(events, cursor, opponentAbbrev) {
  const bos = events?.celtics_score;
  const opp = events?.opponent_score;
  if (!bos || !opp || cursor < 1) return null;

  let team = null;
  let points = 0;
  let startedAt = cursor;

  for (let i = cursor; i >= 1; i -= 1) {
    const bosGain = bos[i] - bos[i - 1];
    const oppGain = opp[i] - opp[i - 1];
    if (bosGain === 0 && oppGain === 0) continue;
    // A single event cannot score for both teams. If one ever did, stop
    // rather than attribute it.
    if (bosGain > 0 && oppGain > 0) break;

    const scorer = bosGain > 0 ? "bos" : "opp";
    if (team === null) team = scorer;
    if (scorer !== team) break;

    points += bosGain > 0 ? bosGain : oppGain;
    startedAt = i;
  }

  if (!team || points === 0) return null;
  return {
    team,
    abbrev: team === "bos" ? "BOS" : String(opponentAbbrev || "OPP"),
    points,
    startedAt,
    since: `${periodLabel(events.period[startedAt])} ${formatClock(events.clock[startedAt])}`,
  };
}

/**
 * What changed, in words. NOT why.
 *
 * Every clause is a recorded quantity: the score margin either side of the
 * play, the clock, and the out-of-fold probability either side. The model's
 * reasoning is not exposed by the model and is not guessed at here.
 */
export function changeSummary(events, cursor, opponentAbbrev) {
  const n = events?.wp?.length || 0;
  if (!n || cursor < 0 || cursor >= n) return null;

  const points = deltaAt(events, cursor);
  const marginNow = events.margin[cursor];
  const marginBefore = cursor > 0 ? events.margin[cursor - 1] : marginNow;
  const clock = `${formatClock(events.clock[cursor])} left in ${periodLabel(events.period[cursor])}`;

  const lead = (margin) => {
    if (margin === 0) return "level";
    const side = margin > 0 ? "BOS" : String(opponentAbbrev || "OPP");
    return `${side} by ${Math.abs(margin)}`;
  };

  const marginPart =
    marginBefore === marginNow
      ? `The score margin did not change (${lead(marginNow)})`
      : `The score margin went from ${lead(marginBefore)} to ${lead(marginNow)}`;

  const wpPart =
    Math.abs(points) < 0.05
      ? "Boston's probability was unchanged"
      : `Boston's probability moved ${points > 0 ? "up" : "down"} ${Math.abs(points).toFixed(1)} percentage points`;

  return {
    text: `${marginPart}, ${clock}. ${wpPart}.`,
    caveat:
      "Margin and time remaining are two of the model's thirteen inputs. " +
      "This states what changed alongside the probability, not what caused it.",
    points,
  };
}
