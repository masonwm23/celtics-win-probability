import { gameOutcome } from "@/lib/outcome";

/**
 * The one place that decides when a probability stops being a forecast.
 *
 * At the last event of a finished game the clock reads 0:00 and the result is
 * already in the score columns. The model still emits something like 0.00206
 * or 0.99700 there, because a boosted tree's leaves never quite reach 0 or 1,
 * and the dashboard was rendering that as "0.2% chance of winning" underneath a
 * headline reading FINAL, TOR win 113-101. Both statements were on screen at
 * once and only one of them was true.
 *
 * So the terminal event is displayed as the settled result. Three things about
 * how that is done matter more than the change itself:
 *
 *   1. It is DISPLAY ONLY. Nothing here writes back into events.wp. The Brier
 *      score, the reliability tables and every figure in the paper are computed
 *      in Python from the stored out-of-fold probabilities and are untouched by
 *      this file. The swing strip (lib/moments) and the comeback ranking
 *      (lib/comebacks) also keep reading the raw series, so no ranking or
 *      measured change is affected by a display decision.
 *
 *   2. It applies to the LAST EVENT ONLY. A game that is mathematically over
 *      with seven seconds left still shows what the model actually said, which
 *      on the Toronto game is 0.3%. That number is a real forecast about a game
 *      still in progress and hiding it would be the dishonest version of this
 *      change. Only the clock hitting zero settles anything.
 *
 *   3. It REFUSES when the data is odd. gameOutcome cross-checks the running
 *      score columns against the boxscore finals and flags ties, which are not
 *      a valid NBA result. If either check fails, this returns the model's own
 *      number and the interface goes on saying "out of fold". A settled result
 *      is only settled if the sources agree on what the result was.
 *
 * Callers that show the settled value are expected to relabel it: it is the
 * final score talking, not the model, and the badge should say so.
 */

/**
 * Is this game finished, trustworthy, and what did it settle to.
 *
 * `ok` false means "do not settle anything", either because the game is not
 * over or because the outcome could not be established safely.
 */
export function settledOutcome(events, meta) {
  const total = events?.wp?.length || 0;
  if (!total) return { ok: false, index: -1, value: null, celticsWon: null };

  const outcome = gameOutcome(events, meta, total - 1);
  const ok =
    outcome.isFinal &&
    outcome.tie === false &&
    outcome.scoresAgree === true &&
    typeof outcome.celticsWon === "boolean";

  return {
    ok,
    index: total - 1,
    value: ok ? (outcome.celticsWon ? 1 : 0) : null,
    celticsWon: ok ? outcome.celticsWon : null,
  };
}

/**
 * The probability to PRINT at `index`, and whether it came from the scoreboard.
 *
 * Returns the model's own value alongside it, so a caller that wants to show
 * both never has to reach back into the array and risk showing a different
 * event's number.
 */
export function settledWp(events, meta, index) {
  const model = events?.wp?.[index];
  const settled = settledOutcome(events, meta);
  if (settled.ok && index === settled.index) {
    return { value: settled.value, isSettled: true, model };
  }
  return { value: model, isSettled: false, model };
}

/**
 * A copy of `series` with the terminal point moved to the settled result.
 *
 * Used for the drawn lines so the chart and the headline cannot disagree. Both
 * the Celtics-specific and the generic series resolve to the same point,
 * because they are both forecasts of one outcome and that outcome happened.
 *
 * Returns the ORIGINAL array when nothing is settled, so callers can keep
 * memoising on identity.
 */
export function settledSeries(series, events, meta) {
  const settled = settledOutcome(events, meta);
  if (!settled.ok || !series || series.length !== events.wp.length) return series;
  const out = Array.from(series);
  out[settled.index] = settled.value;
  return out;
}
