import test from "node:test";
import assert from "node:assert/strict";

import {
  finalHeadline,
  finalSentence,
  gameOutcome,
  overtimeLabel,
  periodEndLabel,
} from "./outcome.js";

/**
 * Boston lose 223 of the 636 games in this dataset. Anything that assumes the
 * Celtics won is wrong more than a third of the time, so the loss cases below
 * matter as much as the wins.
 */

function game({ bos, opp, periods = 4, meta = {} }) {
  const n = 3;
  return {
    events: {
      wp: [0.5, 0.5, 0.5],
      celtics_score: [0, 1, bos],
      opponent_score: [0, 0, opp],
      period: [1, periods, periods],
    },
    meta: {
      opponent: "PHX",
      opponent_name: "Phoenix Suns",
      celtics_name: "Boston Celtics",
      celtics_final: bos,
      opponent_final: opp,
      periods,
      ...meta,
    },
    last: n - 1,
  };
}

// ---------------------------------------------------------------------------
// Only at the end
// ---------------------------------------------------------------------------

test("no winner is claimed before the final event", () => {
  const g = game({ bos: 116, opp: 109 });
  for (const cursor of [0, 1]) {
    const outcome = gameOutcome(g.events, g.meta, cursor);
    assert.equal(outcome.isFinal, false);
    assert.equal(outcome.winnerName, undefined);
    assert.equal(finalHeadline(outcome), null);
    assert.equal(finalSentence(outcome), null);
  }
});

test("the final event reports the game as complete", () => {
  const g = game({ bos: 116, opp: 109 });
  assert.equal(gameOutcome(g.events, g.meta, g.last).isFinal, true);
});

// ---------------------------------------------------------------------------
// Who won
// ---------------------------------------------------------------------------

test("a regulation Boston win names Boston", () => {
  const g = game({ bos: 120, opp: 111 });
  const outcome = gameOutcome(g.events, g.meta, g.last);
  assert.equal(outcome.celticsWon, true);
  assert.equal(outcome.winnerName, "Boston Celtics");
  assert.equal(outcome.loserName, "Phoenix Suns");
  assert.equal(finalHeadline(outcome), "FINAL · Celtics win 120-111");
  assert.equal(finalSentence(outcome),
               "Game complete · Boston Celtics defeat Phoenix Suns");
});

test("a Boston LOSS names the opponent as the winner", () => {
  // The case that a hardcoded "Celtics win" would get wrong.
  const g = game({ bos: 109, opp: 115 });
  const outcome = gameOutcome(g.events, g.meta, g.last);
  assert.equal(outcome.celticsWon, false);
  assert.equal(outcome.winnerName, "Phoenix Suns");
  assert.equal(outcome.loserName, "Boston Celtics");
  assert.equal(finalHeadline(outcome), "FINAL · PHX win 115-109");
  assert.equal(finalSentence(outcome),
               "Game complete · Phoenix Suns defeat Boston Celtics");
});

test("the winner comes from the score, not from the meta flag", () => {
  // celtics_won is deliberately set to the WRONG value here. The score is the
  // authority, so the outcome must ignore the flag.
  const g = game({ bos: 100, opp: 120, meta: { celtics_won: true } });
  const outcome = gameOutcome(g.events, g.meta, g.last);
  assert.equal(outcome.celticsWon, false);
  assert.equal(outcome.winnerName, "Phoenix Suns");
});

// ---------------------------------------------------------------------------
// Overtime
// ---------------------------------------------------------------------------

test("overtime wording follows the period count", () => {
  assert.equal(overtimeLabel(4), null);
  assert.equal(overtimeLabel(5), "in overtime");
  assert.equal(overtimeLabel(6), "in double overtime");
  assert.equal(overtimeLabel(7), "in triple overtime");
  assert.equal(overtimeLabel(8), "in 4 overtimes");
});

test("an overtime win says so in the headline", () => {
  const g = game({ bos: 116, opp: 109, periods: 5 });
  const outcome = gameOutcome(g.events, g.meta, g.last);
  assert.equal(finalHeadline(outcome), "FINAL · Celtics win 116-109 in overtime");
  // Not "End of 1st OT". There was no second one and the game has finished, so
  // the ordinal only invites a question the game already answered.
  assert.equal(outcome.periodEnd, "End of OT");
});

test("a double-overtime loss is right on both counts", () => {
  const g = game({ bos: 118, opp: 121, periods: 6 });
  const outcome = gameOutcome(g.events, g.meta, g.last);
  assert.equal(finalHeadline(outcome), "FINAL · PHX win 121-118 in double overtime");
  assert.equal(outcome.periodEnd, "End of 2nd OT");
});

test("the period label stays available as secondary context", () => {
  assert.equal(periodEndLabel(4), "End of Q4");
  assert.equal(periodEndLabel(5), "End of OT");
  assert.equal(periodEndLabel(6, 6), "End of 2nd OT");
  assert.equal(periodEndLabel(7, 7), "End of 3rd OT");
});

test("the ordinal is dropped only when there was a single overtime", () => {
  // One overtime: no ordinal, because there is nothing to distinguish it from.
  assert.equal(periodEndLabel(5, 5), "End of OT");

  // Two overtimes: the ordinal now carries information, so it is written. This
  // also covers the mid-game case, where the first OT ends in a game that goes
  // on to a second and the reader genuinely needs to know which one this was.
  assert.equal(periodEndLabel(5, 6), "End of 1st OT");
  assert.equal(periodEndLabel(6, 6), "End of 2nd OT");

  // Regulation is unaffected either way.
  assert.equal(periodEndLabel(4, 6), "End of Q4");
  assert.equal(periodEndLabel(1, 4), "End of Q1");

  // Junk in, null out, as before.
  assert.equal(periodEndLabel("nonsense"), null);
});

// ---------------------------------------------------------------------------
// When the sources disagree
// ---------------------------------------------------------------------------

test("a disagreement between the event score and the boxscore is surfaced", () => {
  const g = game({ bos: 116, opp: 109, meta: { celtics_final: 117 } });
  const outcome = gameOutcome(g.events, g.meta, g.last);
  assert.equal(outcome.scoresAgree, false,
    "the panel must be able to say the two sources disagree");
  // A winner is still named from the event columns, but the caller is told.
  assert.equal(outcome.winnerName, "Boston Celtics");
});

test("a tie is reported as invalid rather than resolved", () => {
  const g = game({ bos: 110, opp: 110 });
  const outcome = gameOutcome(g.events, g.meta, g.last);
  assert.equal(outcome.tie, true);
  assert.equal(outcome.winnerName, null);
  assert.match(finalHeadline(outcome), /not a valid result/);
});

test("a payload with no full team names falls back to tricodes", () => {
  const g = game({ bos: 116, opp: 109,
                   meta: { opponent_name: undefined, celtics_name: undefined } });
  const outcome = gameOutcome(g.events, g.meta, g.last);
  assert.equal(outcome.winnerName, "BOS");
  assert.equal(outcome.loserName, "PHX");
});
