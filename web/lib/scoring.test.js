import test from "node:test";
import assert from "node:assert/strict";

import {
  eventPoints,
  pointsAt,
  pointsOnEvent,
  pointsTimeline,
  reconcile,
  teamScoring,
  reboundTimeline,
} from "./scoring.js";

/**
 * The fixture below is Al Horford's actual scoring sequence from game
 * 0022200107, descriptions copied verbatim. It is here because it is the case
 * that decides how this module works: the feed's own running counter goes
 * 2, 4, 12, 6, 7, 9 while his boxscore total is 12.
 */
const HORFORD = 201143;
const TATUM = 1628369;

const GAME = {
  action_type: [
    "Made Shot", "Missed Shot", "Made Shot", "Made Shot",
    "Made Shot", "Free Throw", "Made Shot", "Free Throw",
  ],
  shot_value: [2, 2, 2, 3, 2, 0, 2, 0],
  description: [
    "Horford 1' Driving Reverse Layup (2 PTS)",
    "MISS Tatum 25' 3PT Jump Shot",
    "Horford 1' Driving Layup (4 PTS) (Tatum 4 AST)",
    "Horford 3PT Jump Shot (12 PTS) (Tatum 6 AST)",
    "Horford 1' Layup (6 PTS) (White 5 AST)",
    "Horford Free Throw 1 of 1 (7 PTS)",
    "Horford 6' Turnaround Bank Shot (9 PTS)",
    "MISS Tatum Free Throw 2 of 2",
  ],
  person_id: [HORFORD, TATUM, HORFORD, HORFORD, HORFORD, HORFORD, HORFORD, TATUM],
};

// ---------------------------------------------------------------------------
// What one event is worth
// ---------------------------------------------------------------------------

test("a made field goal is worth its shot value", () => {
  assert.equal(eventPoints({ action_type: "Made Shot", shot_value: 2 }), 2);
  assert.equal(eventPoints({ action_type: "Made Shot", shot_value: 3 }), 3);
});

test("a missed shot is worth nothing", () => {
  assert.equal(eventPoints({ action_type: "Missed Shot", shot_value: 3 }), 0);
});

test("a free throw is worth one, and its outcome comes from the description", () => {
  // shot_value is 0 and shot_result is empty on every free throw in the
  // dataset, so the leading MISS is the only record of the outcome.
  assert.equal(eventPoints({
    action_type: "Free Throw", shot_value: 0,
    description: "Brown Free Throw 2 of 2 (7 PTS)",
  }), 1);
  assert.equal(eventPoints({
    action_type: "Free Throw", shot_value: 0,
    description: "MISS Brown Free Throw 1 of 2",
  }), 0);
});

test("a made shot with a corrupt value counts as two, never as zero", () => {
  // A made shot worth nothing is a claim the feed never makes. Two is the
  // conservative reading; zero would silently lose points from the total.
  assert.equal(eventPoints({ action_type: "Made Shot", shot_value: 0 }), 2);
  assert.equal(eventPoints({ action_type: "Made Shot", shot_value: null }), 2);
});

test("non-scoring events are worth nothing", () => {
  for (const action of ["Rebound", "Foul", "Turnover", "Substitution", "period", ""]) {
    assert.equal(eventPoints({ action_type: action, shot_value: 3 }), 0, action);
  }
});

// ---------------------------------------------------------------------------
// The running total, and the feed counter it deliberately ignores
// ---------------------------------------------------------------------------

test("the total is summed, not read from the feed's counter", () => {
  // The counter in these descriptions reads 2, 4, 12, 6, 7, 9. It is not even
  // monotone. Summing gives 2, 4, 7, 9, 10, 12, and 12 is the boxscore total.
  const timeline = pointsTimeline(GAME);
  const totals = timeline.get(HORFORD).map((entry) => entry.total);
  assert.deepEqual(totals, [2, 4, 7, 9, 10, 12]);
  assert.equal(pointsAt(timeline, HORFORD, 7), 12, "final total must be 12");
});

test("the feed's counter is never trusted, even when it disagrees loudly", () => {
  const timeline = pointsTimeline(GAME);
  // At the 3PT shot the description says 12. The real running total is 7.
  assert.equal(pointsAt(timeline, HORFORD, 3), 7);
  assert.notEqual(pointsAt(timeline, HORFORD, 3), 12);
});

test("the total is monotone: it never goes down as the game runs", () => {
  const timeline = pointsTimeline(GAME);
  let previous = 0;
  for (let i = 0; i < GAME.action_type.length; i += 1) {
    const now = pointsAt(timeline, HORFORD, i);
    assert.ok(now >= previous, `points fell at event ${i}: ${previous} -> ${now}`);
    previous = now;
  }
});

test("points before a player's first score are zero, not undefined", () => {
  const timeline = pointsTimeline(GAME);
  assert.equal(pointsAt(timeline, TATUM, 0), 0);
  assert.equal(pointsAt(timeline, 999999, 5), 0, "an unknown player scores zero");
  assert.equal(pointsAt(new Map(), HORFORD, 5), 0);
});

test("a missed free throw adds nothing to the timeline", () => {
  const timeline = pointsTimeline(GAME);
  assert.equal(timeline.has(TATUM), false,
    "Tatum missed both his attempts and must not appear as a scorer");
});

test("the per-event contribution is what drives the flash", () => {
  const timeline = pointsTimeline(GAME);
  assert.equal(pointsOnEvent(timeline, HORFORD, 3), 3);
  assert.equal(pointsOnEvent(timeline, HORFORD, 5), 1);
  assert.equal(pointsOnEvent(timeline, HORFORD, 1), 0, "not his event");
});

// ---------------------------------------------------------------------------
// The board
// ---------------------------------------------------------------------------

const PLAYERS = {
  201143: { person_id: HORFORD, name: "Al Horford", is_celtics: true },
  1628369: { person_id: TATUM, name: "Jayson Tatum", is_celtics: true },
  203999: { person_id: 203999, name: "Nikola Jokic", is_celtics: false },
};

test("the board keeps players who have not scored", () => {
  const timeline = pointsTimeline(GAME);
  const rows = teamScoring(timeline, PLAYERS, 7, { celtics: true });
  assert.equal(rows.length, 2);
  assert.equal(rows[0].player.person_id, HORFORD);
  assert.equal(rows[0].points, 12);
  assert.equal(rows[1].points, 0, "a scoreless player stays on the board");
});

test("the board separates the teams", () => {
  const timeline = pointsTimeline(GAME);
  const opponent = teamScoring(timeline, PLAYERS, 7, { celtics: false });
  assert.equal(opponent.length, 1);
  assert.equal(opponent[0].player.name, "Nikola Jokic");
});

test("ties break by name, so the order does not jitter as points move", () => {
  const timeline = pointsTimeline({
    action_type: [], shot_value: [], description: [], person_id: [],
  });
  const rows = teamScoring(timeline, PLAYERS, 0, { celtics: true });
  assert.deepEqual(rows.map((r) => r.player.name), ["Al Horford", "Jayson Tatum"]);
});

test("reconciliation against the scoreboard is reported, not assumed", () => {
  const timeline = pointsTimeline(GAME);
  const rows = teamScoring(timeline, PLAYERS, 7, { celtics: true });
  assert.deepEqual(reconcile(rows, 12), { total: 12, expected: 12, agrees: true });
  assert.equal(reconcile(rows, 11).agrees, false,
    "a disagreement must surface rather than being rounded away");
});

// ---------------------------------------------------------------------------
// Rebounds, counted rather than read
// ---------------------------------------------------------------------------

const REBOUNDS = {
  action_type: ["Rebound", "Missed Shot", "Rebound", "Rebound", "Rebound"],
  description: [
    "Horford REBOUND (Off:0 Def:1)",
    "MISS Tatum 25' 3PT Jump Shot",
    "Tatum REBOUND (Off:1 Def:0)",
    "Horford REBOUND (Off:1 Def:1)",
    "CELTICS Rebound",
  ],
  person_id: [HORFORD, TATUM, TATUM, HORFORD, 0],
};

test("rebounds accumulate one per event, per player", () => {
  const timeline = reboundTimeline(REBOUNDS);
  assert.equal(pointsAt(timeline, HORFORD, 4), 2);
  assert.equal(pointsAt(timeline, TATUM, 4), 1);
});

test("a rebound counts only from the event it happened on", () => {
  const timeline = reboundTimeline(REBOUNDS);
  assert.equal(pointsAt(timeline, HORFORD, 0), 1);
  assert.equal(pointsAt(timeline, HORFORD, 2), 1, "not yet his second");
  assert.equal(pointsAt(timeline, HORFORD, 3), 2);
});

test("a team rebound belongs to no player", () => {
  const timeline = reboundTimeline(REBOUNDS);
  const everyone = [...timeline.values()].flat().length;
  assert.equal(everyone, 3, "the CELTICS rebound must not be credited to anyone");
  assert.equal(timeline.has(0), false);
});

test("scrubbing backward reduces rebound totals the same way it does points", () => {
  const timeline = reboundTimeline(REBOUNDS);
  const forward = [0, 1, 2, 3, 4].map((i) => pointsAt(timeline, HORFORD, i));
  assert.deepEqual(forward, [1, 1, 1, 2, 2]);
  // And going back down the same path returns the same values.
  const backward = [4, 3, 2, 1, 0].map((i) => pointsAt(timeline, HORFORD, i));
  assert.deepEqual(backward, [2, 2, 1, 1, 1]);
});
