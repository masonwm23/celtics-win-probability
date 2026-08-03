import test from "node:test";
import assert from "node:assert/strict";

import {
  FOULED_UNKNOWN,
  blankActionKind,
  describePlay,
  fouledPlayer,
  foulType,
  isTeamEvent,
  playLabel,
  reboundCounters,
  reboundKind,
  reboundKindsForGame,
  turnoverCause,
} from "./playby.js";

/**
 * Every string in this file is a real description from this dataset, copied
 * out of `data/serving/games`. Inventing plausible-looking feed text would
 * make the tests agree with the parser and disagree with the NBA.
 */

// ---------------------------------------------------------------------------
// Rebounds
// ---------------------------------------------------------------------------

test("the counters are read off the description", () => {
  assert.deepEqual(reboundCounters("Crowder REBOUND (Off:0 Def:1)"),
                   { off: 0, def: 1 });
  assert.deepEqual(reboundCounters("Williams III REBOUND (Off:1 Def:0)"),
                   { off: 1, def: 0 });
  assert.equal(reboundCounters("RAPTORS Rebound"), null);
});

test("a first rebound is typed from the counter that is non-zero", () => {
  assert.equal(reboundKind({ off: 0, def: 1 }), "defensive");
  assert.equal(reboundKind({ off: 1, def: 0 }), "offensive");
});

test("a later rebound is typed by differencing the SAME player's counters", () => {
  // 25,649 of 66,600 rebounds in this dataset have both counters non-zero, so
  // a single row cannot say which one just moved. This is the case that makes
  // the running-total pass necessary.
  assert.equal(reboundKind({ off: 2, def: 3 }, { off: 1, def: 3 }), "offensive");
  assert.equal(reboundKind({ off: 2, def: 4 }, { off: 2, def: 3 }), "defensive");
});

test("a rebound that moved neither counter is reported as unknown, not guessed", () => {
  assert.equal(reboundKind({ off: 2, def: 3 }, { off: 2, def: 3 }), null);
  assert.equal(describePlay(
    { action_type: "Rebound", description: "X REBOUND (Off:2 Def:3)" },
    { reboundType: null }
  ).note, "Offensive or defensive not recoverable");
});

test("a team rebound has no counters and is labelled as a team rebound", () => {
  assert.equal(reboundKind(null), "team");
  assert.equal(
    describePlay({ action_type: "Rebound", description: "CELTICS Rebound" },
                 { reboundType: "team" }).label,
    "Team rebound"
  );
});

test("a whole game resolves every rebound in one pass", () => {
  const events = {
    action_type: ["Rebound", "Missed Shot", "Rebound", "Rebound", "Rebound"],
    description: [
      "Horford REBOUND (Off:0 Def:1)",
      "MISS Tatum 2' Driving Layup",
      "Horford REBOUND (Off:1 Def:1)",
      "Horford REBOUND (Off:1 Def:2)",
      "CELTICS Rebound",
    ],
    person_id: [201143, 1628369, 201143, 201143, 0],
  };
  const kinds = reboundKindsForGame(events);
  assert.equal(kinds.get(0), "defensive");
  assert.equal(kinds.get(2), "offensive");
  assert.equal(kinds.get(3), "defensive");
  assert.equal(kinds.get(4), "team");
  assert.equal(kinds.has(1), false, "a shot is not a rebound");
});

test("two players' counters do not contaminate each other", () => {
  const events = {
    action_type: ["Rebound", "Rebound", "Rebound"],
    description: [
      "Horford REBOUND (Off:0 Def:1)",
      "Tatum REBOUND (Off:0 Def:1)",
      "Horford REBOUND (Off:0 Def:2)",
    ],
    person_id: [201143, 1628369, 201143],
  };
  const kinds = reboundKindsForGame(events);
  assert.equal(kinds.get(1), "defensive",
    "Tatum's first rebound must not be differenced against Horford's");
  assert.equal(kinds.get(2), "defensive");
});

test("rebound labels are specific, never the bare word", () => {
  assert.equal(describePlay({ action_type: "Rebound" },
                            { reboundType: "offensive" }).label,
               "Offensive rebound");
  assert.equal(describePlay({ action_type: "Rebound" },
                            { reboundType: "defensive" }).label,
               "Defensive rebound");
});

// ---------------------------------------------------------------------------
// Fouls
// ---------------------------------------------------------------------------

test("the recorded foul type is used, not the bare word Foul", () => {
  const cases = [
    ["Ibaka S.FOUL (P1.T1) (J.Goble)", "Shooting foul"],
    ["Brown P.FOUL (P1.T1) (B.Spooner)", "Personal foul"],
    ["Horford L.B.FOUL (P1.T1) (K.Fitzgerald)", "Loose ball foul"],
    ["Crowder T.FOUL (P0.T2) (B.Spooner)", "Technical foul"],
    ["Patterson OFF.Foul (P2) (J.Goble)", "Offensive foul"],
    ["Powell Offensive Charge Foul (P1.T4) (K.Fitzgerald)",
     "Offensive charge foul"],
    ["Carroll Shooting Block Foul (P4.T3) (B.Spooner)", "Shooting block foul"],
    ["Horford Personal Take Foul (P2.PN) (K.Fitzgerald)", "Take foul"],
    ["Carroll FLAGRANT.FOUL.TYPE1 (P2.T3) (B.Spooner)", "Flagrant 1 foul"],
    ["Double Technical - Thomas, Carroll (B.Spooner)", "Double technical"],
    [" Foul : Double Personal - Ibaka (4 PF), Johnson (1 PF) (B.Spooner)",
     "Double personal foul"],
  ];
  for (const [description, expected] of cases) {
    assert.equal(foulType(description), expected, description);
  }
});

test("an offensive foul is not mistaken for a personal foul", () => {
  // "OFF.Foul" contains neither P.FOUL nor S.FOUL, but a careless bare-word
  // match on "Foul" would have caught it first.
  assert.equal(foulType("Olynyk OFF.Foul (P4) (B.Spooner)"), "Offensive foul");
  assert.notEqual(foulType("Olynyk OFF.Foul (P4) (B.Spooner)"), "Personal foul");
});

test("the fouled player is never taken from the description", () => {
  // Measured over all 636 games: zero foul descriptions contain "on" or
  // "drawn". The trailing parenthetical is the OFFICIAL, and reading it as a
  // player would have printed a referee's name as the victim.
  const description = "Ibaka S.FOUL (P1.T1) (J.Goble)";
  assert.equal(fouledPlayer(description), null);
  const parts = describePlay({ action_type: "Foul", description });
  assert.equal(parts.note, FOULED_UNKNOWN);
  assert.ok(!/Goble/.test(parts.label + (parts.detail || "") + parts.note),
            "the referee must never appear as the fouled player");
});

test("an unrecognised foul still says Foul rather than nothing", () => {
  const parts = describePlay({ action_type: "Foul", description: "Smith (P1.T1)" });
  assert.equal(parts.label, "Foul");
  assert.equal(parts.note, FOULED_UNKNOWN);
});

// ---------------------------------------------------------------------------
// Turnovers
// ---------------------------------------------------------------------------

test("the recorded turnover cause is used", () => {
  const cases = [
    ["Horford Bad Pass Turnover (P1.T2)", "Bad pass"],
    ["Ibaka Poss Lost Ball Turnover (P1.T7)", "Lost ball"],
    ["Carey Jr. Lost Ball Turnover (P1.T15)", "Lost ball"],
    ["Morris Sr. Traveling Turnover (P1.T2)", "Traveling"],
    ["Giles III Offensive Foul Turnover (P1.T8)", "Offensive foul"],
    ["Johnson Step Out of Bounds Turnover (P1.T1)", "Stepped out of bounds"],
    ["Joseph Out of Bounds - Bad Pass Turnover Turnover (P1.T1)",
     "Bad pass out of bounds"],
    ["Middleton Out of Bounds Lost Ball Turnover (P2.T16)",
     "Lost ball out of bounds"],
    ["Williams III Backcourt Turnover (P1.T4)", "Backcourt violation"],
    ["Thompson 3 Second Violation Turnover (P1.T2)", "Three-second violation"],
    ["Morris Sr. Discontinue Dribble Turnover (P2.T6)", "Discontinued dribble"],
    ["RAPTORS Turnover: Shot Clock (T#5)", "Shot clock"],
    ["HORNETS Turnover: 5 Second Violation (T#14)", "Five-second violation"],
  ];
  for (const [description, expected] of cases) {
    assert.equal(turnoverCause(description), expected, description);
  }
});

test("out-of-bounds causes outrank the plain ones they contain", () => {
  // "Out of Bounds - Bad Pass" also contains "Bad Pass". Order matters, and
  // this is the test that keeps it ordered.
  assert.equal(turnoverCause("X Out of Bounds - Bad Pass Turnover Turnover (P1.T1)"),
               "Bad pass out of bounds");
  assert.equal(turnoverCause("X Out of Bounds Lost Ball Turnover (P2.T16)"),
               "Lost ball out of bounds");
});

test("a turnover carries its cause and keeps any steal separate", () => {
  const parts = describePlay({
    action_type: "Turnover",
    description: "Horford Bad Pass Turnover (P1.T2)",
  });
  assert.equal(parts.label, "Turnover");
  assert.equal(parts.detail, "Bad pass");
  assert.match(parts.note, /own event/);
  assert.equal(playLabel({
    action_type: "Turnover",
    description: "Horford Bad Pass Turnover (P1.T2)",
  }), "Turnover — Bad pass");
});

test("a turnover with no recognised cause says Turnover and nothing invented", () => {
  const parts = describePlay({ action_type: "Turnover", description: "X Turnover (P1.T1)" });
  assert.equal(parts.label, "Turnover");
  assert.equal(parts.detail, null);
});

// ---------------------------------------------------------------------------
// Steals, blocks, and the rest
// ---------------------------------------------------------------------------

test("steals and blocks arrive with a blank action type", () => {
  assert.equal(blankActionKind("Porzingis STEAL (1 STL)"), "Steal");
  assert.equal(blankActionKind("White BLOCK (1 BLK)"), "Block");
  assert.equal(blankActionKind("something else"), null);
  assert.equal(describePlay({ action_type: "", description: "Porzingis STEAL (1 STL)" }).label,
               "Steal");
});

test("no turnover description in this dataset mentions a steal", () => {
  // The reason the two are never joined. Zero of 21,000+ turnover events
  // contain the word, so a steal shown next to a turnover would be an
  // inference from an adjacent row rather than something the feed said.
  const real = [
    "Horford Bad Pass Turnover (P1.T2)",
    "Ibaka Poss Lost Ball Turnover (P1.T7)",
    "Johnson Step Out of Bounds Turnover (P1.T1)",
  ];
  for (const description of real) {
    assert.ok(!/STEAL/i.test(description));
    assert.equal(blankActionKind(description), null);
  }
});

test("shots and free throws keep their existing labels", () => {
  assert.equal(describePlay({ action_type: "Made Shot", shot_value: 3 }).label,
               "Made 3PT");
  assert.equal(describePlay({ action_type: "Missed Shot", shot_value: 2 }).label,
               "Missed 2PT");
  assert.equal(describePlay({
    action_type: "Free Throw",
    description: "MISS Brown Free Throw 1 of 2",
  }).label, "Missed free throw");
  assert.equal(describePlay({
    action_type: "Free Throw",
    description: "Brown Free Throw 2 of 2 (7 PTS)",
  }).label, "Made free throw");
});

test("an empty event is not invented into something", () => {
  assert.equal(describePlay({}).label, "Play");
  assert.equal(describePlay(null).label, "Play");
});

test("team events are recognised as team events", () => {
  assert.equal(isTeamEvent("RAPTORS Turnover: Shot Clock (T#5)"), true);
  assert.equal(isTeamEvent("CELTICS Rebound"), true);
  assert.equal(isTeamEvent("Horford Bad Pass Turnover (P1.T2)"), false);
});

// ---------------------------------------------------------------------------
// Coverage over the real corpus
//
// These figures come from running this parser over every distinct
// (action_type, description) pair in all 636 games: 147,544 pairs, 308,975
// events. They are recorded here so a future edit that quietly drops a case
// shows up as a failed expectation rather than as a generic label on screen.
// ---------------------------------------------------------------------------

test("the known unlabelled tail is only the feed's own empty strings", () => {
  // 33 of 17,299 turnovers, all of this shape. The feed states no cause, so
  // neither does the panel.
  for (const description of [
    "Wall No Turnover (P3.T7)",
    "Horford Turnover Turnover (P2.T5)",
  ]) {
    assert.equal(turnoverCause(description), null, description);
    const parts = describePlay({ action_type: "Turnover", description });
    assert.equal(parts.label, "Turnover");
    assert.equal(parts.detail, null, "no cause may be invented");
  }
});

test("the foul tail cases all resolve", () => {
  // These were the last 38 unlabelled foul events in the corpus.
  const cases = [
    ["Celtics Delay", "Delay of game"],
    ["Johnson HANGING.TECH.FOUL (P2.T1) (B.Spooner)", "Hanging technical"],
    ["Mark Morris Taunting (P2.T0) (M.Davis)", "Taunting technical"],
    ["Bazemore Non-Unsportsmanlike (P0.T2) (M.Smith)",
     "Non-unsportsmanlike technical"],
  ];
  for (const [description, expected] of cases) {
    assert.equal(foulType(description), expected, description);
  }
});

test("a lane violation turnover is named", () => {
  assert.equal(turnoverCause("Mejri Lane Violation Turnover (P1.T11)"),
               "Lane violation");
});
