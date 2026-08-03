import test from "node:test";
import assert from "node:assert/strict";

import {
  assistLabel, assistTimeline, descriptionAliases, fold,
  resolveAssister, stripSuffix,
} from "./assists.js";
import { pointsAt } from "./scoring.js";

const P = (id, name, team) => ({ person_id: id, name, team });

test("accents are folded, because the feed and the roster disagree about them", () => {
  // 1,434 of 30,429 assist mentions failed on this alone: the description
  // writes "Schroder", "Porzingis", "Jokic"; the roster writes them accented.
  assert.equal(fold("Porziņģis"), "porzingis");
  assert.equal(fold("Jokić"), "jokic");
  assert.equal(fold("Dončić"), "doncic");
  assert.equal(fold("Schröder"), "schroder");
});

test("an exact surname wins before any suffix is stripped", () => {
  // The regression that matters. Boston carried both of these at once, and
  // stripping "III" first collapsed them and made the whole thing worse:
  // 554 games reconciling fell to 444.
  const roster = [P(1, "Grant Williams", "BOS"), P(2, "Robert Williams III", "BOS")];
  assert.equal(resolveAssister("Williams", roster, new Map())?.person_id, 1);
  assert.equal(resolveAssister("Williams III", roster, new Map())?.person_id, 2);
});

test("the suffix fallback only fires when the exact form found nothing", () => {
  // Description drops the suffix the roster carries.
  assert.equal(
    resolveAssister("Butler", [P(3, "Jimmy Butler III", "MIA")], new Map())?.person_id, 3);
  assert.equal(
    resolveAssister("Bullock", [P(4, "Reggie Bullock Jr.", "DAL")], new Map())?.person_id, 4);
  // And the other direction: description carries a suffix the roster does not.
  assert.equal(
    resolveAssister("Boston Jr.", [P(5, "Brandon Boston", "LAC")], new Map())?.person_id, 5);
});

test("a first-name prefix separates two players with one surname", () => {
  const roster = [P(6, "Marcus Morris Sr.", "BOS"), P(7, "Markieff Morris", "BOS")];
  assert.equal(resolveAssister("Marc Morris", roster, new Map())?.person_id, 6);
  assert.equal(resolveAssister("Mark Morris", roster, new Map())?.person_id, 7);
});

test("an initial plus surname works the same way", () => {
  const roster = [P(8, "Giannis Antetokounmpo", "MIL"), P(9, "Thanasis Antetokounmpo", "MIL")];
  assert.equal(resolveAssister("G. Antetokounmpo", roster, new Map())?.person_id, 8);
  assert.equal(resolveAssister("T. Antetokounmpo", roster, new Map())?.person_id, 9);
});

test("an ambiguous name resolves to nothing rather than to a coin flip", () => {
  const roster = [P(10, "Jrue Holiday", "BOS"), P(11, "Justin Holiday", "BOS")];
  assert.equal(resolveAssister("Holiday", roster, new Map()), null);
});

test("the alias map is derived from the game, not typed in", () => {
  // Enes Kanter appears as "Kanter" in descriptions and as his current name,
  // Enes Freedom, in the roster. His OWN events carry both the text and the
  // person id, so the game states the alias itself.
  const events = {
    person_id: [202683, 202683, 0],
    description: [
      "Kanter REBOUND (Off:0 Def:3)",
      "Kanter 2' Layup (4 PTS)",
      "CELTICS Rebound",
    ],
  };
  const aliases = descriptionAliases(events);
  assert.equal(aliases.get("kanter"), 202683);
  const roster = [P(202683, "Enes Freedom", "BOS")];
  assert.equal(resolveAssister("Kanter", roster, aliases)?.person_id, 202683);
  // And with no alias map it correctly fails rather than guessing.
  assert.equal(resolveAssister("Kanter", roster, new Map()), null);
});

test("an alias pointing at two people is discarded", () => {
  const events = {
    person_id: [1, 2],
    description: ["Smith 2' Layup (2 PTS)", "Smith REBOUND (Off:1 Def:0)"],
  };
  assert.equal(descriptionAliases(events).has("smith"), false);
});

test("an alias is only used for a player on the shooting team", () => {
  const aliases = new Map([["kanter", 202683]]);
  assert.equal(resolveAssister("Kanter", [P(999, "Someone Else", "DEN")], aliases), null);
});

test("assists accumulate through the game and start at zero", () => {
  const players = {
    1: P(1, "Aaron Gordon", "DEN"),
    2: P(2, "Michael Porter Jr.", "DEN"),
  };
  const events = {
    action_type: ["Made Shot", "Missed Shot", "Made Shot", "Made Shot"],
    team: ["DEN", "DEN", "DEN", "DEN"],
    person_id: [2, 2, 2, 1],
    description: [
      "Porter Jr. 1' Cutting Layup Shot (2 PTS) (Gordon 1 AST)",
      "MISS Porter Jr. 25' 3PT Jump Shot",
      "Porter Jr. 3PT Jump Shot (5 PTS) (Gordon 2 AST)",
      "Gordon 1' Dunk (2 PTS)",
    ],
  };
  const timeline = assistTimeline(events, players);
  assert.equal(pointsAt(timeline, 1, 0), 1);
  assert.equal(pointsAt(timeline, 1, 1), 1, "a miss adds nothing");
  assert.equal(pointsAt(timeline, 1, 2), 2);
  assert.equal(pointsAt(timeline, 1, 3), 2, "an unassisted dunk adds nothing");
  assert.equal(pointsAt(timeline, 2, 3), 0, "the shooter is not the assister");
});

test("the count comes from the events, never from the feed's running (N AST)", () => {
  // Same reasoning as points: a running counter can be wrong, a count of
  // events cannot disagree with itself.
  const events = {
    action_type: ["Made Shot"],
    team: ["DEN"],
    person_id: [2],
    description: ["Porter Jr. 3PT (12 PTS) (Gordon 9 AST)"],
  };
  const timeline = assistTimeline(events, { 1: P(1, "Aaron Gordon", "DEN") });
  assert.equal(pointsAt(timeline, 1, 0), 1, "one event is one assist, not nine");
});

test("the assisting label is read off the description", () => {
  assert.equal(assistLabel("Porter Jr. 1' Layup (2 PTS) (Gordon 1 AST)"), "Gordon");
  assert.equal(assistLabel("MISS Tatum 25' 3PT Jump Shot"), null);
});

test("stripSuffix leaves a plain surname alone", () => {
  assert.equal(stripSuffix("morris sr"), "morris");
  assert.equal(stripSuffix("williams iii"), "williams");
  assert.equal(stripSuffix("holiday"), "holiday");
});
