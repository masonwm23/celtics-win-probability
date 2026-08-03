import test from "node:test";
import assert from "node:assert/strict";

import { biggestSwings, changeSummary, deltaAt, scoringRun, swingSize } from "./moments.js";

const GAME = {
  //            0     1     2     3     4     5     6
  wp:        [0.50, 0.54, 0.53, 0.71, 0.70, 0.55, 0.56],
  period:    [1, 1, 1, 2, 2, 4, 4],
  clock:     ["PT12M00.00S","PT11M40.00S","PT11M20.00S","PT06M00.00S",
              "PT05M40.00S","PT02M00.00S","PT01M40.00S"],
  margin:    [0, 2, 2, 5, 5, 2, 4],
  celtics_score:  [0, 2, 2, 5, 5, 5, 7],
  opponent_score: [0, 0, 0, 0, 0, 3, 3],
  description: ["Start", "Tatum 2PT", "Rebound", "Brown 3PT", "Foul", "Jokic 3PT", "White 2PT"],
};

test("a probability change is measured against the play before it", () => {
  assert.ok(Math.abs(deltaAt(GAME, 1) - 4) < 1e-9);
  assert.ok(Math.abs(deltaAt(GAME, 5) - -15) < 1e-9);
  assert.equal(deltaAt(GAME, 0), 0, "the first event has nothing before it");
});

test("the biggest swings are ranked by size, either direction", () => {
  const rows = biggestSwings(GAME, { limit: 3 });
  assert.deepEqual(rows.map((r) => r.index), [3, 5, 1]);
  assert.equal(rows[0].towards, "bos");
  assert.equal(rows[1].towards, "opp");
  assert.match(rows[0].label, /^Q2 /);
});

test("every swing row points at a real event", () => {
  for (const row of biggestSwings(GAME, { limit: 10 })) {
    assert.ok(row.index >= 1 && row.index < GAME.wp.length);
    assert.equal(row.description, GAME.description[row.index]);
  }
});

test("swing size is a percentile within THIS game, and says so", () => {
  // Event 3 is the largest move in the fixture (+18), event 4 the smallest.
  const biggest = swingSize(GAME, 3);
  const smallest = swingSize(GAME, 4);
  assert.equal(biggest.of, GAME.wp.length - 1, "measured against every other play");
  assert.ok(biggest.percentile > smallest.percentile);
  assert.equal(smallest.label, "Small");
  assert.ok(["Large", "Very large"].includes(biggest.label), biggest.label);
});

test("the bands are ordered, so a bigger swing never reads smaller", () => {
  const ranked = [1, 2, 3, 4, 5, 6]
    .map((i) => swingSize(GAME, i))
    .sort((a, b) => a.size - b.size);
  for (let i = 1; i < ranked.length; i += 1) {
    assert.ok(ranked[i].percentile >= ranked[i - 1].percentile,
      `size ${ranked[i].size} ranked below size ${ranked[i - 1].size}`);
  }
});

test("a scoring run counts consecutive points by one team", () => {
  // Boston scored on events 1 and 3 with nothing for the opponent between.
  const run = scoringRun(GAME, 4, "DEN");
  assert.equal(run.team, "bos");
  assert.equal(run.abbrev, "BOS");
  assert.equal(run.points, 5);
  assert.equal(run.startedAt, 1);
});

test("a run ends when the other team scores", () => {
  const run = scoringRun(GAME, 5, "DEN");
  assert.equal(run.team, "opp");
  assert.equal(run.abbrev, "DEN");
  assert.equal(run.points, 3);
});

test("a run resumes for the other team after they answer", () => {
  const run = scoringRun(GAME, 6, "DEN");
  assert.equal(run.team, "bos");
  assert.equal(run.points, 2, "only the points since the opponent last scored");
});

test("there is no run before anyone has scored", () => {
  assert.equal(scoringRun(GAME, 0, "DEN"), null);
});

test("the change summary states what changed, never why", () => {
  const s = changeSummary(GAME, 5, "DEN");
  assert.match(s.text, /score margin went from BOS by 5 to BOS by 2/i);
  assert.match(s.text, /left in Q4/);
  assert.match(s.text, /moved down 15\.0 percentage points/);
  // The words that would make it a causal claim must not appear.
  for (const word of [" because ", " caused ", " due to ", " thanks to ", " led to "]) {
    assert.ok(!s.text.toLowerCase().includes(word), `"${word.trim()}" is a causal claim`);
  }
  assert.match(s.caveat, /not what caused it/);
});

test("an unchanged margin is described as unchanged rather than omitted", () => {
  const s = changeSummary(GAME, 4, "DEN");
  assert.match(s.text, /did not change/);
});

test("a level score is described as level, not as a lead of zero", () => {
  const s = changeSummary({ ...GAME, margin: [0, 0, 0, 0, 0, 0, 0] }, 2, "DEN");
  assert.match(s.text, /level/);
});

test("the opponent is named from the payload, never assumed", () => {
  const s = changeSummary(GAME, 5, "PHX");
  assert.ok(!s.text.includes("DEN"));
  const flipped = changeSummary({ ...GAME, margin: GAME.margin.map((m) => -m) }, 5, "PHX");
  assert.match(flipped.text, /PHX by/);
});
