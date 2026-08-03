import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import {
  UNDERDOG,
  comebackSummary,
  comebackWins,
  deficitMoment,
  hasDeficitData,
  longShot,
} from "./comebacks.js";

/**
 * Two kinds of test here.
 *
 * The first kind uses small hand-written rows for the rules: wins only, the
 * 50% cut, ordering, ties, and a row with no deficit field. Those are the
 * cases that are easy to get wrong and hard to see going wrong.
 *
 * The second kind runs against the real 636-game index, so the rules are also
 * checked against every game in the study rather than against six rows I chose.
 * The headline is externally checkable: Boston were at 0.18% out of fold in
 * Phoenix on 8 November 2018 and won 116-109.
 */

const row = (over) => ({
  game_id: "0000000000",
  season: "2020-21",
  date: "2021-01-01",
  matchup: "BOS vs. XXX",
  opponent: "XXX",
  celtics_won: true,
  celtics_final: 100,
  opponent_final: 99,
  lowest_wp: 0.2,
  largest_deficit: 10,
  deficit_period: 2,
  deficit_clock: "PT03M58.00S",
  deficit_event: 100,
  ...over,
});

// ---------------------------------------------------------------- the rules

test("a loss never appears, however low the probability went", () => {
  const games = [
    row({ game_id: "win", lowest_wp: 0.4 }),
    row({ game_id: "loss", lowest_wp: 0.001, celtics_won: false }),
  ];
  assert.deepEqual(
    comebackWins(games, "2020-21").map((g) => g.game_id),
    ["win"]
  );
});

test("a win the model never had below even money is not a comeback", () => {
  const games = [
    row({ game_id: "sweated", lowest_wp: 0.49 }),
    row({ game_id: "comfortable", lowest_wp: 0.5 }),
    row({ game_id: "cruise", lowest_wp: 0.71 }),
  ];
  assert.deepEqual(
    comebackWins(games, "2020-21").map((g) => g.game_id),
    ["sweated"]
  );
  // The cut is exactly at 50% and is exclusive.
  assert.equal(UNDERDOG, 0.5);
  // And the excluded wins are counted, not silently dropped.
  assert.equal(comebackSummary(games, "2020-21").neverBehind, 2);
});

test("the lowest probability comes first", () => {
  const games = [
    row({ game_id: "a", lowest_wp: 0.31 }),
    row({ game_id: "b", lowest_wp: 0.002 }),
    row({ game_id: "c", lowest_wp: 0.14 }),
  ];
  assert.deepEqual(
    comebackWins(games, "2020-21").map((g) => g.game_id),
    ["b", "c", "a"]
  );
});

test("a bigger deficit does NOT outrank a lower probability", () => {
  // The distinction the ranking exists to make. A 20-point hole in the second
  // quarter is a bigger deficit and a less improbable win than a 4-point hole
  // with a minute left.
  const games = [
    row({ game_id: "big hole early", largest_deficit: 20, lowest_wp: 0.18 }),
    row({ game_id: "small hole late", largest_deficit: 4, lowest_wp: 0.01 }),
  ];
  assert.deepEqual(
    comebackWins(games, "2020-21").map((g) => g.game_id),
    ["small hole late", "big hole early"]
  );
});

test("equal probabilities break on the larger deficit, not on input order", () => {
  const games = [
    row({ game_id: "shallow", lowest_wp: 0.05, largest_deficit: 6 }),
    row({ game_id: "deep", lowest_wp: 0.05, largest_deficit: 19 }),
  ];
  assert.deepEqual(
    comebackWins(games, "2020-21").map((g) => g.game_id),
    ["deep", "shallow"]
  );
  assert.deepEqual(
    comebackWins([...games].reverse(), "2020-21").map((g) => g.game_id),
    ["deep", "shallow"]
  );
});

test("everything equal breaks on date, so the order is total", () => {
  const games = [
    row({ game_id: "later", date: "2021-03-04" }),
    row({ game_id: "earlier", date: "2021-01-04" }),
  ];
  assert.deepEqual(
    comebackWins(games, "2020-21").map((g) => g.game_id),
    ["earlier", "later"]
  );
});

test("another season's games do not leak into this season's list", () => {
  const games = [
    row({ game_id: "this", lowest_wp: 0.3 }),
    row({ game_id: "other", lowest_wp: 0.001, season: "2017-18" }),
  ];
  assert.deepEqual(
    comebackWins(games, "2020-21").map((g) => g.game_id),
    ["this"]
  );
});

test("an index without the deficit fields still ranks, it just says less", () => {
  // The whole point of ranking on probability: lowest_wp has been in the index
  // since it was first built, so this works before script 42 has ever run.
  const bare = [
    { season: "2020-21", celtics_won: true, lowest_wp: 0.03, date: "2021-01-01",
      game_id: "a", celtics_final: 100, opponent_final: 99 },
    { season: "2020-21", celtics_won: true, lowest_wp: 0.30, date: "2021-01-02",
      game_id: "b", celtics_final: 100, opponent_final: 99 },
  ];
  assert.equal(hasDeficitData(bare), false);
  const ranked = comebackWins(bare, "2020-21");
  assert.deepEqual(ranked.map((g) => g.game_id), ["a", "b"]);
  assert.equal(ranked[0].moment, null, "no deficit moment to show");
  assert.equal(comebackSummary(bare, "2020-21").largestDeficit, 0);
  // A half-built index counts as missing: one bad row would poison the display.
  assert.equal(hasDeficitData([row({}), bare[0]]), false);
  assert.equal(hasDeficitData([row({})]), true);
  assert.equal(hasDeficitData([]), false);
});

test("the probability label keeps a digit that matters at long odds", () => {
  const [tiny, small] = comebackWins(
    [row({ game_id: "t", lowest_wp: 0.0018 }), row({ game_id: "s", lowest_wp: 0.21 })],
    "2020-21"
  );
  assert.equal(tiny.lowestLabel, "0.18%", "two digits below 1%");
  assert.equal(small.lowestLabel, "21.0%", "one digit above it");
});

test("the odds phrasing is a restatement, not an extra claim", () => {
  assert.equal(longShot(0.002), "about 1 in 500");
  assert.equal(longShot(0.25), "about 1 in 4");
  assert.equal(longShot(0.6), null, "not a long shot at all");
  assert.equal(longShot(0.49), null, "1 in 2 says nothing worth saying");
  assert.equal(longShot(0), null);
  assert.equal(longShot(undefined), null);
});

test("the moment reads like a clock, and is null when there was no deficit", () => {
  assert.equal(deficitMoment(row({ deficit_period: 2, deficit_clock: "PT03M58.00S" })),
               "Q2 3:58");
  assert.equal(deficitMoment(row({ deficit_period: 5, deficit_clock: "PT00M45.60S" })),
               "OT 45.6");
  assert.equal(deficitMoment(row({ deficit_period: null, deficit_clock: null })), null);
});

// ------------------------------------------------------- the real 636 games

/**
 * The real index, read from the serving folder this dashboard is fed by.
 *
 * If it is not there these tests say so and skip, rather than passing on an
 * empty list. A green tick from a file that does not exist is worse than a red
 * one. Note that they do NOT require script 42 to have run: the ranking needs
 * only `lowest_wp`, which every index has.
 */
function realIndex() {
  for (const candidate of ["../../data/serving/index.json",
                           "../fixtures/index_real.json"]) {
    try {
      const parsed = JSON.parse(readFileSync(new URL(candidate, import.meta.url)));
      if (Array.isArray(parsed?.games) && parsed.games.length > 100) return parsed.games;
    } catch {
      /* try the next one */
    }
  }
  return null;
}

const REAL = realIndex();
const onReal = { skip: REAL ? false : "no serving index found to test against" };

test("the real index has a probability for every game", onReal, () => {
  assert.equal(REAL.length, 636);
  assert.ok(REAL.every((g) => Number.isFinite(g.lowest_wp)));
});

test("the longest odds in eight seasons are the 0.18% win in Phoenix", onReal, () => {
  const top = comebackWins(REAL, null)[0];
  assert.equal(top.date, "2018-11-08");
  assert.equal(top.opponent, "PHX");
  assert.equal(top.celtics_final, 116);
  assert.equal(top.opponent_final, 109);
  assert.equal(top.lowestLabel, "0.18%");
  assert.equal(top.odds, "about 1 in 556");
});

test("across all 636 games the list is wins only, below 50%, strictly ordered", onReal, () => {
  const ranked = comebackWins(REAL, null);
  assert.ok(ranked.length > 0);
  for (const game of ranked) {
    assert.equal(game.celtics_won, true, `${game.game_id} is not a win`);
    assert.ok(game.lowest_wp < UNDERDOG, `${game.game_id} was never behind`);
    assert.ok(game.lowest_wp >= 0, `${game.game_id} has a negative probability`);
  }
  for (let i = 1; i < ranked.length; i += 1) {
    assert.ok(ranked[i - 1].lowest_wp <= ranked[i].lowest_wp, `out of order at ${i}`);
  }
});

test("the season split adds up: comebacks plus never-behind equals wins", onReal, () => {
  const seasons = [...new Set(REAL.map((g) => g.season))];
  assert.equal(seasons.length, 8);
  let comebacks = 0;
  for (const season of seasons) {
    const s = comebackSummary(REAL, season);
    assert.equal(s.wins + s.losses, s.games);
    assert.equal(s.comebacks + s.neverBehind, s.wins);
    assert.ok(s.games > 0, `${season} has no games`);
    assert.ok(s.comebacks > 0, `${season} has no comeback to show`);
    comebacks += s.comebacks;
  }
  assert.equal(comebacks, comebackWins(REAL, null).length);
  assert.equal(comebackSummary(REAL, null).wins, 413);
  assert.equal(comebackSummary(REAL, null).comebacks, 244);
  assert.equal(comebackSummary(REAL, null).neverBehind, 169);
});

test("no win below 50% is dropped from the list", onReal, () => {
  const ranked = new Set(comebackWins(REAL, null).map((g) => g.game_id));
  for (const game of REAL) {
    if (game.celtics_won && game.lowest_wp < UNDERDOG) {
      assert.ok(ranked.has(game.game_id), `${game.game_id} was dropped`);
    }
  }
});
