import test from "node:test";
import assert from "node:assert/strict";

import { CELTICS_ABBREV, scoreLine, scoreSides } from "./teams.js";

/**
 * Boston play 318 of their 636 games away. Any rule that puts them on the left
 * unconditionally is wrong half the time, and wrong in a way nobody notices:
 * the numbers are real, they are just attached to the wrong teams.
 */

const AWAY = {
  opponent: "DEN",
  opponent_name: "Denver Nuggets",
  opponent_logo: "https://cdn/den.svg",
  celtics_logo: "https://cdn/bos.svg",
  celtics_name: "Boston Celtics",
  celtics_is_home: false,
};

const HOME = { ...AWAY, celtics_is_home: true };

test("the home team is on the left when Boston are at home", () => {
  const sides = scoreSides({ meta: HOME, celticsScore: 104, opponentScore: 101 });
  assert.equal(sides.left.abbrev, "BOS");
  assert.equal(sides.right.abbrev, "DEN");
});

test("the home team is on the left when Boston are AWAY", () => {
  // The case the current-play card used to get wrong.
  const sides = scoreSides({ meta: AWAY, celticsScore: 109, opponentScore: 115 });
  assert.equal(sides.left.abbrev, "DEN");
  assert.equal(sides.right.abbrev, "BOS");
});

test("each score travels with its own team, not with its position", () => {
  const away = scoreSides({ meta: AWAY, celticsScore: 109, opponentScore: 115 });
  assert.equal(away.left.score, 115, "Denver's score sits under Denver");
  assert.equal(away.right.score, 109, "Boston's score sits under Boston");

  const home = scoreSides({ meta: HOME, celticsScore: 104, opponentScore: 101 });
  assert.equal(home.left.score, 104);
  assert.equal(home.right.score, 101);
});

test("each logo travels with its own team", () => {
  const away = scoreSides({ meta: AWAY, celticsScore: 1, opponentScore: 2 });
  assert.equal(away.left.logo, "https://cdn/den.svg");
  assert.equal(away.right.logo, "https://cdn/bos.svg");

  const home = scoreSides({ meta: HOME, celticsScore: 1, opponentScore: 2 });
  assert.equal(home.left.logo, "https://cdn/bos.svg");
  assert.equal(home.right.logo, "https://cdn/den.svg");
});

test("exactly one side is Boston, and exactly one side is at home", () => {
  for (const meta of [HOME, AWAY]) {
    const sides = scoreSides({ meta, celticsScore: 90, opponentScore: 88 });
    const both = [sides.left, sides.right];
    assert.equal(both.filter((s) => s.isCeltics).length, 1);
    assert.equal(both.filter((s) => s.isHome).length, 1);
    // And the left side is always the home one. That is the whole rule.
    assert.equal(sides.left.isHome, true);
  }
});

test("the written line reads in the same order as the logos", () => {
  assert.equal(
    scoreLine(scoreSides({ meta: AWAY, celticsScore: 101, opponentScore: 104 })),
    "DEN 104 – 101 BOS"
  );
  assert.equal(
    scoreLine(scoreSides({ meta: HOME, celticsScore: 104, opponentScore: 101 })),
    "BOS 104 – 101 DEN"
  );
});

test("a payload with no logos still produces a usable pair", () => {
  const sides = scoreSides({
    meta: { opponent: "NYK", celtics_is_home: false },
    celticsScore: 101,
    opponentScore: 104,
  });
  assert.equal(sides.left.abbrev, "NYK");
  assert.equal(sides.left.logo, null, "a missing logo is null, not an empty string");
  assert.equal(sides.right.abbrev, CELTICS_ABBREV);
  assert.equal(scoreLine(sides), "NYK 104 – 101 BOS");
});

test("scores are numbers, so a string from the payload does not concatenate", () => {
  const sides = scoreSides({ meta: AWAY, celticsScore: "9", opponentScore: "10" });
  assert.equal(sides.left.score + sides.right.score, 19);
});
