import test from "node:test";
import assert from "node:assert/strict";

import {
  LEGEND_BOX,
  MADE_CLASS,
  MADE_RADIUS,
  MISS_ARM,
  MISS_CLASS,
  armLength,
  legendMissLines,
  missMarkLines,
} from "./marks.js";

/**
 * The legend has to show the court's symbols, not lookalikes.
 *
 * The bug this file exists for: the court drew a missed shot as a large orange
 * X while the legend showed a single diagonal slash built from a CSS border.
 * Same colour, different shape, so a viewer matching the key against the floor
 * was matching two different things.
 *
 * The fix was to give both surfaces one source for the geometry and one set of
 * class names. These tests pin that, and they run without a browser, so a
 * future edit that re-specifies the legend in CSS fails here rather than
 * quietly on screen.
 */

test("a miss marker is two crossing arms, not a slash", () => {
  const lines = missMarkLines(0, 0);
  assert.equal(lines.length, 2, "a slash is one line; an X is two");

  // Both arms the same length, or it is a chevron.
  assert.equal(armLength(lines[0]), armLength(lines[1]));

  // Perpendicular, or it is a cross-hatch rather than an X.
  const dot =
    (lines[0].x2 - lines[0].x1) * (lines[1].x2 - lines[1].x1) +
    (lines[0].y2 - lines[0].y1) * (lines[1].y2 - lines[1].y1);
  assert.equal(dot, 0, "the two arms must meet at a right angle");
});

test("the marker is centred on the point it marks", () => {
  const lines = missMarkLines(123, -45);
  for (const line of lines) {
    assert.equal((line.x1 + line.x2) / 2, 123);
    assert.equal((line.y1 + line.y2) / 2, -45);
  }
});

test("the legend's X is the court's X, only translated", () => {
  const court = missMarkLines(0, 0);
  const legend = legendMissLines();

  assert.equal(legend.length, court.length);
  for (let i = 0; i < court.length; i += 1) {
    // Identical arm length means identical proportions once the SVG viewport
    // scales the 40-unit box down to 15 pixels. Nothing is re-specified.
    assert.equal(armLength(legend[i]), armLength(court[i]));
    // MISS_ARM is the extent along each AXIS; the drawn arm runs diagonally,
    // so it is longer by root two. Asserting the relationship rather than a
    // magic number keeps the two definitions from silently diverging.
    assert.ok(
      Math.abs(armLength(legend[i]) - MISS_ARM * Math.SQRT2) < 1e-9,
      `arm was ${armLength(legend[i])}, expected ${MISS_ARM * Math.SQRT2}`
    );
  }

  // And centred in its box, so it does not sit off to one side of the swatch.
  for (const line of legend) {
    assert.equal((line.x1 + line.x2) / 2, LEGEND_BOX / 2);
    assert.equal((line.y1 + line.y2) / 2, LEGEND_BOX / 2);
  }
});

test("both surfaces carry the same class names, which is where the colour lives", () => {
  // Colour is one CSS rule on these names. If the legend ever stops using
  // them, it stops being orange with the court and this test says so.
  assert.equal(MISS_CLASS, "shotspot shotspot--miss");
  assert.equal(MADE_CLASS, "shotspot");
  assert.ok(
    MISS_CLASS.split(" ").includes(MADE_CLASS),
    "the miss marker must inherit the made marker's base class so a single " +
      "colour change reaches both"
  );
});

test("the made marker fits inside the legend box with room to breathe", () => {
  assert.ok(MADE_RADIUS * 2 < LEGEND_BOX);
  assert.ok(MISS_ARM * 2 < LEGEND_BOX);
});

test("the legend box is big enough that the stroke is not clipped", () => {
  // Stroke is 4.6 units wide in CSS and centred on the line, so an arm ending
  // at LEGEND_BOX/2 + MISS_ARM needs 2.3 units of margin beyond it.
  const STROKE = 4.6;
  const furthest = LEGEND_BOX / 2 + MISS_ARM + STROKE / 2;
  assert.ok(
    furthest <= LEGEND_BOX + 0.001,
    `the X reaches ${furthest} in a ${LEGEND_BOX} box`
  );
});
