/**
 * Shot marker geometry, defined ONCE.
 *
 * The court draws a made shot as a filled green circle and a miss as an orange
 * cross. The legend has to show the same two symbols, and for a while it did
 * not: the legend's "missed" swatch was a single diagonal slash built out of a
 * CSS border, so a viewer matching the key to the court was matching a slash
 * against a cross.
 *
 * Fixing the CSS would have fixed that instance and left the next one open.
 * Instead both the court and the legend now build their marks from the
 * functions here and wear the same class names, so the colour, the stroke
 * weight, the round caps and the arm-to-stroke ratio cannot drift apart. The
 * legend renders in a 40-unit box with the same arm length as the court and is
 * scaled down by the SVG viewport, which keeps the proportions identical by
 * construction rather than by somebody remembering to update two numbers.
 *
 * Colour lives in CSS, on these class names, so the theme stays in one file.
 */

/** Class names. Both surfaces use these, which is what the tests pin. */
export const MADE_CLASS = "shotspot";
export const MISS_CLASS = "shotspot shotspot--miss";

/** The court's own sizes, in court units (tenths of a foot). */
export const MADE_RADIUS = 10;
export const MISS_ARM = 10;

/** The legend's box. Same arm length, so the shape is the court's shape. */
export const LEGEND_BOX = 40;

/**
 * The two crossing lines of a miss marker.
 *
 * Returned as data rather than markup so a test can check the shape without a
 * browser. Both arms are the same length and meet at right angles, which is
 * what makes it read as an X rather than as a slash or a chevron.
 */
export function missMarkLines(cx, cy, arm = MISS_ARM) {
  return [
    { x1: cx - arm, y1: cy - arm, x2: cx + arm, y2: cy + arm },
    { x1: cx - arm, y1: cy + arm, x2: cx + arm, y2: cy - arm },
  ];
}

/** The legend's copy of the miss marker, centred in its box. */
export function legendMissLines(arm = MISS_ARM) {
  return missMarkLines(LEGEND_BOX / 2, LEGEND_BOX / 2, arm);
}

/**
 * Half the drawn length of a line: the arm of the X, measured diagonally.
 *
 * Note this is NOT `MISS_ARM`, which is the extent along each axis. The drawn
 * arm is longer by root two. Two different quantities, deliberately not given
 * the same name.
 */
export function armLength(line) {
  return Math.hypot(line.x2 - line.x1, line.y2 - line.y1) / 2;
}
