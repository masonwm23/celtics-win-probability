/**
 * Half-court geometry, in the NBA's own shot coordinate system.
 *
 * The event table carries `xLegacy` and `yLegacy` straight from the play-by-play
 * feed. Those are TENTHS OF A FOOT with the hoop centre at the origin, which is
 * confirmed by the data itself: a shot recorded at (99, 11) has a stated
 * distance of 10 feet, and hypot(99, 11) / 10 = 9.96.
 *
 * Drawing the court in the same units means shot markers land where the shots
 * were actually taken, with no fudge factor anywhere. Every constant below is a
 * real NBA dimension converted at ten units per foot.
 */

// Court, in units of a tenth of a foot.
export const COURT = {
  // 50 ft wide, so 25 ft either side of the centre line.
  minX: -250,
  maxX: 250,
  // The hoop centre sits 5.25 ft from the baseline: a 4 ft backboard overhang
  // plus 15 inches to the ring centre.
  baselineY: -52.5,
  // A half court is 47 ft from baseline to the division line.
  halfCourtY: 417.5,

  hoopRadius: 7.5, // 9 inch ring radius, 18 inch diameter
  hoopY: 0,
  backboardY: -12.5, // 4 ft from baseline, so 1.25 ft behind the ring centre
  backboardHalfWidth: 30, // a 6 ft board

  // The paint is 16 ft wide and runs 19 ft from the baseline.
  paintHalfWidth: 80,
  freeThrowLineY: 137.5,
  freeThrowCircleRadius: 60, // 12 ft diameter

  // The restricted area is a 4 ft arc under the basket.
  restrictedRadius: 40,

  // Three-point line: 22 ft in the corners, 23.75 ft above the break. The
  // corner segment runs until the arc reaches the same x.
  cornerThreeX: 220,
  arcRadius: 237.5,
};

/** Where the corner three meets the arc, so the two join without a kink. */
export function cornerBreakY() {
  const { cornerThreeX, arcRadius } = COURT;
  return Math.sqrt(arcRadius * arcRadius - cornerThreeX * cornerThreeX);
}

/** The three-point line as one SVG path: corner, arc, corner. */
export function threePointPath() {
  const { cornerThreeX, arcRadius, baselineY } = COURT;
  const breakY = cornerBreakY();
  return [
    `M ${-cornerThreeX} ${baselineY}`,
    `L ${-cornerThreeX} ${breakY}`,
    `A ${arcRadius} ${arcRadius} 0 0 0 ${cornerThreeX} ${breakY}`,
    `L ${cornerThreeX} ${baselineY}`,
  ].join(" ");
}

/**
 * Fold a shot onto the half court the offence was attacking.
 *
 * The feed records both baskets in one coordinate system, so shots at the far
 * end arrive with a large positive y. Anything past the division line is
 * reflected, which is what every public shot chart does and is the only way to
 * put a whole game on one half court.
 */
export function foldToHalfCourt(x, y) {
  if (y > COURT.halfCourtY) {
    return { x: -x, y: 2 * COURT.halfCourtY - y };
  }
  return { x, y };
}

/** Distance in feet, the same way the feed computes `shot_distance`. */
export function shotDistanceFeet(x, y) {
  return Math.hypot(x, y) / 10;
}

export const ACTION_IS_SHOT = new Set(["Made Shot", "Missed Shot"]);

/**
 * Shots taken up to and including an event, most recent first.
 *
 * Only real attempts with coordinates. A shot recorded at exactly (0, 0) is a
 * missing coordinate rather than a shot from inside the ring, so it is dropped
 * instead of being drawn on top of the basket.
 */
export function shotsUpTo(events, upToIndex, { team = null, limit = 400 } = {}) {
  const shots = [];
  for (let i = 0; i <= upToIndex && i < events.action_type.length; i += 1) {
    if (!ACTION_IS_SHOT.has(events.action_type[i])) continue;
    const x = events.loc_x[i];
    const y = events.loc_y[i];
    if (x === 0 && y === 0) continue;
    if (team && events.team[i] !== team) continue;
    const folded = foldToHalfCourt(x, y);
    shots.push({
      i,
      x: folded.x,
      y: folded.y,
      made: events.shot_result[i] === "Made",
      value: events.shot_value[i],
      team: events.team[i],
      personId: events.person_id[i],
      description: events.description[i],
    });
  }
  return shots.slice(-limit);
}
