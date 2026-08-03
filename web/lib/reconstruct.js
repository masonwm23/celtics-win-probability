/**
 * Play reconstruction: fixed schematic slots plus verified shot geometry.
 *
 * WHAT IS REAL AND WHAT IS A DIAGRAM
 * ----------------------------------
 * Public NBA play-by-play carries ONE coordinate per shot attempt and nothing
 * else. Measured directly on game 0022300906: 174 of 490 events carry a
 * coordinate, and the only action types that ever do are `Made Shot` and
 * `Missed Shot`. Continuous player-and-ball tracking is Second Spectrum data,
 * available to teams and broadcasters, not published.
 *
 * So the ten players sit in FIXED SCHEMATIC SLOTS. They do not move, they are
 * not spaced by a guess about this possession, and no path is drawn between
 * them. A slot is a place to put a name, the way a team sheet is.
 *
 * Three layers, and every position carries its own:
 *
 *   VERIFIED    the recorded shot coordinate, and the ring it travels to.
 *   RULE        the free throw line. Known from the rulebook, never observed:
 *               the feed writes 0,0 for free throws like every non-shot event.
 *   SCHEMATIC   the ten lineup slots. A diagram, not a position.
 *
 * An earlier version estimated player spacing per possession, animated
 * movement between events and drew pass trajectories. All of that is gone. It
 * looked like tracking data, and it was not.
 *
 * There is no randomness in this module. The same lineup always draws the same
 * diagram, which is the only sense in which a seating plan can be correct.
 *
 * Coordinates are the NBA's own system: tenths of a foot, hoop at the origin,
 * matching lib/court.js. No scaling fudge anywhere.
 */

import { COURT, foldToHalfCourt } from "./court.js";
import { describePlay } from "./playby.js";

export const LAYER = {
  VERIFIED: "verified",
  RULE: "rule",
  SCHEMATIC: "schematic",
};

/** Positions that are known rather than drawn. */
export const FIXED_LAYERS = new Set([LAYER.VERIFIED, LAYER.RULE]);

export const DISCLOSURE =
  "Player locations are schematic. Shot location, lineup, clock, score and " +
  "event are verified.";

// ---------------------------------------------------------------------------
// VERIFIED: read straight off the event record
// ---------------------------------------------------------------------------

/**
 * The assisting player's name as the feed wrote it, or null.
 *
 * Descriptions look like:
 *   "Porter Jr. 1' Cutting Layup Shot (2 PTS) (Gordon 1 AST)"
 * The assist is a recorded fact. Where the pass travelled is not, which is why
 * this returns a name and the interface draws a label rather than a path.
 */
export function assistFrom(description) {
  if (typeof description !== "string") return null;
  const match = description.match(/\(([^()]+?)\s+\d+\s+AST\)/);
  return match ? match[1].trim() : null;
}

const SUFFIXES = new Set(["jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"]);

/**
 * The name to put on a marker or in an assist label.
 *
 * Not simply the last token: "Michael Porter Jr." would display as "Jr.", and
 * so would "Larry Nance Jr.", which is how two different players end up
 * looking identical.
 */
export function lastName(name) {
  const parts = String(name || "").trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return "";
  const last = parts[parts.length - 1];
  if (parts.length > 1 && SUFFIXES.has(last.toLowerCase())) {
    return `${parts[parts.length - 2]} ${last}`;
  }
  return last;
}

/**
 * Find the player a feed label refers to, within a lineup.
 *
 * Returns null rather than a guess when nothing, or more than one thing, fits.
 */
export function findPlayerByLabel(label, lineup, players) {
  if (!label || !Array.isArray(lineup)) return null;
  const wanted = label.toLowerCase().replace(/\./g, "").trim();
  const hits = lineup
    .map((id) => players?.[id])
    .filter(Boolean)
    .filter((p) => {
      const name = String(p.name || "").toLowerCase().replace(/\./g, "");
      return name === wanted || name.endsWith(` ${wanted}`);
    });
  return hits.length === 1 ? hits[0] : null;
}

/** Real win probability before and after an event. */
export function probabilityChange(events, index) {
  const wp = events?.wp;
  if (!Array.isArray(wp) || index < 0 || index >= wp.length) return null;
  const after = wp[index];
  const before = index > 0 ? wp[index - 1] : wp[index];
  return { before, after, delta: after - before, layer: LAYER.VERIFIED };
}

/**
 * The three probability figures the interface shows, rounded ONCE.
 *
 * The card shows Before, After and Change side by side, and a reader adds them
 * up. Rounding each independently breaks that: a true step of 58.85 to 58.00
 * prints as "58.8", "58.0" and "-0.9", which reads as an arithmetic error and
 * costs the panel more credibility than the 0.05 of precision is worth.
 *
 * So the change is computed FROM the rounded endpoints. It is the change to
 * the precision displayed, and the three numbers on screen always agree.
 * `raw` keeps the unrounded delta for anything that needs full precision.
 */
export function displayedChange(events, index) {
  const change = probabilityChange(events, index);
  if (!change) return null;
  const before = Number((change.before * 100).toFixed(1));
  const after = Number((change.after * 100).toFixed(1));
  const points = Number((after - before).toFixed(1));
  return {
    before,
    after,
    points,
    raw: change.delta * 100,
    direction: points > 0 ? "up" : points < 0 ? "down" : "flat",
  };
}

export const SHOT_ACTIONS = new Set(["Made Shot", "Missed Shot"]);
export const FREE_THROW_ACTION = "Free Throw";

/**
 * A short description of what happened.
 *
 * Delegates to lib/playby.js, which parses the feed's own description rather
 * than collapsing everything to the action type. Kept here as a thin wrapper
 * because several call sites already import it, and because having two
 * functions that answer "what was this play" is how they end up disagreeing.
 */
export function playResult(event, options) {
  return describePlay(event, options).label;
}

/** The free throw line. Fixed by the rules, never recorded by the feed. */
export const FREE_THROW_SPOT = { x: 0, y: COURT.freeThrowLineY };

/** Events where nothing is in play. */
export const DEAD_BALL = new Set([
  "Substitution",
  "Timeout",
  "Instant Replay",
  "period",
  "Ejection",
]);

/**
 * Did the attempt go in?
 *
 * For a field goal the feed says so outright. For a FREE THROW it does not:
 * `shot_result` is empty on every one of the 47 free throws in game 0022300906,
 * and the outcome lives in the description as a leading "MISS". Reading
 * `shot_result` alone drew a made free throw with the miss symbol while the
 * card beside it said "Made free throw", which is the kind of contradiction
 * that makes a viewer stop trusting the whole panel.
 */
export function shotWasMade(event) {
  if (event?.action_type === FREE_THROW_ACTION) {
    return !/^MISS\b/i.test(String(event?.description || "").trim());
  }
  return event?.shot_result === "Made";
}

/** Does this event carry a real, usable shot coordinate? */
export function hasShotCoordinates(event) {
  if (!event || !SHOT_ACTIONS.has(event.action_type)) return false;
  // (0, 0) is a missing coordinate, not a shot from inside the ring.
  return !(event.loc_x === 0 && event.loc_y === 0);
}

/** The shot's real location, folded onto the attacking half court. */
export function shotOrigin(event) {
  if (!hasShotCoordinates(event)) return null;
  const { x, y } = foldToHalfCourt(event.loc_x, event.loc_y);
  return { x, y, layer: LAYER.VERIFIED };
}

function arcLift(origin) {
  return Math.min(120, 30 + Math.hypot(origin.x, origin.y - COURT.hoopY) * 0.28);
}

/**
 * Verified endpoints, drawn arc.
 *
 * Both ends are real: the recorded shot location and the ring. The curve
 * between them is a drawing convention, which is why the height is a plain
 * function of distance rather than anything pretending to be physics.
 */
export function shotArcPath(origin) {
  if (!origin) return null;
  const lift = arcLift(origin);
  return `M ${origin.x} ${origin.y} Q ${origin.x / 2} ${
    (origin.y + COURT.hoopY) / 2 - lift
  } 0 ${COURT.hoopY}`;
}

/**
 * A point along that arc, for animating the ball from the shot to the ring.
 *
 * Quadratic Bezier at parameter t. Endpoints real, path conventional.
 */
export function pointOnArc(origin, t) {
  if (!origin) return null;
  const s = Math.max(0, Math.min(1, t));
  const cx = origin.x / 2;
  const cy = (origin.y + COURT.hoopY) / 2 - arcLift(origin);
  const inv = 1 - s;
  return {
    x: inv * inv * origin.x + 2 * inv * s * cx,
    y: inv * inv * origin.y + 2 * inv * s * cy + s * s * COURT.hoopY,
  };
}

/** Where along the line to try putting the label, in order of preference. */
export const LABEL_STOPS = [0.36, 0.28, 0.44, 0.22, 0.5, 0.6, 0.68, 0.16, 0.76, 0.86];

/** How far off the line the label sits, and how much room it needs. */
export const LABEL_OFFSETS = [15, 22, 30, 38, 48, 60, 74, 90];
export const LABEL_CLEARANCE = 34;

/**
 * Half the drawn width of the words "schematic link", in court units.
 *
 * The label is a BAR, not a point. Checking only its centre is how it kept
 * landing with its tail written through a player's name: the centre cleared
 * and the last five characters did not.
 */
export const LABEL_HALF_LENGTH = 42;

/**
 * Below this, no link is drawn at all.
 *
 * A shot taken within 13 feet of the shooter's own slot is already beside that
 * player on the diagram, so a stub of a line with fourteen characters of label
 * on it is noise. There is also nowhere to put the words: "schematic link" is
 * 84 units wide, so on a shorter line the label necessarily overhangs one end
 * and lands on the marker it started from.
 */
export const LINK_MIN_LENGTH = 130;

/** Is the link worth drawing between these two points? */
export function shouldDrawLink(from, to) {
  if (!from || !to) return false;
  return Math.hypot(to.x - from.x, to.y - from.y) >= LINK_MIN_LENGTH;
}

/**
 * The first position along the line that clears every marker.
 *
 * Both sides of the line are tried at each stop. A straight free throw runs
 * from the shooter's slot to the free throw line THROUGH the defending point
 * guard's slot, so a fixed midpoint put the words across a player's circle;
 * that is the whole reason this search exists. Falls back to the first stop
 * rather than dropping the label, because the label is the honest part.
 */
export function labelSpot(x1, y1, x2, y2, avoid = []) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const length = Math.hypot(dx, dy) || 1;
  const px = -dy / length;
  const py = dx / length;

  // Every obstacle is an ellipse. A marker is a circle, but a NAME is wide and
  // short: "Caldwell-Pope" is six times the width of the circle above it, and
  // treating it as a circle is how the words ended up written through it.
  //
  // Distances are NORMALISED: 1 means exactly touching the obstacle's boundary,
  // more than 1 means clear. That lets the search rank candidates instead of
  // taking the first that happens to fit, which is what it takes to keep the
  // label off ten circles and ten names at once.
  const room = (x, y) =>
    Math.min(
      ...avoid.map((a) => {
        const rx = a.rx ?? LABEL_CLEARANCE;
        const ry = a.ry ?? LABEL_CLEARANCE;
        return Math.hypot((a.x - x) / rx, (a.y - y) / ry);
      })
    );

  // Five samples along the label, so both ends are tested and not just the
  // middle. The label runs parallel to the line, so the line's own unit vector
  // is the text direction.
  const ux = dx / length;
  const uy = dy / length;
  const SAMPLES = [-1, -0.5, 0, 0.5, 1];
  const barRoom = (x, y) =>
    Math.min(
      ...SAMPLES.map((k) =>
        room(x + ux * LABEL_HALF_LENGTH * k, y + uy * LABEL_HALF_LENGTH * k)
      )
    );

  if (!avoid.length) {
    return { x: x1 + dx * 0.36 + px * LABEL_OFFSETS[0], y: y1 + dy * 0.36 + py * LABEL_OFFSETS[0] };
  }

  let best = null;
  for (const offset of LABEL_OFFSETS) {
    for (const t of LABEL_STOPS) {
      for (const side of [-1, 1]) {
        const x = x1 + dx * t + px * offset * side;
        const y = y1 + dy * t + py * offset * side;
        // A small penalty for drifting off the line, so among positions that
        // all clear, the one nearest the line wins and the label still reads
        // as belonging to it.
        const score = barRoom(x, y) - offset / 400;
        if (!best || score > best.score) best = { x, y, score };
      }
    }
  }
  return { x: best.x, y: best.y };
}

/**
 * What to write under each circle.
 *
 * Surnames, until two players ON THE FLOOR AT ONCE share one. Denver's Justin
 * Holiday and Boston's Jrue Holiday were on court together on 2024-03-07 and
 * the diagram labelled both of them "Holiday", which is worse than useless:
 * the viewer reads two players as one.
 *
 * A first initial does not save it either, because both are J. So a collision
 * falls back to the full name. It is longer, and it is unambiguous, and there
 * is no third option that is both.
 *
 * The same problem in the substitution feed is handled in src/names.py with a
 * variable-length first-name prefix ("Marc" and "Mark" for the Morris twins).
 * That is the right answer for parsing free text, where the feed chose the
 * prefix. Here we own the label, so we can just print the name.
 */
export function displayNames(entries = []) {
  const counts = new Map();
  for (const entry of entries) {
    const last = lastName(entry?.player?.name);
    if (!last) continue;
    counts.set(last, (counts.get(last) || 0) + 1);
  }

  const labels = new Map();
  for (const entry of entries) {
    const player = entry?.player;
    if (!player) continue;
    const last = lastName(player.name);
    labels.set(
      player.person_id,
      counts.get(last) > 1 ? String(player.name).trim() : last
    );
  }
  return labels;
}

/**
 * Everything on the diagram the link's label has to stay off.
 *
 * Two obstacles per player: the circle, and the name beside it. The name's
 * width is estimated from its character count at the drawn font size, which is
 * approximate but errs wide.
 */
export const NAME_CHAR_WIDTH = 7;

export function slotObstacles(entries = [], labels = null) {
  const out = [];
  for (const e of entries) {
    if (!e) continue;
    out.push({ x: e.x, y: e.y });
    // The width estimate has to use the name that is actually DRAWN. A
    // collision turns "Holiday" into "Justin Holiday", more than twice as
    // wide, and sizing the obstacle off the short one would let the link
    // label land on top of it.
    const text = labels?.get(e.player?.person_id) ?? lastName(e.player?.name || "");
    if (!text) continue;
    const half = (text.length * NAME_CHAR_WIDTH) / 2;
    const cx =
      e.anchor === "start"
        ? e.labelX + half
        : e.anchor === "end"
          ? e.labelX - half
          : e.labelX;
    out.push({ x: cx, y: e.labelY, rx: half + 14, ry: 20 });
  }
  return out;
}

// ---------------------------------------------------------------------------
// SCHEMATIC: fixed lineup slots
// ---------------------------------------------------------------------------

/**
 * Ten fixed slots, five a side.
 *
 * These never change, for any event, in any game. The attacking five sit
 * further out and the defending five nearer the ring, so the diagram reads
 * like a team sheet rather than a formation anybody observed.
 *
 * Every pair is at least 58 units apart, comfortably more than the 44 two
 * markers need, so nothing overlaps and no de-collision pass is required.
 */
export const SLOTS = {
  // `lx`, `ly` and `anchor` place the name. Because the slots never move,
  // label placement is settled once here rather than by a de-collision pass at
  // render time.
  //
  // Compressed vertically in the 2026-07-31 layout pass. A half court is very
  // nearly square, so a court drawn to fill 65% of a laptop's width was taller
  // than the viewport and pushed the controls off screen. The deepest label now
  // sits at 290 rather than 362, which lets the drawing crop at 312 and come
  // out at roughly 3:2 instead of 5:4. Every coordinate a SHOT uses is
  // untouched: this moves the seating plan, not the measurements.
  //
  // Four bands rather than two rows, which is what buys the vertical room:
  // defending bigs, attacking bigs, defending perimeter, attacking perimeter.
  // Closest pair is 76 units, comfortably clear of the 46 two 23-unit markers
  // need. The tests pin both that and the label clearances.
  offense: [
    { key: "PG", x: 0, y: 272, lx: 0, ly: 305, anchor: "middle" },
    { key: "SG", x: -210, y: 258, lx: -210, ly: 291, anchor: "middle" },
    { key: "SF", x: 210, y: 258, lx: 210, ly: 291, anchor: "middle" },
    { key: "PF", x: -120, y: 118, lx: -150, ly: 124, anchor: "end" },
    { key: "C", x: 120, y: 118, lx: 150, ly: 124, anchor: "start" },
  ],
  defense: [
    { key: "PG", x: 0, y: 196, lx: 0, ly: 160, anchor: "middle" },
    { key: "SG", x: -165, y: 196, lx: -165, ly: 160, anchor: "middle" },
    { key: "SF", x: 165, y: 196, lx: 165, ly: 160, anchor: "middle" },
    { key: "PF", x: -85, y: 40, lx: -85, ly: 6, anchor: "middle" },
    { key: "C", x: 85, y: 40, lx: 85, ly: 6, anchor: "middle" },
  ],
};

function rankOf(player) {
  const raw = String(player?.coarse_position || player?.position || "")
    .toUpperCase();
  if (raw.startsWith("G")) return raw.includes("F") ? 1 : 0;
  if (raw.startsWith("F")) return raw.includes("C") ? 3 : 2;
  if (raw.startsWith("C")) return 4;
  return 2; // unknown sits in the middle rather than being called a centre
}

/**
 * Put five players in five slots, guards outside and bigs inside.
 *
 * Sorted by listed position then by person id, so the result never depends on
 * object key order or on anything that could differ between machines.
 */
export function assignSlots(lineup, players, slots) {
  return (lineup || [])
    .map((id) => players?.[id])
    .filter(Boolean)
    .sort((a, b) => {
      const byRank = rankOf(a) - rankOf(b);
      if (byRank !== 0) return byRank;
      return String(a.person_id).localeCompare(String(b.person_id));
    })
    .slice(0, slots.length)
    .map((player, i) => ({
      player,
      x: slots[i].x,
      y: slots[i].y,
      labelX: slots[i].lx,
      labelY: slots[i].ly,
      anchor: slots[i].anchor,
      slot: slots[i].key,
      layer: LAYER.SCHEMATIC,
      note: "schematic lineup slot, not a court position",
    }));
}

// ---------------------------------------------------------------------------
// The reconstruction
// ---------------------------------------------------------------------------

/**
 * Everything the court needs for one event.
 *
 * `shot` is the only thing on the floor with a real location. `offense` and
 * `defense` are the seating plan. `involved` names who the event was about, so
 * the interface can highlight them without moving them.
 */
export function reconstruct({
  event,
  offenseLineup = [],
  defenseLineup = [],
  players = {},
}) {
  const isFreeThrow = event?.action_type === FREE_THROW_ACTION;
  const deadBall = DEAD_BALL.has(event?.action_type);

  const origin = isFreeThrow
    ? { ...FREE_THROW_SPOT, layer: LAYER.RULE }
    : shotOrigin(event);

  const offense = assignSlots(offenseLineup, players, SLOTS.offense);
  const defense = assignSlots(defenseLineup, players, SLOTS.defense);

  const shooter = players?.[event?.person_id] || null;
  const assister = findPlayerByLabel(
    assistFrom(event?.description),
    offenseLineup,
    players
  );

  const slotOf = (person) =>
    person
      ? offense.find((e) => e.player.person_id === person.person_id) ||
        defense.find((e) => e.player.person_id === person.person_id) ||
        null
      : null;

  const shooterSlot = deadBall ? null : slotOf(shooter);
  const assisterSlot = deadBall ? null : slotOf(assister);

  return {
    deadBall,
    isFreeThrow,
    offense,
    defense,
    shot: origin
      ? {
          origin,
          arc: shotArcPath(origin),
          made: shotWasMade(event),
          value: event?.shot_value ?? 0,
          layer: origin.layer,
          note:
            origin.layer === LAYER.RULE
              ? "rule-defined location: the free throw line. Not tracked, " +
                "not recorded by the feed."
              : "recorded shot location",
        }
      : null,
    involved: {
      shooter: deadBall ? null : shooter,
      shooterSlot,
      assister: deadBall ? null : assister,
      assisterSlot,
      // Only when both ends are on the floor, so the label names two people
      // who were actually there.
      label:
        shooterSlot && assisterSlot
          ? `${lastName(assister.name)} → ${lastName(shooter.name)}`
          : null,
    },
    disclosure: DISCLOSURE,
  };
}
