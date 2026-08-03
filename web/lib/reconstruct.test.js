/**
 * Tests for the play reconstruction.
 *
 * Run with:  node --test lib/reconstruct.test.js
 *
 * Node 22's built-in runner and its ESM detection handle this with zero new
 * dependencies, so the repo gains no test framework.
 *
 * What matters, in order of how badly it would hurt:
 *
 *   1. NOTHING MOVES AND NOTHING IS INVENTED. The ten slots are fixed, no
 *      path is drawn between players, and no position is derived from a guess
 *      about the possession.
 *   2. THE LAYER BOUNDARY. A real coordinate is never tagged schematic, and a
 *      slot is never tagged verified.
 *   3. NO INVENTED FACTS. No assist means no label. No coordinate means no
 *      shot and no arc.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import {
  DISCLOSURE,
  playResult,
  displayedChange,
  displayNames,
  shotWasMade,
  labelSpot,
  LABEL_CLEARANCE,
  LABEL_HALF_LENGTH,
  shouldDrawLink,
  slotObstacles,
  FIXED_LAYERS,
  LAYER,
  SLOTS,
  assignSlots,
  assistFrom,
  findPlayerByLabel,
  hasShotCoordinates,
  lastName,
  pointOnArc,
  probabilityChange,
  reconstruct,
  shotArcPath,
  shotOrigin,
} from "./reconstruct.js";

const PLAYERS = {
  1628369: { person_id: 1628369, name: "Jayson Tatum", coarse_position: "F" },
  1627759: { person_id: 1627759, name: "Jaylen Brown", coarse_position: "G-F" },
  1628401: { person_id: 1628401, name: "Derrick White", coarse_position: "G" },
  201950: { person_id: 201950, name: "Jrue Holiday", coarse_position: "G" },
  204001: { person_id: 204001, name: "Kristaps Porzingis", coarse_position: "C" },
  1629008: { person_id: 1629008, name: "Michael Porter Jr.", coarse_position: "F" },
  203999: { person_id: 203999, name: "Nikola Jokic", coarse_position: "C" },
  1627750: { person_id: 1627750, name: "Jamal Murray", coarse_position: "G" },
  203932: { person_id: 203932, name: "Aaron Gordon", coarse_position: "F" },
  201566: { person_id: 201566, name: "Russell Westbrook", coarse_position: "G" },
};

const BOS = [1628369, 1627759, 1628401, 201950, 204001];
const DEN = [1629008, 203999, 1627750, 203932, 201566];

function shotEvent(overrides = {}) {
  return {
    action_type: "Made Shot",
    shot_result: "Made",
    shot_value: 3,
    person_id: 1628369,
    loc_x: 99,
    loc_y: 220,
    description: "Tatum 25' 3PT Jump Shot (3 PTS) (Brown 1 AST)",
    ...overrides,
  };
}

const build = (event = shotEvent()) =>
  reconstruct({
    event,
    offenseLineup: BOS,
    defenseLineup: DEN,
    players: PLAYERS,
  });

// ---------------------------------------------------------------------------
// 1. Nothing moves, nothing is invented
// ---------------------------------------------------------------------------

test("the ten slots are identical for every event", () => {
  const shot = build();
  const rebound = build(
    shotEvent({ action_type: "Rebound", loc_x: 0, loc_y: 0, description: "White REBOUND" })
  );
  const sub = build(
    shotEvent({ action_type: "Substitution", loc_x: 0, loc_y: 0, description: "SUB" })
  );
  const positions = (r) =>
    [...r.offense, ...r.defense].map((e) => `${e.player.person_id}@${e.x},${e.y}`);

  assert.deepEqual(positions(shot), positions(rebound));
  assert.deepEqual(positions(rebound), positions(sub));
});

test("no randomness anywhere in the module", () => {
  const source = readFileSync(new URL("./reconstruct.js", import.meta.url), "utf8");
  const code = source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/.*$/gm, "");
  assert.ok(!code.includes("Math.random"));
  assert.ok(!code.includes("Date.now"));
});

test("no estimated movement or pass geometry survives", () => {
  const source = readFileSync(new URL("./reconstruct.js", import.meta.url), "utf8");
  // These were real exports in an earlier version. Their absence is the point.
  for (const gone of [
    "movementTrails",
    "interpolatePositions",
    "passPath",
    "defensiveSpot",
    "SPACING",
  ]) {
    assert.ok(
      !source.includes(`export function ${gone}`) &&
        !source.includes(`export const ${gone}`),
      `${gone} must not come back: it drew things nobody recorded`
    );
  }
  const result = build();
  assert.equal(result.estimated, undefined);
  assert.equal(result.involved.passPath, undefined);
});

test("slots are far enough apart that nothing overlaps", () => {
  const all = [...SLOTS.offense, ...SLOTS.defense];
  const MARKER_DIAMETER = 44;
  for (let a = 0; a < all.length; a += 1) {
    for (let b = a + 1; b < all.length; b += 1) {
      const d = Math.hypot(all[a].x - all[b].x, all[a].y - all[b].y);
      assert.ok(d >= MARKER_DIAMETER, `${all[a].key}/${all[b].key} only ${d.toFixed(0)} apart`);
    }
  }
});

test("all ten players are placed, five a side", () => {
  const r = build();
  assert.equal(r.offense.length, 5);
  assert.equal(r.defense.length, 5);
});

test("a short lineup places fewer players rather than inventing any", () => {
  const r = reconstruct({
    event: shotEvent(),
    offenseLineup: [1628369, 1627759],
    defenseLineup: [203999],
    players: PLAYERS,
  });
  assert.equal(r.offense.length, 2);
  assert.equal(r.defense.length, 1);
});

test("slot assignment does not depend on the order ids arrive in", () => {
  const a = assignSlots(BOS, PLAYERS, SLOTS.offense);
  const b = assignSlots([...BOS].reverse(), PLAYERS, SLOTS.offense);
  assert.deepEqual(
    a.map((e) => e.player.person_id),
    b.map((e) => e.player.person_id)
  );
});

test("bigs sit inside and guards outside", () => {
  const placed = assignSlots(BOS, PLAYERS, SLOTS.offense);
  assert.equal(placed[placed.length - 1].player.name, "Kristaps Porzingis");
  assert.equal(placed[0].slot, "PG");
});

// ---------------------------------------------------------------------------
// 2. The layer boundary
// ---------------------------------------------------------------------------

test("every lineup slot is tagged schematic", () => {
  const r = build();
  [...r.offense, ...r.defense].forEach((e) => {
    assert.equal(e.layer, LAYER.SCHEMATIC, `${e.player.name} was ${e.layer}`);
    assert.match(e.note, /not a court position/);
  });
});

test("the shot is the only verified position on the floor", () => {
  const r = build();
  assert.equal(r.shot.origin.x, 99);
  assert.equal(r.shot.origin.y, 220);
  assert.equal(r.shot.layer, LAYER.VERIFIED);
  const onFloor = [...r.offense, ...r.defense];
  assert.equal(onFloor.filter((e) => FIXED_LAYERS.has(e.layer)).length, 0);
});

test("a free throw is rule-defined, never verified", () => {
  const r = build(
    shotEvent({
      action_type: "Free Throw",
      loc_x: 0,
      loc_y: 0,
      description: "Tatum Free Throw 1 of 2 (5 PTS)",
    })
  );
  assert.ok(r.isFreeThrow);
  assert.equal(r.shot.layer, LAYER.RULE);
  assert.notEqual(r.shot.layer, LAYER.VERIFIED);
  assert.match(r.shot.note, /rule-defined location/);
  assert.match(r.shot.note, /Not tracked, not recorded by the feed/);
});

test("the disclosure says what it should", () => {
  assert.equal(build().disclosure, DISCLOSURE);
  assert.match(DISCLOSURE, /Player locations are schematic/);
  assert.match(DISCLOSURE, /Shot location, lineup, clock, score and event are verified/);
});

// ---------------------------------------------------------------------------
// 3. Nothing is invented
// ---------------------------------------------------------------------------

test("an event with no coordinate produces no shot and no arc", () => {
  const rebound = shotEvent({
    action_type: "Rebound",
    loc_x: 0,
    loc_y: 0,
    description: "Holiday REBOUND (Off:1 Def:0)",
  });
  assert.equal(hasShotCoordinates(rebound), false);
  assert.equal(shotOrigin(rebound), null);
  assert.equal(shotArcPath(null), null);
  assert.equal(build(rebound).shot, null);
});

test("a shot recorded at exactly 0,0 is missing, not a shot from the ring", () => {
  assert.equal(hasShotCoordinates(shotEvent({ loc_x: 0, loc_y: 0 })), false);
});

test("the assist label names both players, and only when both are on the floor", () => {
  assert.equal(build().involved.label, "Brown → Tatum");

  const unassisted = build(shotEvent({ description: "Tatum 25' 3PT Jump Shot (3 PTS)" }));
  assert.equal(unassisted.involved.label, null);
  assert.equal(unassisted.involved.assister, null);

  const offFloor = build(
    shotEvent({ description: "Tatum 25' 3PT Jump Shot (3 PTS) (Jokic 1 AST)" })
  );
  assert.equal(offFloor.involved.label, null);
});

test("the assist is read from the description, not guessed", () => {
  assert.equal(
    assistFrom("Porter Jr. 1' Cutting Layup Shot (2 PTS) (Gordon 1 AST)"),
    "Gordon"
  );
  assert.equal(assistFrom("Tatum 25' 3PT Jump Shot (3 PTS)"), null);
  assert.equal(assistFrom(null), null);
});

test("a suffixed surname resolves to the right player and displays correctly", () => {
  assert.equal(findPlayerByLabel("Porter Jr.", DEN, PLAYERS).name, "Michael Porter Jr.");
  assert.equal(lastName("Michael Porter Jr."), "Porter Jr.");
  assert.equal(lastName("Jayson Tatum"), "Tatum");
  assert.equal(lastName(""), "");
});

test("an ambiguous or absent label resolves to nobody", () => {
  assert.equal(findPlayerByLabel("Nobody", BOS, PLAYERS), null);
  assert.equal(findPlayerByLabel(null, BOS, PLAYERS), null);
  assert.equal(findPlayerByLabel("Brown", DEN, PLAYERS), null);
});

test("a dead ball highlights nobody and labels nothing", () => {
  for (const type of ["Substitution", "Timeout", "period", "Instant Replay"]) {
    const r = build(shotEvent({ action_type: type, loc_x: 0, loc_y: 0, description: type }));
    assert.ok(r.deadBall, `${type} should be a dead ball`);
    assert.equal(r.shot, null);
    assert.equal(r.involved.shooter, null);
    assert.equal(r.involved.label, null);
    assert.equal(r.offense.length, 5, "the five are still on the floor");
  }
});

test("a jump shot is not a dead ball", () => {
  assert.equal(build().deadBall, false);
});

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

test("the arc starts at the shot and ends at the ring", () => {
  const path = shotArcPath({ x: 99, y: 220 });
  assert.match(path, /^M 99 220 Q /);
  assert.match(path, / 0 0$/);
});

test("the ball travels from the shot to the ring and nowhere else", () => {
  const origin = { x: 99, y: 220 };
  const start = pointOnArc(origin, 0);
  const end = pointOnArc(origin, 1);
  assert.ok(Math.abs(start.x - 99) < 1e-9 && Math.abs(start.y - 220) < 1e-9);
  assert.ok(Math.abs(end.x) < 1e-9 && Math.abs(end.y) < 1e-9);
});

test("the ball position is clamped and deterministic", () => {
  const origin = { x: 99, y: 220 };
  assert.deepEqual(pointOnArc(origin, -3), pointOnArc(origin, 0));
  assert.deepEqual(pointOnArc(origin, 7), pointOnArc(origin, 1));
  assert.deepEqual(pointOnArc(origin, 0.37), pointOnArc(origin, 0.37));
  assert.equal(pointOnArc(null, 0.5), null);
});

test("the ball stays on the court while travelling", () => {
  for (const origin of [
    { x: 99, y: 220 },
    { x: -230, y: 20 },
    { x: 0, y: 400 },
    { x: 5, y: 10 },
  ]) {
    for (let i = 0; i <= 20; i += 1) {
      const p = pointOnArc(origin, i / 20);
      assert.ok(p.x >= -260 && p.x <= 260, `x ${p.x} off court`);
      assert.ok(p.y >= -140 && p.y <= 430, `y ${p.y} off court`);
    }
  }
});

// ---------------------------------------------------------------------------
// Win probability, which is entirely real
// ---------------------------------------------------------------------------

test("probability change is read from the recorded series", () => {
  const change = probabilityChange({ wp: [0.5, 0.412, 0.586] }, 2);
  assert.equal(change.before, 0.412);
  assert.equal(change.after, 0.586);
  assert.ok(Math.abs(change.delta - 0.174) < 1e-9);
  assert.equal(change.layer, LAYER.VERIFIED);
});

test("the first event reports no change rather than inventing one", () => {
  const change = probabilityChange({ wp: [0.566, 0.6] }, 0);
  assert.equal(change.before, 0.566);
  assert.equal(change.delta, 0);
});

test("an out of range index yields nothing rather than a fabricated number", () => {
  assert.equal(probabilityChange({ wp: [0.5] }, 5), null);
  assert.equal(probabilityChange({ wp: [0.5] }, -1), null);
  assert.equal(probabilityChange({}, 0), null);
});

// ---------------------------------------------------------------------------
// Label placement, settled once because the slots never move
// ---------------------------------------------------------------------------

test("every slot carries a fixed label position", () => {
  [...SLOTS.offense, ...SLOTS.defense].forEach((s) => {
    assert.ok(Number.isFinite(s.lx) && Number.isFinite(s.ly), `${s.key} has no label spot`);
    assert.ok(["start", "middle", "end"].includes(s.anchor));
  });
  const r = build();
  [...r.offense, ...r.defense].forEach((e) => {
    assert.ok(Number.isFinite(e.labelX) && Number.isFinite(e.labelY));
  });
});

test("no name label overlaps another marker", () => {
  // Names are far wider than the 23-unit markers, and the first version hid
  // "Gordon", "Jokic", "Caldwell-Pope" and "Porter Jr." behind adjacent
  // circles. Worst case is a long surname at roughly 7.6 units per character.
  const CHAR = 7.6;
  const HEIGHT = 15;
  const RADIUS = 23;
  const longest = "Caldwell-Pope".length * CHAR;
  const all = [...SLOTS.offense, ...SLOTS.defense];

  all.forEach((label) => {
    const half = longest / 2;
    const cx =
      label.anchor === "start"
        ? label.lx + half
        : label.anchor === "end"
        ? label.lx - half
        : label.lx;
    all.forEach((marker) => {
      const dx = Math.max(0, Math.abs(cx - marker.x) - (half + RADIUS));
      const dy = Math.max(0, Math.abs(label.ly - marker.y) - (HEIGHT / 2 + RADIUS));
      assert.ok(
        dx > 0 || dy > 0,
        `${label.key} label lands on the ${marker.key} marker`
      );
    });
  });
});

test("no two name labels overlap each other", () => {
  const CHAR = 7.6;
  const HEIGHT = 15;
  const longest = "Caldwell-Pope".length * CHAR;
  const boxes = [...SLOTS.offense, ...SLOTS.defense].map((s) => {
    const half = longest / 2;
    const cx = s.anchor === "start" ? s.lx + half : s.anchor === "end" ? s.lx - half : s.lx;
    return { key: s.key, x0: cx - half, x1: cx + half, y0: s.ly - HEIGHT / 2, y1: s.ly + HEIGHT / 2 };
  });
  for (let a = 0; a < boxes.length; a += 1) {
    for (let b = a + 1; b < boxes.length; b += 1) {
      const A = boxes[a];
      const B = boxes[b];
      const clash = A.x0 < B.x1 && A.x1 > B.x0 && A.y0 < B.y1 && A.y1 > B.y0;
      assert.ok(!clash, `${A.key} and ${B.key} labels overlap`);
    }
  }
});

// ---------------------------------------------------------------------------
// playResult: the short label the ribbon and the card both show
// ---------------------------------------------------------------------------

test("a made and a missed field goal carry their point value", () => {
  assert.equal(
    playResult({ action_type: "Made Shot", shot_value: 3, description: "x" }),
    "Made 3PT"
  );
  assert.equal(
    playResult({ action_type: "Missed Shot", shot_value: 2, description: "x" }),
    "Missed 2PT"
  );
});

test("a free throw is read from the description, because the feed leaves shot_result empty", () => {
  // Both of these are real rows from game 0022300906. shot_result is "" and
  // shot_value is 0 on every free throw, so the leading MISS is the only
  // record of the outcome.
  assert.equal(
    playResult({
      action_type: "Free Throw",
      shot_result: "",
      shot_value: 0,
      description: "MISS Brown Free Throw 1 of 2",
    }),
    "Missed free throw"
  );
  assert.equal(
    playResult({
      action_type: "Free Throw",
      shot_result: "",
      shot_value: 0,
      description: "Brown Free Throw 2 of 2 (7 PTS)",
    }),
    "Made free throw"
  );
});

test("steals and blocks arrive with a blank action type", () => {
  assert.equal(
    playResult({ action_type: "", description: "Porzingis STEAL (1 STL)" }),
    "Steal"
  );
  assert.equal(
    playResult({ action_type: "", description: "White BLOCK (1 BLK)" }),
    "Block"
  );
});

test("anything else is the action type as written, and an empty event is not invented", () => {
  assert.equal(playResult({ action_type: "Rebound", description: "" }), "Rebound");
  assert.equal(playResult({ action_type: "Turnover", description: "" }), "Turnover");
  assert.equal(playResult({ action_type: "period", description: "" }), "Period break");
  assert.equal(playResult({}), "Play");
  assert.equal(playResult(null), "Play");
});

// ---------------------------------------------------------------------------
// shotWasMade: the free throw outcome the feed does not put in shot_result
// ---------------------------------------------------------------------------

test("a field goal uses shot_result", () => {
  assert.equal(shotWasMade({ action_type: "Made Shot", shot_result: "Made" }), true);
  assert.equal(shotWasMade({ action_type: "Missed Shot", shot_result: "Missed" }), false);
});

test("a free throw uses the description, because shot_result is empty on every one", () => {
  const made = {
    action_type: "Free Throw",
    shot_result: "",
    description: "Murray Free Throw Technical (1 PTS)",
  };
  const missed = {
    action_type: "Free Throw",
    shot_result: "",
    description: "MISS Brown Free Throw 1 of 2",
  };
  assert.equal(shotWasMade(made), true);
  assert.equal(shotWasMade(missed), false);
  // The regression this guards: reading shot_result alone made every free
  // throw a miss on the court while the card beside it said "Made free throw".
  assert.notEqual(shotWasMade(made), made.shot_result === "Made");
});

test("reconstruct carries the free throw outcome through", () => {
  const players = { 1: { person_id: 1, name: "Jamal Murray", position: "G" } };
  const built = (description) =>
    reconstruct({
      event: { action_type: "Free Throw", shot_result: "", person_id: 1, description },
      offenseLineup: [1],
      defenseLineup: [],
      players,
    });
  assert.equal(built("Murray Free Throw Technical (1 PTS)").shot.made, true);
  assert.equal(built("MISS Murray Free Throw 1 of 2").shot.made, false);
});

// ---------------------------------------------------------------------------
// labelSpot: the "schematic link" words must not land on a player
// ---------------------------------------------------------------------------

test("the label clears every marker on a straight free throw line", () => {
  // The exact case that broke: shooter in the offensive point guard slot, shot
  // at the free throw line, and the DEFENDING point guard slot sitting on the
  // straight line between them at (0, 250).
  const markers = [...SLOTS.offense, ...SLOTS.defense];
  const spot = labelSpot(0, 297, 0, 152.5, markers);
  for (const m of markers) {
    assert.ok(
      Math.hypot(m.x - spot.x, m.y - spot.y) >= LABEL_CLEARANCE,
      `label at ${spot.x},${spot.y} is too close to slot ${m.key}`
    );
  }
});

test("the label clears every marker wherever the shot was taken", () => {
  // Sweeps the whole half court rather than a handful of hand-picked spots.
  // Two earlier collisions in this project were found by a viewer, not by a
  // test, because the test only tried the cases I had already thought of.
  // Real obstacles: ten circles AND ten names, with the longest surnames in
  // the dataset so the widest case is the one under test.
  const LONG = ["Caldwell-Pope", "Porzingis", "Antetokounmpo", "Porter Jr.",
                "Holiday", "Gordon", "Murray", "Tatum", "Brown", "White"];
  const slots = [...SLOTS.offense, ...SLOTS.defense].map((s, i) => ({
    ...s,
    labelX: s.lx,
    labelY: s.ly,
    player: { name: `First ${LONG[i]}` },
  }));
  // The same rule labelSpot applies, written out here rather than imported, so
  // the test would catch the predicate itself being loosened.
  const outside = (a, p) => {
    const rx = a.rx ?? LABEL_CLEARANCE;
    const ry = a.ry ?? LABEL_CLEARANCE;
    return ((a.x - p.x) / rx) ** 2 + ((a.y - p.y) / ry) ** 2 >= 1;
  };
  const markers = slotObstacles(slots);
  let drawn = 0;
  let skipped = 0;
  for (const slot of slots) {
    for (let x = -240; x <= 240; x += 20) {
      for (let y = -40; y <= 400; y += 20) {
        const to = { x, y };
        if (!shouldDrawLink(slot, to)) {
          skipped += 1;
          continue;
        }
        drawn += 1;
        const spot = labelSpot(slot.x, slot.y, to.x, to.y, markers);
        // Check the whole bar the words occupy, not just its centre.
        const len = Math.hypot(to.x - slot.x, to.y - slot.y) || 1;
        const ux = (to.x - slot.x) / len;
        const uy = (to.y - slot.y) / len;
        for (const k of [-1, -0.5, 0, 0.5, 1]) {
          const pt = {
            x: spot.x + ux * LABEL_HALF_LENGTH * k,
            y: spot.y + uy * LABEL_HALF_LENGTH * k,
          };
          const hit = markers.find((m) => !outside(m, pt));
          assert.ok(
            !hit,
            `slot ${slot.key} -> ${x},${y} at k=${k}: label lands on ${JSON.stringify(hit)}`
          );
        }
      }
    }
  }
  assert.ok(drawn > 4000, `expected a wide sweep, drew ${drawn}`);
  assert.ok(skipped > 0, "expected some shots too close to their slot to link");
});

test("no link is drawn when the shot is already beside the shooter's slot", () => {
  const slot = SLOTS.offense[2];
  assert.equal(shouldDrawLink(slot, { x: slot.x - 20, y: slot.y + 10 }), false);
  assert.equal(shouldDrawLink(slot, { x: slot.x - 45, y: slot.y - 55 }), false);
  assert.equal(shouldDrawLink(slot, { x: 0, y: 0 }), true);
  assert.equal(shouldDrawLink(null, { x: 0, y: 0 }), false);
});

// ---------------------------------------------------------------------------
// displayedChange: Before, After and Change must add up on screen
// ---------------------------------------------------------------------------

test("the change is computed from the rounded endpoints, not rounded separately", () => {
  // The exact case a browser check caught. 0.5885 * 100 is 58.850000000000001
  // in binary floating point, so it rounds UP to 58.9 rather than to the 58.8
  // a decimal reading would predict. That is precisely why the change has to
  // be derived from the rounded endpoints: whatever the endpoints print, the
  // difference between them is what the reader subtracts.
  const events = { wp: [0.5885, 0.5800] };
  const shown = displayedChange(events, 1);
  assert.equal(shown.before, 58.9);
  assert.equal(shown.after, 58.0);
  assert.equal(shown.points, -0.9);
  // Compared through toFixed because 58 - 58.9 is -0.9000000000000057 in
  // binary floating point. The property test below is the real guarantee.
  assert.equal(Number((shown.after - shown.before).toFixed(1)), shown.points);
  assert.equal(shown.direction, "down");
});

test("the three displayed numbers always add up, across the range", () => {
  for (let i = 0; i < 400; i += 1) {
    const before = i / 400;
    const after = ((i * 7) % 401) / 400;
    const shown = displayedChange({ wp: [before, after] }, 1);
    assert.equal(
      Number((shown.after - shown.before).toFixed(1)),
      shown.points,
      `${before} -> ${after}`
    );
  }
});

test("the first event has no prior probability, so the change is zero", () => {
  const shown = displayedChange({ wp: [0.61, 0.55] }, 0);
  assert.equal(shown.points, 0);
  assert.equal(shown.direction, "flat");
  assert.equal(shown.before, shown.after);
});

test("the raw delta is kept for anything that needs full precision", () => {
  const shown = displayedChange({ wp: [0.5885, 0.58] }, 1);
  assert.ok(Math.abs(shown.raw - -0.85) < 1e-9, `raw was ${shown.raw}`);
});

// ---------------------------------------------------------------------------
// displayNames: two players on the floor can share a surname
// ---------------------------------------------------------------------------

const entry = (id, name) => ({ player: { person_id: id, name } });

test("a surname shared by two players on the floor falls back to full names", () => {
  // The real case, 2024-03-07: Boston's Jrue Holiday and Denver's Justin
  // Holiday were on court together and the diagram labelled both "Holiday".
  const labels = displayNames([
    entry(1, "Jrue Holiday"),
    entry(2, "Justin Holiday"),
    entry(3, "Nikola Jokic"),
  ]);
  assert.equal(labels.get(1), "Jrue Holiday");
  assert.equal(labels.get(2), "Justin Holiday");
  assert.equal(labels.get(3), "Jokic");
});

test("a first initial would not have been enough", () => {
  // Both are J, which is why the fallback is the whole name rather than an
  // initial. This test exists so nobody "simplifies" it back later.
  const names = ["Jrue Holiday", "Justin Holiday"];
  const initials = names.map((n) => `${n[0]}. ${lastName(n)}`);
  assert.equal(initials[0], initials[1]);
});

test("suffixes still resolve, and are not treated as a collision", () => {
  const labels = displayNames([
    entry(1, "Michael Porter Jr."),
    entry(2, "Larry Nance Jr."),
  ]);
  assert.equal(labels.get(1), "Porter Jr.");
  assert.equal(labels.get(2), "Nance Jr.");
});

test("three players sharing a surname all get full names", () => {
  const labels = displayNames([
    entry(1, "Marcus Morris"),
    entry(2, "Markieff Morris"),
    entry(3, "Monte Morris"),
  ]);
  assert.equal(labels.get(1), "Marcus Morris");
  assert.equal(labels.get(2), "Markieff Morris");
  assert.equal(labels.get(3), "Monte Morris");
});

test("the link label sizes its obstacle from the name actually drawn", () => {
  const slots = [
    { ...SLOTS.offense[0], labelX: SLOTS.offense[0].lx,
      labelY: SLOTS.offense[0].ly, ...entry(1, "Jrue Holiday") },
    { ...SLOTS.defense[0], labelX: SLOTS.defense[0].lx,
      labelY: SLOTS.defense[0].ly, ...entry(2, "Justin Holiday") },
  ];
  const labels = displayNames(slots);
  const wide = slotObstacles(slots, labels);
  const narrow = slotObstacles(slots);
  const widest = (list) => Math.max(...list.filter((o) => o.rx).map((o) => o.rx));
  assert.ok(widest(wide) > widest(narrow),
            "the full name must produce a wider obstacle than the surname");
});

// ---------------------------------------------------------------------------
// Slot geometry
//
// The slots were compressed vertically so the court fits a laptop viewport.
// These pin the properties that compression could quietly break.
// ---------------------------------------------------------------------------

test("no two slots are close enough for their markers to touch", () => {
  const MARKER = 23;
  const all = [...SLOTS.offense, ...SLOTS.defense];
  for (let i = 0; i < all.length; i += 1) {
    for (let j = i + 1; j < all.length; j += 1) {
      const gap = Math.hypot(all[i].x - all[j].x, all[i].y - all[j].y);
      assert.ok(
        gap >= MARKER * 2 + 12,
        `${all[i].key} and ${all[j].key} are ${gap.toFixed(1)} apart`
      );
    }
  }
});

test("every slot and every label fits inside the drawn court", () => {
  // The court crops at 312 and spans -250 to 250. A slot or a name outside
  // that is simply invisible, which is a silent failure.
  const CROP = 312;
  for (const slot of [...SLOTS.offense, ...SLOTS.defense]) {
    assert.ok(Math.abs(slot.x) + 23 <= 266, `${slot.key} marker off the side`);
    assert.ok(slot.y + 23 <= CROP, `${slot.key} marker below the crop`);
    assert.ok(slot.ly <= CROP - 6, `${slot.key} label below the crop`);
    assert.ok(slot.y - 23 >= -60, `${slot.key} marker above the baseline`);
  }
});

test("the attacking five sit further out than the defending five", () => {
  // Not decoration: it is how a viewer tells the two apart before reading a
  // single name.
  const meanY = (slots) => slots.reduce((s, x) => s + x.y, 0) / slots.length;
  assert.ok(meanY(SLOTS.offense) > meanY(SLOTS.defense));
});
