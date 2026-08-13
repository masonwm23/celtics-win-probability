"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { COURT, shotsUpTo, threePointPath } from "@/lib/court";
import {
  LAYER,
  displayNames,
  labelSpot,
  pointOnArc,
  reconstruct,
  shouldDrawLink,
  slotObstacles,
} from "@/lib/reconstruct";
import { courtLabel } from "@/lib/format";
import {
  MADE_CLASS,
  MADE_RADIUS,
  MISS_CLASS,
  missMarkLines,
} from "@/lib/marks";

/**
 * The half court.
 *
 * Ten players in FIXED SCHEMATIC SLOTS. They do not move between events, no
 * path is drawn between them, and their positions say nothing about where
 * anybody stood. Public play-by-play records one coordinate per shot attempt
 * and nothing else, so a diagram is the honest shape for the rest.
 *
 * What IS real on this drawing:
 *   - the shot marker, at the feed's own coordinate;
 *   - who was on the floor;
 *   - who the event was about, shown by a ring rather than by moving them.
 *
 * The dotted line from a shooter's slot to the shot marker is labelled
 * "schematic link" on the drawing itself, because it joins a diagram position
 * to a recorded one and is not a route anybody ran.
 *
 * Coordinates are the NBA's own: tenths of a foot, hoop at the origin. No
 * scaling fudge anywhere, which is why the court keeps its true proportions
 * rather than being widened to fill the column.
 */

/**
 * The drawing area.
 *
 * Cropped at 312 rather than running to the division line at 417.5. The
 * deepest slot sits at 258 and its name at 290, so nothing is lost, and a half
 * court drawn in full is very nearly square: at 65% of a laptop's width it was
 * taller than the viewport and pushed the playback controls off screen.
 *
 * A crop is not a distortion. Every coordinate inside it is still exactly
 * where the feed put it, and a shot beyond the crop cannot occur because the
 * far half belongs to the other basket. Scaling the court to a wider shape
 * WOULD be a distortion, so that is not done.
 */
const CROP_Y = 312;

const VIEW = {
  x: COURT.minX - 16,
  y: COURT.baselineY - 14,
  w: COURT.maxX - COURT.minX + 32,
  h: CROP_Y - COURT.baselineY + 14,
};

/** How many faded earlier markers "Recent shots" is allowed to draw. */
export const RECENT_SHOTS = 8;

/**
 * Slot marker radius. The fixed slots in lib/reconstruct.js are at least 58
 * units apart, so two 23-unit markers (46 across) never touch.
 */
export const MARKER_R = 23;

export default function PlayCourt({
  events,
  cursor,
  players,
  celticsLineup,
  opponentLineup,
  opponentAbbrev,
  mode = "current",
  shotFilter = {},
  fullHistory = false,
  ballMs = 0,
  photos = true,
  homeLogo = null,
}) {
  const event = useMemo(
    () => ({
      action_type: events.action_type[cursor],
      shot_result: events.shot_result[cursor],
      shot_value: events.shot_value[cursor],
      person_id: events.person_id[cursor],
      loc_x: events.loc_x[cursor],
      loc_y: events.loc_y[cursor],
      description: events.description[cursor],
      team: events.team[cursor],
    }),
    [events, cursor]
  );

  const celticsHaveBall = event.team !== opponentAbbrev;

  const play = useMemo(
    () =>
      reconstruct({
        event,
        offenseLineup: celticsHaveBall ? celticsLineup : opponentLineup,
        defenseLineup: celticsHaveBall ? opponentLineup : celticsLineup,
        players,
      }),
    [event, celticsHaveBall, celticsLineup, opponentLineup, players]
  );

  // The ball travels to the ring once, on the event it belongs to, and ONLY on
  // a make. Animating a miss into the ring would say the shot went in, which
  // is the opposite of what the feed recorded. Nothing else on this drawing
  // moves, and the arc line itself is never drawn: only the marker stays
  // behind, which is the one thing with a recorded location.
  const madeShot = Boolean(play.shot?.made);
  const flight = useBallFlight(cursor, madeShot ? ballMs : 0);
  const ball = madeShot && flight < 1 ? pointOnArc(play.shot.origin, flight) : null;

  // "Current play" is the default and draws nothing but this event's shot.
  // "Recent shots" adds a short, faded tail. The whole game is available only
  // behind the collapsed row, never by default.
  const history = useMemo(() => {
    if (mode !== "recent") return [];
    const all = shotsUpTo(events, cursor, { limit: 400 }).filter((shot) => {
      const boston = shot.team !== opponentAbbrev;
      if (shotFilter.team === "bos" && !boston) return false;
      if (shotFilter.team === "opp" && boston) return false;
      if (shotFilter.result === "made" && !shot.made) return false;
      if (shotFilter.result === "missed" && shot.made) return false;
      if (shotFilter.value === "3" && shot.value !== 3) return false;
      if (shotFilter.value === "2" && shot.value !== 2) return false;
      if (
        shotFilter.period === "current" &&
        events.period[shot.i] !== events.period[cursor]
      )
        return false;
      return true;
    });
    return fullHistory ? all : all.slice(-RECENT_SHOTS);
  }, [events, cursor, mode, fullHistory, shotFilter, opponentAbbrev]);

  // Two photos per player, in order: the SEASON one, showing the jersey he
  // actually wore that year, and the CURRENT one as a fallback. 3,680 of the
  // 4,009 player-seasons in the dataset have a confirmed season photo; the
  // rest returned 403 and get the current picture instead.
  //
  // Failures are tracked by URL, not by player, so a season image that 404s
  // falls through to the current image rather than skipping straight to the
  // number. The last resort is the jersey number or initials, never a generic
  // silhouette: a silhouette reads as "this player" and a number reads as
  // "no photo", and only the second is true.
  const [brokenPhotos, setBrokenPhotos] = useState(() => new Set());
  const markPhotoBroken = (url) =>
    setBrokenPhotos((prev) => (prev.has(url) ? prev : new Set(prev).add(url)));

  const photoFor = (player) => {
    if (!photos) return null;
    for (const url of [player.headshot, player.headshot_current]) {
      if (url && !brokenPhotos.has(url)) return url;
    }
    return null;
  };

  // Two players on the floor can share a surname. When they do, both get their
  // full name rather than two circles labelled the same thing.
  const labels = displayNames([...play.offense, ...play.defense]);

  const highlighted = new Set(
    [play.involved.shooterSlot, play.involved.assisterSlot]
      .filter(Boolean)
      .map((e) => e.player.person_id)
  );

  const marker = (entry, side) => {
    const id = entry.player.person_id;
    // TWO independent things, kept independent.
    //
    //   team  who they play for. Boston green, opponent slate. Identity, and
    //         it never changes within a game.
    //   side  who has the ball right now. The attacking five are drawn
    //         brightly and the defending five are muted.
    //
    // Neither says anything about where a player stood. Emphasis is a fact
    // about possession, which the feed records; position is not.
    const team = entry.player.is_celtics ? "bos" : "opp";
    const lit = highlighted.has(id);
    const photo = photoFor(entry.player);
    const clipId = `hs-${team}-${id}`;
    return (
      <g
        key={`${team}-${id}`}
        className={`slot slot--${team} slot--${side} ${lit ? "slot--lit" : ""}`}
      >
        <circle cx={entry.x} cy={entry.y} r={MARKER_R} />
        {photo ? (
          <>
            <clipPath id={clipId}>
              <circle cx={entry.x} cy={entry.y} r={MARKER_R - 1.4} />
            </clipPath>
            <image
              className="slot__photo"
              href={photo}
              x={entry.x - MARKER_R}
              y={entry.y - MARKER_R}
              width={MARKER_R * 2}
              height={MARKER_R * 2}
              preserveAspectRatio="xMidYMid slice"
              clipPath={`url(#${clipId})`}
              key={photo}
              onError={() => markPhotoBroken(photo)}
            />
          </>
        ) : (
          <text x={entry.x} y={entry.y} textAnchor="middle" dominantBaseline="central">
            {courtLabel(entry.player)}
          </text>
        )}
        <text
          x={entry.labelX}
          y={entry.labelY}
          textAnchor={entry.anchor}
          className="slot__name"
        >
          {labels.get(id) || ""}
        </text>
      </g>
    );
  };

  return (
    <div className="courtwrap">
      <svg
        className="court"
        viewBox={`${VIEW.x} ${VIEW.y} ${VIEW.w} ${VIEW.h}`}
        role="img"
        aria-label="Schematic lineup and the recorded shot for the current play"
      >
        <rect
          x={COURT.minX}
          y={COURT.baselineY}
          width={COURT.maxX - COURT.minX}
          height={COURT.halfCourtY - COURT.baselineY}
          className="court__floor"
        />
        <rect
          x={-COURT.paintHalfWidth}
          y={COURT.baselineY}
          width={COURT.paintHalfWidth * 2}
          height={COURT.freeThrowLineY - COURT.baselineY}
          className="court__paint"
        />
        <circle
          cx="0"
          cy={COURT.freeThrowLineY}
          r={COURT.freeThrowCircleRadius}
          className="court__line"
        />
        <path
          d={`M ${-COURT.restrictedRadius} ${COURT.hoopY} A ${COURT.restrictedRadius} ${COURT.restrictedRadius} 0 0 0 ${COURT.restrictedRadius} ${COURT.hoopY}`}
          className="court__line"
        />
        <path d={threePointPath()} className="court__line" />
        <line
          x1={-COURT.backboardHalfWidth}
          x2={COURT.backboardHalfWidth}
          y1={COURT.backboardY}
          y2={COURT.backboardY}
          className="court__line"
          strokeWidth="3.4"
        />
        <circle cx="0" cy={COURT.hoopY} r={COURT.hoopRadius} className="court__line" />
        <rect
          x={COURT.minX}
          y={COURT.baselineY}
          width={COURT.maxX - COURT.minX}
          height={COURT.halfCourtY - COURT.baselineY}
          className="court__line"
        />
        <circle cx="0" cy={COURT.halfCourtY} r="60" className="court__line" />

        {/* The home team's logo at centre, the way a real floor carries it.
            Which side is home is the feed's own flag (celtics_is_home): Boston's
            crest when they host, the opponent's when they do. Placed in the
            free-throw circle because centre court proper sits past the crop.
            Faint and non-interactive, so it reads as paint under the players and
            the shot marker rather than a control. */}
        {homeLogo && (
          <image
            href={homeLogo}
            x={-46}
            y={COURT.freeThrowLineY - 46}
            width={92}
            height={92}
            preserveAspectRatio="xMidYMid meet"
            opacity="0.2"
            style={{ pointerEvents: "none" }}
            aria-hidden="true"
          />
        )}

        {/* Recent shots only, and only when asked for. Faded, behind everything. */}
        {history.map((shot) => {
          if (shot.i === cursor) return null;
          if (shot.made) {
            return (
              <circle key={shot.i} cx={shot.x} cy={shot.y} r="7" className="past past--made" />
            );
          }
          return (
            <g key={shot.i} className="past past--missed">
              <line x1={shot.x - 6} y1={shot.y - 6} x2={shot.x + 6} y2={shot.y + 6} />
              <line x1={shot.x - 6} y1={shot.y + 6} x2={shot.x + 6} y2={shot.y - 6} />
            </g>
          );
        })}

        {/* The ten schematic slots. Defending five first, so the attacking
            five sit over them if a label ever runs close. */}
        {play.defense.map((entry) => marker(entry, "guarding"))}
        {play.offense.map((entry) => marker(entry, "attacking"))}

        {/* A diagram position joined to a recorded one, and said so on the
            drawing. Drawn LAST so its label is never buried under a marker: an
            earlier version put it first and the words came out as "sc...link"
            with a player's circle across the middle of them. */}
        {play.shot &&
          play.involved.shooterSlot &&
          shouldDrawLink(play.involved.shooterSlot, play.shot.origin) && (
          <SchematicLink
            from={play.involved.shooterSlot}
            to={play.shot.origin}
            avoid={slotObstacles([...play.offense, ...play.defense], labels)}
          />
        )}

        {/* The shot, at the coordinate the feed recorded. */}
        {play.shot && (
          <g className={play.shot.layer === LAYER.RULE ? "shot shot--rule" : "shot"}>
            {play.shot.made ? (
              <circle
                cx={play.shot.origin.x}
                cy={play.shot.origin.y}
                r={MADE_RADIUS}
                className={MADE_CLASS}
              />
            ) : (
              // Same geometry the legend draws, from lib/marks.js, so the key
              // and the court can never disagree about what a miss looks like.
              <g className={MISS_CLASS}>
                {missMarkLines(play.shot.origin.x, play.shot.origin.y).map(
                  (line, i) => (
                    <line key={i} {...line} />
                  )
                )}
              </g>
            )}
            {ball && <circle cx={ball.x} cy={ball.y} r="6" className="ball" />}
          </g>
        )}
      </svg>
    </div>
  );
}

/**
 * Shooter's slot to the recorded shot location.
 *
 * Dotted, faint, and carrying the words "schematic link" on the drawing itself.
 * One end is a diagram and the other is a measurement, so the line exists to
 * say which player the marker belongs to and nothing more. The label is rotated
 * to the line's own angle, flipped when that would leave it upside down.
 */

export function SchematicLink({ from, to, avoid = [] }) {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const length = Math.hypot(dx, dy) || 1;
  // Start clear of the marker and stop clear of the shot symbol, so neither is
  // touched by the line.
  const ux = dx / length;
  const uy = dy / length;
  const x1 = from.x + ux * (MARKER_R + 4);
  const y1 = from.y + uy * (MARKER_R + 4);
  const x2 = to.x - ux * 15;
  const y2 = to.y - uy * 15;

  let angle = (Math.atan2(y2 - y1, x2 - x1) * 180) / Math.PI;
  if (angle > 90 || angle < -90) angle += 180;

  const spot = labelSpot(x1, y1, x2, y2, avoid);

  return (
    <g className="schematiclink">
      <line x1={x1} y1={y1} x2={x2} y2={y2} />
      <text
        x={spot.x}
        y={spot.y}
        textAnchor="middle"
        dominantBaseline="central"
        transform={`rotate(${angle} ${spot.x} ${spot.y})`}
      >
        schematic link
      </text>
    </g>
  );
}

/**
 * 0 to 1, restarting on each event, for the ball's flight.
 *
 * Returns 1 immediately when the duration is zero or the viewer has asked for
 * reduced motion, so the animation is opt-out rather than unavoidable.
 */
function useBallFlight(key, duration) {
  const [t, setT] = useState(1);
  const raf = useRef(null);

  useEffect(() => {
    const reduced =
      typeof window !== "undefined" &&
      window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    if (!duration || duration <= 0 || reduced) {
      setT(1);
      return undefined;
    }
    let start;
    setT(0);
    const step = (now) => {
      if (start === undefined) start = now;
      const progress = Math.min(1, (now - start) / duration);
      setT(progress);
      if (progress < 1) raf.current = requestAnimationFrame(step);
    };
    raf.current = requestAnimationFrame(step);
    return () => {
      if (raf.current) cancelAnimationFrame(raf.current);
    };
  }, [key, duration]);

  return t;
}
