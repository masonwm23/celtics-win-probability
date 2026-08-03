"use client";

import { useMemo } from "react";

import { formatClock, periodLabel } from "@/lib/format";
import { displayedChange } from "@/lib/reconstruct";

/**
 * The win probability immediately around the current play.
 *
 * Same series as the big chart above, out of fold, just a short window of it.
 * The big chart answers "how did this game go"; this one answers "what did
 * THIS play do", which is the research question in miniature.
 *
 * Two honesty points are built into the drawing rather than left to a caption:
 *
 *   1. The window is a slice of the real series. Nothing is smoothed, resampled
 *      or interpolated. Every vertex is an event that happened.
 *   2. The step from `before` to `after` is drawn as the actual pair of points
 *      either side of the cursor, so the number in the Change card and the jump
 *      on the line are the same measurement, not two calculations that could
 *      drift apart.
 *
 * The window is measured in EVENTS, not seconds. Events are not evenly spaced
 * in game time, and a fixed 90-second window would show 40 events during a
 * scoring run and 3 during a free throw sequence.
 *
 * This chart deliberately does NOT settle its last point to 0 or 100 the way
 * the scoreboard readouts do, and that is not an oversight. The dashboard draws
 * a line between two kinds of number: what is the STATE OF THE GAME, which
 * resolves the moment the clock hits zero, and what the MODEL DID, which is a
 * measurement and must never be edited. This panel is the second kind. Point 2
 * above requires the drawn jump and the Change card to be one measurement, and
 * settling the endpoint here would make them disagree at the final event while
 * inventing a movement the model never made. lib/settled has the full split.
 */

const WIDTH = 420;
const HEIGHT = 186;
const PAD = { top: 14, right: 12, bottom: 26, left: 34 };

/** How many events either side of the cursor the window holds. */
export const WINDOW = 26;

export default function MiniProbability({ events, cursor }) {
  const geometry = useMemo(() => {
    const n = events.wp.length;
    if (!n) return null;

    // Clamped rather than wrapped, so the first and last plays of a game get a
    // full-width window instead of a stub.
    let from = Math.max(0, cursor - WINDOW);
    let to = Math.min(n - 1, cursor + WINDOW);
    if (to - from < Math.min(n - 1, WINDOW)) {
      if (from === 0) to = Math.min(n - 1, WINDOW * 2);
      if (to === n - 1) from = Math.max(0, n - 1 - WINDOW * 2);
    }

    const span = Math.max(1, to - from);
    const plotW = WIDTH - PAD.left - PAD.right;
    const plotH = HEIGHT - PAD.top - PAD.bottom;

    const xOf = (i) => PAD.left + ((i - from) / span) * plotW;
    const yOf = (p) => PAD.top + (1 - p) * plotH;

    let line = "";
    for (let i = from; i <= to; i += 1) {
      line += `${i === from ? "M" : "L"} ${xOf(i).toFixed(2)} ${yOf(events.wp[i]).toFixed(2)} `;
    }

    const midline = yOf(0.5);
    const area =
      `${line.trim()} L ${xOf(to).toFixed(2)} ${midline} ` +
      `L ${xOf(from).toFixed(2)} ${midline} Z`;

    // The two points the Change card is computed from. On the very first event
    // there is no prior probability, so `before` is the same point and the
    // change is correctly zero rather than a jump from an invented starting
    // value.
    const beforeIndex = cursor > 0 ? cursor - 1 : cursor;
    const before = { x: xOf(beforeIndex), y: yOf(events.wp[beforeIndex]) };
    const after = { x: xOf(cursor), y: yOf(events.wp[cursor]) };

    // Ticks read the clock off the events themselves rather than dividing the
    // window evenly, so the labels are times that actually occurred.
    const ticks = [from, Math.round((from + to) / 2), to].map((i) => ({
      i,
      x: xOf(i),
      label: formatClock(events.clock[i]),
    }));

    return { from, to, xOf, yOf, line: line.trim(), area, before, after,
             midline, ticks };
  }, [events, cursor]);

  if (!geometry) return null;

  // Same rounding the card uses, so a step drawn green is never labelled with
  // a negative number underneath it.
  const direction = displayedChange(events, cursor)?.direction || "flat";

  return (
    <div className="mini">
      <div className="mini__head">
        <span className="mini__title">Boston win probability around this play</span>
        <span className="mini__scope">
          {periodLabel(events.period[cursor])} · {geometry.to - geometry.from + 1} plays
        </span>
      </div>

      <svg
        className="mini__svg"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label="Out-of-fold win probability for the plays around this one"
      >
        {[0, 0.5, 1].map((p) => (
          <g key={p}>
            <line
              x1={PAD.left}
              x2={WIDTH - PAD.right}
              y1={geometry.yOf(p)}
              y2={geometry.yOf(p)}
              className={p === 0.5 ? "mini__mid" : "mini__grid"}
            />
            <text x={PAD.left - 7} y={geometry.yOf(p) + 4} className="mini__axis">
              {p * 100}%
            </text>
          </g>
        ))}

        <path d={geometry.area} className="mini__area" />
        <path d={geometry.line} className="mini__line" />

        {/* The cursor, and the two points the change is measured between. */}
        <line
          x1={geometry.after.x}
          x2={geometry.after.x}
          y1={PAD.top}
          y2={HEIGHT - PAD.bottom}
          className="mini__cursor"
        />
        <line
          x1={geometry.before.x}
          y1={geometry.before.y}
          x2={geometry.after.x}
          y2={geometry.after.y}
          className={`mini__step mini__step--${direction}`}
        />
        <circle cx={geometry.before.x} cy={geometry.before.y} r="3.4"
                className="mini__before" />
        <circle cx={geometry.after.x} cy={geometry.after.y} r="5"
                className="mini__after" />

        {geometry.ticks.map((tick) => (
          <text key={tick.i} x={tick.x} y={HEIGHT - 8}
                textAnchor="middle" className="mini__axis">
            {tick.label}
          </text>
        ))}
      </svg>
    </div>
  );
}
