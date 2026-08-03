"use client";

import { useMemo, useRef } from "react";
import { periodLabel, percent } from "@/lib/format";
import { settledSeries, settledWp } from "@/lib/settled";

/**
 * The win probability trace, hand-drawn in SVG.
 *
 * No chart library. The shape needed here is specific enough that a general
 * purpose library would cost more in configuration than the 120 lines below,
 * and this way every pixel is accounted for.
 *
 * Two lines are drawn on purpose. The Celtics-specific model and the generic
 * margin-and-time baseline are statistically indistinguishable on 636 games
 * (bootstrap difference +0.0011, interval spanning zero), and showing them
 * together makes the paper's headline result something you can see rather than
 * something you have to be told.
 *
 * Both lines end on the settled result rather than on the model's residual, so
 * the trace agrees with the headline number above it. That is the final point
 * only, it is display-only, and lib/settled explains what it refuses to do. The
 * rest of the trace is exactly what the models said.
 */

const WIDTH = 1000;

/**
 * Two sizes, one component.
 *
 * The research view keeps the tall chart. The live view needs one that fits
 * beside the court on a laptop without pushing the reconstruction below the
 * fold, which was the whole problem: at 100% zoom the full-height chart alone
 * used most of the viewport.
 *
 * Same geometry, same scales, same click-and-drag seeking. A second chart
 * component would be a second thing to keep in step.
 */
const SIZES = {
  full: { height: 260, pad: { top: 14, right: 14, bottom: 26, left: 40 } },
  compact: { height: 218, pad: { top: 12, right: 10, bottom: 26, left: 32 } },
};

export default function WinProbabilityChart({
  events,
  meta,
  cursor,
  onCursor,
  showBaseline,
  periods,
  variant = "full",
}) {
  const svgRef = useRef(null);
  const { height: HEIGHT, pad: PAD } = SIZES[variant] || SIZES.full;

  // The two drawn series, with their terminal point resolved to the result.
  // settledSeries hands back the ORIGINAL array when nothing is settled, so
  // these stay referentially stable and the memo below is not defeated.
  const drawn = useMemo(
    () => ({
      celtics: settledSeries(events.wp, events, meta),
      generic: settledSeries(events.wp_generic, events, meta),
    }),
    [events, meta],
  );

  const geometry = useMemo(() => {
    const n = events.wp.length;
    const maxElapsed = events.elapsed[n - 1] || 1;
    const plotWidth = WIDTH - PAD.left - PAD.right;
    const plotHeight = HEIGHT - PAD.top - PAD.bottom;

    const xOf = (i) => PAD.left + (events.elapsed[i] / maxElapsed) * plotWidth;
    const yOf = (p) => PAD.top + (1 - p) * plotHeight;

    const path = (values) => {
      let d = "";
      for (let i = 0; i < n; i += 1) {
        d += `${i === 0 ? "M" : "L"} ${xOf(i).toFixed(2)} ${yOf(values[i]).toFixed(2)} `;
      }
      return d.trim();
    };

    // The filled area is drawn against the 50% line, so a lead reads as green
    // above the midline and a deficit as amber below it.
    const midline = yOf(0.5);
    const area = `${path(drawn.celtics)} L ${xOf(n - 1).toFixed(2)} ${midline} L ${xOf(0).toFixed(2)} ${midline} Z`;

    // Period boundaries, from the data rather than assumed, so overtime lands
    // in the right place.
    const boundaries = [];
    for (let i = 1; i < n; i += 1) {
      if (events.period[i] !== events.period[i - 1]) {
        boundaries.push({ x: xOf(i), period: events.period[i] });
      }
    }

    return { xOf, yOf, path, area, boundaries, maxElapsed, plotWidth };
  }, [events, drawn, HEIGHT, PAD]);

  function indexFromClientX(clientX) {
    const svg = svgRef.current;
    if (!svg) return 0;
    const rect = svg.getBoundingClientRect();
    const ratio = (clientX - rect.left) / rect.width;
    const target =
      ((ratio * WIDTH - PAD.left) / (WIDTH - PAD.left - PAD.right)) *
      geometry.maxElapsed;

    // Binary search on elapsed time, which is monotone non-decreasing.
    let lo = 0;
    let hi = events.elapsed.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (events.elapsed[mid] < target) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  const atCursor = settledWp(events, meta, cursor);
  const cursorX = geometry.xOf(cursor);
  const cursorY = geometry.yOf(atCursor.value);

  return (
    <svg
      ref={svgRef}
      className="chart"
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      preserveAspectRatio="none"
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId);
        onCursor(indexFromClientX(e.clientX));
      }}
      onPointerMove={(e) => {
        if (e.buttons === 1) onCursor(indexFromClientX(e.clientX));
      }}
      role="slider"
      aria-label="Scrub the game timeline"
      aria-valuemin={0}
      aria-valuemax={events.wp.length - 1}
      aria-valuenow={cursor}
    >
      <defs>
        <linearGradient id="wpFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#14e07a" stopOpacity="0.34" />
          <stop offset="100%" stopColor="#14e07a" stopOpacity="0" />
        </linearGradient>
        <clipPath id="aboveMid">
          <rect
            x={PAD.left}
            y={PAD.top}
            width={WIDTH - PAD.left - PAD.right}
            height={geometry.yOf(0.5) - PAD.top}
          />
        </clipPath>
      </defs>

      {[0, 0.25, 0.5, 0.75, 1].map((p) => (
        <g key={p}>
          <line
            x1={PAD.left}
            x2={WIDTH - PAD.right}
            y1={geometry.yOf(p)}
            y2={geometry.yOf(p)}
            stroke={p === 0.5 ? "#2a3644" : "#161d26"}
            strokeWidth={p === 0.5 ? 1.4 : 1}
            strokeDasharray={p === 0.5 ? "5 5" : ""}
          />
          <text
            x={PAD.left - 8}
            y={geometry.yOf(p) + 4}
            fill="#5d6875"
            fontSize="10"
            textAnchor="end"
          >
            {Math.round(p * 100)}
          </text>
        </g>
      ))}

      {geometry.boundaries.map((b) => (
        <g key={b.period}>
          <line
            x1={b.x}
            x2={b.x}
            y1={PAD.top}
            y2={HEIGHT - PAD.bottom}
            stroke="#1e2732"
            strokeWidth="1"
          />
          <text
            x={b.x + 5}
            y={HEIGHT - PAD.bottom + 15}
            fill="#5d6875"
            fontSize="10"
          >
            {periodLabel(b.period)}
          </text>
        </g>
      ))}
      <text x={PAD.left + 4} y={HEIGHT - PAD.bottom + 15} fill="#5d6875" fontSize="10">
        Q1
      </text>

      <path d={geometry.area} fill="url(#wpFill)" clipPath="url(#aboveMid)" />

      {showBaseline && (
        <path
          d={geometry.path(drawn.generic)}
          fill="none"
          stroke="#6b7787"
          strokeWidth="1.4"
          strokeDasharray="4 4"
          opacity="0.85"
        />
      )}

      <path
        d={geometry.path(drawn.celtics)}
        fill="none"
        stroke="#14e07a"
        strokeWidth="2.1"
        strokeLinejoin="round"
      />

      <line
        x1={cursorX}
        x2={cursorX}
        y1={PAD.top}
        y2={HEIGHT - PAD.bottom}
        stroke="#e8edf3"
        strokeWidth={variant === "compact" ? 1.8 : 1}
        opacity={variant === "compact" ? 0.85 : 0.5}
      />
      <circle cx={cursorX} cy={cursorY} r="5.5" fill="#14e07a" />
      <circle cx={cursorX} cy={cursorY} r="10" fill="#14e07a" opacity="0.18" />
      <title>{percent(atCursor.value)}</title>
    </svg>
  );
}
