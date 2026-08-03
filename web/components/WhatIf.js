"use client";

import { useState } from "react";
import { fetchWhatIf } from "@/lib/api";
import { percent, signed } from "@/lib/format";

/**
 * What-if: re-predict the current moment with the score margin changed.
 *
 * TWO DIFFERENT NUMBERS LIVE ON THIS PAGE and this panel is where they meet, so
 * the distinction is spelled out on screen rather than assumed:
 *
 *   - the timeline probability is OUT OF FOLD, from a model that never saw this
 *     season;
 *   - this one comes from the deployment model, which was fitted on all eight
 *     seasons and is therefore in-sample for this game.
 *
 * The API returns a `caveat` string with every what-if response and it is
 * rendered verbatim below, so the warning cannot drift out of step with the
 * backend.
 *
 * The margin is what the reader drags, but it is not the only column that
 * moves, and it must not be. Two of the thirteen features are FUNCTIONS of the
 * margin: `margin_per_minute_remaining` divides it by the minutes left, and
 * `is_clutch` is the NBA definition, which tests whether the game is within
 * five points. An earlier version overrode the margin alone and left those two
 * holding values computed from the real score, which describes no possible game
 * and got the answer such a row deserves: on a real second-quarter event,
 * dragging from -12 to -32 RAISED the win probability from 31.9% to 32.5%, and
 * asking for +20 returned 21.8%. The API now recomputes them and reports what
 * it changed, which this panel shows.
 *
 * Everything genuinely independent of the margin still keeps its real value, so
 * the answer stays about the change that was asked for rather than about a
 * dozen invented features.
 */
export default function WhatIf({ gameId, eventIndex, actualMargin, actualWp }) {
  const [delta, setDelta] = useState(0);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const hypothetical = actualMargin + delta;
  // Named by the API rather than listed here, so the panel cannot claim a
  // recomputation the backend did not perform.
  const recomputed = Object.keys(result?.derived_recomputed || {});

  async function run(nextDelta) {
    setDelta(nextDelta);
    if (nextDelta === 0) {
      setResult(null);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const response = await fetchWhatIf(gameId, eventIndex, {
        celtics_margin: actualMargin + nextDelta,
      });
      setResult(response);
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="panel">
      <div className="panel__head">
        <h2 className="panel__title">What if the margin were different</h2>
        <span className="badge badge--warn">in-sample</span>
      </div>
      <div className="panel__body" style={{ display: "grid", gap: 12 }}>
        <div className="row" style={{ justifyContent: "space-between" }}>
          <span className="note">
            Actual margin <strong>{signed(actualMargin)}</strong>
          </span>
          <span className="note">
            Hypothetical <strong>{signed(hypothetical)}</strong>
          </span>
        </div>

        <input
          className="scrub"
          type="range"
          min={-20}
          max={20}
          step={1}
          value={delta}
          onChange={(e) => run(Number(e.target.value))}
        />

        <div className="impact">
          <div className="impact__card">
            <div className="impact__label">Actual, out of fold</div>
            <div className="impact__value" style={{ color: "var(--celtics)" }}>
              {percent(actualWp)}
            </div>
            <div className="impact__sub">model never saw this season</div>
          </div>
          <div className="impact__card">
            <div className="impact__label">Hypothetical</div>
            <div className="impact__value">
              {busy ? "…" : result ? percent(result.probability) : "—"}
            </div>
            <div className="impact__sub">deployment model</div>
          </div>
        </div>

        {error && <p className="note" style={{ color: "var(--warn)" }}>{error}</p>}

        {result && (
          <p className="note" style={{ margin: 0 }}>
            <strong>Read this carefully.</strong> {result.caveat} You changed{" "}
            <code>celtics_margin</code>
            {recomputed.length > 0 && (
              <>
                , and{" "}
                {recomputed.map((name, i) => (
                  <span key={name}>
                    {i > 0 && " and "}
                    <code>{name}</code>
                  </span>
                ))}{" "}
                moved with it, because {recomputed.length > 1 ? "they are" : "it is"}{" "}
                calculated from the margin rather than measured separately
              </>
            )}
            . Everything else keeps its real value for this event.
          </p>
        )}

        {!result && !error && (
          <p className="note" style={{ margin: 0 }}>
            Drag to replace the score margin at this moment and ask the
            deployment model what it would have said. The two features
            calculated from the margin move with it; everything measured
            independently stays real, because inventing those would produce a
            confident number about a game state nobody described.
          </p>
        )}
      </div>
    </div>
  );
}
