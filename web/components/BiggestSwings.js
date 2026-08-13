"use client";

import { useEffect, useState } from "react";
import TeamLogo from "./TeamLogo";
import { prettyDate } from "@/lib/format";

async function fetchSwings() {
  const response = await fetch("/data/swings.json");
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}

/**
 * The biggest single-play win-probability swings, ranked across every game.
 *
 * Each row is one play: the largest one-event jump in Boston's OUT-OF-FOLD win
 * probability in that game, kept only when a Boston made shot caused it. Out of
 * fold is the whole point, same as the comeback list — the jump is a forecast
 * from a model that never saw this game's season, not a model reciting a result
 * it was trained on.
 *
 * Where scripts/45_probe_swing_clips.py confirmed an official single-play video
 * of the play, the clip embeds inline. It is not synchronised trickery: the
 * clip IS the play, verified against the player, the game date and the moment.
 * Everything else opens in the play reconstruction, which shows every play.
 */
export default function BiggestSwings({ current, onPick, onWatch }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let alive = true;
    fetchSwings()
      .then((d) => alive && setData(d))
      .catch((e) => alive && setError(String(e)));
    return () => {
      alive = false;
    };
  }, []);

  if (error) {
    return (
      <div className="drawerempty">
        <p className="note">
          Could not load the swings list. Run scripts/46_build_swings.py and
          resync, then reload. ({error})
        </p>
      </div>
    );
  }
  if (!data) {
    return (
      <div className="drawerempty">
        <p className="note">Loading the biggest swings…</p>
      </div>
    );
  }

  // Only the swings that have a verified clip. The rank is the play's true
  // position among ALL biggest swings across every season (computed before the
  // filter), so "#2" still means the second-biggest swing on record, not the
  // second row in this shortened list.
  const clipped = (data.swings || [])
    .map((s, i) => ({ swing: s, rank: i + 1 }))
    .filter((x) => x.swing.clip);

  if (clipped.length === 0) {
    return (
      <div className="drawerempty">
        <p className="note">
          No swings have a verified clip yet. Run
          scripts/45_probe_swing_clips.py on a clear quota day, then
          scripts/46_build_swings.py, and resync.
        </p>
      </div>
    );
  }

  return (
    <>
      <p className="drawercount">
        {clipped.length} plays with a verified clip · ranked across all seasons
      </p>

      <div className="drawerscroll">
        {clipped.map(({ swing, rank }) => (
          <SwingCard
            key={swing.game_id}
            swing={swing}
            rank={rank}
            onWatch={() => onWatch(swing)}
            onOpen={() => onPick(swing.game_id)}
            isCurrent={current?.game_id === swing.game_id}
          />
        ))}
      </div>

      <p className="note drawerfoot">
        Each row is the single play that moved Boston&apos;s out-of-fold win
        probability the most in that game, Boston made shots only, ranked across
        every season. Out of fold means the model that produced the jump never
        saw this game&apos;s season. Only plays with an official single-play clip
        — confirmed against the player, the game date and the moment — are shown
        here. &ldquo;Watch the play&rdquo; opens that game and holds the chart on
        the moment.
      </p>
    </>
  );
}

function fmtClock(raw) {
  if (typeof raw !== "string") return "";
  const m = raw.match(/PT(?:(\d+)M)?(?:([\d.]+)S)?/);
  if (!m) return raw;
  const mins = parseInt(m[1] || "0", 10);
  const secs = Math.round(parseFloat(m[2] || "0"));
  return `${mins}:${String(secs).padStart(2, "0")}`;
}

function SwingCard({ swing, rank, onWatch, onOpen, isCurrent }) {
  const deltaPP = Math.round(swing.delta * 100);
  const before = (swing.wp_before * 100).toFixed(1);
  const after = (swing.wp_after * 100).toFixed(1);
  const won = swing.celtics_won;

  return (
    <div
      style={{
        border: `1px solid ${isCurrent ? "var(--celtics)" : "var(--line)"}`,
        borderRadius: 12,
        background: "var(--bg-raised)",
        padding: "12px 14px",
        marginBottom: 10,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span
          style={{
            fontVariantNumeric: "tabular-nums",
            color: "var(--text-faint)",
            width: 20,
            textAlign: "right",
            fontWeight: 700,
          }}
        >
          {rank}
        </span>
        <TeamLogo src={swing.opponent_logo} abbr={swing.opponent} size={26} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontWeight: 700 }}>
            {swing.celtics_is_home ? "vs" : "at"} {swing.opponent}
            <span
              style={{
                marginLeft: 8,
                fontWeight: 600,
                color: won ? "var(--celtics)" : "var(--warn)",
              }}
            >
              {won ? "W" : "L"}
            </span>{" "}
            <span style={{ color: "var(--text-dim)", fontWeight: 500 }}>
              {swing.celtics_final}–{swing.opponent_final}
              {swing.periods > 4 &&
                ` ${swing.periods === 5 ? "OT" : `${swing.periods - 4}OT`}`}
            </span>
          </div>
          <div
            style={{
              fontSize: 12,
              color: "var(--text-faint)",
            }}
          >
            {prettyDate(swing.date)} · Q{swing.period} {fmtClock(swing.clock)}
          </div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div
            style={{
              color: "var(--celtics)",
              fontWeight: 800,
              fontVariantNumeric: "tabular-nums",
            }}
          >
            +{deltaPP} pts
          </div>
          <div
            style={{
              fontSize: 11,
              color: "var(--text-dim)",
              fontVariantNumeric: "tabular-nums",
            }}
          >
            {before}% → {after}%
          </div>
        </div>
      </div>

      <div
        style={{
          marginTop: 8,
          fontSize: 13,
          color: "var(--text-dim)",
        }}
      >
        <b style={{ color: "var(--text)" }}>{swing.player}</b>
        {swing.player ? " · " : ""}
        {swing.description}
      </div>

      <button
        onClick={onWatch}
        style={{
          marginTop: 10,
          width: "100%",
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
          gap: 8,
          border: "1px solid var(--celtics)",
          background: "var(--celtics)",
          color: "#04120a",
          borderRadius: 10,
          padding: "9px 14px",
          fontWeight: 800,
          cursor: "pointer",
        }}
      >
        ▶ Watch the play
      </button>

      <button
        onClick={onOpen}
        style={{
          marginTop: 8,
          border: "1px solid var(--line)",
          background: "transparent",
          color: "var(--text-dim)",
          borderRadius: 8,
          padding: "6px 12px",
          fontSize: 13,
          cursor: "pointer",
        }}
      >
        Open the game without the clip →
      </button>
    </div>
  );
}
