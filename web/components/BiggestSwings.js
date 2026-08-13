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
export default function BiggestSwings({ current, onPick }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [openClip, setOpenClip] = useState(null);

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

  const swings = data.swings || [];
  return (
    <>
      <p className="drawercount">
        {swings.length} biggest swings · {data.with_clip} with a verified clip ·
        ranked across all seasons
      </p>

      <div className="drawerscroll">
        {swings.map((s, i) => (
          <SwingCard
            key={s.game_id}
            swing={s}
            rank={i + 1}
            expanded={openClip === s.game_id}
            onToggleClip={() =>
              setOpenClip(openClip === s.game_id ? null : s.game_id)
            }
            onOpen={() => onPick(s.game_id)}
            isCurrent={current?.game_id === s.game_id}
          />
        ))}
      </div>

      <p className="note drawerfoot">
        Each row is the single play that moved Boston&apos;s out-of-fold win
        probability the most in that game, Boston made shots only, ranked across
        every season. Out of fold means the model that produced the jump never
        saw this game&apos;s season. Clips are official single-play videos
        confirmed against the player, the game date and the moment; every row
        also opens in the play reconstruction.
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

function SwingCard({ swing, rank, expanded, onToggleClip, onOpen, isCurrent }) {
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

      {swing.clip && (
        <div style={{ marginTop: 10 }}>
          <button
            onClick={onToggleClip}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 8,
              border: "1px solid var(--celtics)",
              background: expanded ? "var(--celtics)" : "transparent",
              color: expanded ? "#04120a" : "var(--celtics)",
              borderRadius: 999,
              padding: "6px 14px",
              fontWeight: 700,
              cursor: "pointer",
            }}
          >
            {expanded ? "▾ Hide clip" : "▶ Watch the play"}
          </button>
          {expanded && (
            <div style={{ marginTop: 10 }}>
              <div
                style={{
                  position: "relative",
                  paddingBottom: "56.25%",
                  height: 0,
                  overflow: "hidden",
                  borderRadius: 10,
                  border: "1px solid var(--line)",
                }}
              >
                <iframe
                  src={`https://www.youtube.com/embed/${swing.clip.video_id}`}
                  title={swing.clip.title}
                  loading="lazy"
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                  allowFullScreen
                  style={{
                    position: "absolute",
                    inset: 0,
                    width: "100%",
                    height: "100%",
                    border: 0,
                  }}
                />
              </div>
              <div
                style={{
                  fontSize: 11,
                  color: "var(--text-faint)",
                  marginTop: 6,
                }}
              >
                Official clip · {swing.clip.channel}
                {" · "}
                <a
                  href={swing.clip.url}
                  target="_blank"
                  rel="noreferrer"
                  style={{ color: "var(--text-dim)" }}
                >
                  open on YouTube
                </a>
              </div>
            </div>
          )}
        </div>
      )}

      <button
        onClick={onOpen}
        style={{
          marginTop: 10,
          border: "1px solid var(--line)",
          background: "transparent",
          color: "var(--text-dim)",
          borderRadius: 8,
          padding: "6px 12px",
          fontSize: 13,
          cursor: "pointer",
        }}
      >
        Open in play reconstruction →
      </button>
    </div>
  );
}
