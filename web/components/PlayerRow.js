"use client";

import Avatar from "./Avatar";
import { jerseyNumber, signed } from "@/lib/format";

/**
 * One player, used by both the lineup panel and the roster panel.
 *
 * A missing height or position renders as an em dash rather than a plausible
 * default. About 1% of playing time belongs to players with no bio row, and a
 * blank field is the honest way to show that.
 */
export default function PlayerRow({
  player,
  active = false,
  opponent = false,
  metric = "minutes",
  onClick,
  action,
}) {
  const statValue =
    metric === "value"
      ? player.player_value === null
        ? "—"
        : signed(player.player_value, 3)
      : player.minutes.toFixed(1);
  const statLabel = metric === "value" ? "value" : "min";

  return (
    <button
      type="button"
      className={`player ${active ? "player--active" : ""} ${opponent ? "player--opp" : ""}`}
      onClick={onClick}
    >
      <Avatar player={player} />
      <div className="player__body">
        <div className="player__name">
          {player.name}
          {player.is_starter && (
            <span className="pill" style={{ marginLeft: 6 }}>ST</span>
          )}
        </div>
        <div className="player__meta">
          <span className="pill">
            {player.position || player.coarse_position || "—"}
          </span>
          <span>{player.height || "—"}</span>
          {player.jersey && <span>#{jerseyNumber(player.jersey)}</span>}
        </div>
      </div>
      <div className="player__stat">
        {statValue}
        <small>{statLabel}</small>
      </div>
      {action}
    </button>
  );
}
