"use client";

import { useMemo, useState } from "react";
import { jerseyNumber } from "@/lib/format";
import { pointsAt, pointsOnEvent, pointsTimeline, reboundTimeline } from "@/lib/scoring";
import { assistTimeline } from "@/lib/assists";

/**
 * The ten players on the floor, with their totals through this exact play.
 *
 * Sits beside the court so it is readable while playback runs, rather than at
 * the bottom of the page where you would have to scroll away from the thing
 * you are watching. It follows the reconstructed substitution log, so the five
 * a side change when the game changes them, not when the quarter does.
 *
 * WHAT IS SHOWN, AND WHY NOT MORE
 * -------------------------------
 * Points and rebounds. Both are running totals through the selected event, and
 * both were reconciled against the boxscore across all 636 games before being
 * put on screen: points match in 636 of 636, rebounds in 635 of 636.
 *
 * Assists now too. They were held back at 554 of 636 games reconciling, which
 * was not good enough to display; lib/assists.js closes that to 635 of 636 by
 * trying the exact surname before any suffix-stripped form and by deriving a
 * per-game alias map from the game's own rows. The one game that still
 * disagrees is a play-by-play and boxscore disagreement, not a resolver
 * failure.
 */
export default function OnCourtBox({
  events,
  cursor,
  players,
  meta,
  celticsLineup,
  opponentLineup,
}) {
  const points = useMemo(() => pointsTimeline(events), [events]);
  const rebounds = useMemo(() => reboundTimeline(events), [events]);
  const assists = useMemo(() => assistTimeline(events, players), [events, players]);

  const involved = events.person_id[cursor];

  const row = (id) => {
    const player = players[id];
    if (!player) return null;
    return {
      player,
      points: pointsAt(points, player.person_id, cursor),
      rebounds: pointsAt(rebounds, player.person_id, cursor),
      assists: pointsAt(assists, player.person_id, cursor),
      scored: pointsOnEvent(points, player.person_id, cursor),
      involved: player.person_id === involved,
    };
  };

  const boston = (celticsLineup || []).map(row).filter(Boolean);
  const opponent = (opponentLineup || []).map(row).filter(Boolean);

  return (
    <aside className="oncourt">
      <div className="oncourt__head">
        <span className="oncourt__title">On the floor</span>
        <span className="oncourt__scope">through play {cursor + 1}</span>
      </div>
      {/* Side by side rather than stacked: stacked, the box was the tallest
          thing in the rail and pushed the playback controls off a 1440-wide
          laptop screen. */}
      <div className="oncourt__pair">
        <Side label="BOS" rows={boston} celtics />
        <Side label={meta.opponent} rows={opponent} />
      </div>
    </aside>
  );
}

function Side({ label, rows, celtics = false }) {
  return (
    <div className={`oncourt__side ${celtics ? "is-bos" : "is-opp"}`}>
      <div className="oncourt__team">
        <span>{label}</span>
        <span className="oncourt__cols">
          <b>PTS</b>
          <b>REB</b>
          <b>AST</b>
        </span>
      </div>
      {rows.length === 0 && (
        <p className="oncourt__empty">No lineup recorded for this event</p>
      )}
      {rows.map((row) => (
        <div
          key={row.player.person_id}
          className={`ocrow ${row.involved ? "ocrow--involved" : ""}`}
        >
          <Face player={row.player} />
          <span className="ocrow__name">
            {row.player.name}
            {row.player.jersey && (
              <span className="ocrow__num"> #{jerseyNumber(row.player.jersey)}</span>
            )}
          </span>
          <span className="ocrow__stat">
            {row.points}
            {row.scored > 0 && <i className="ocrow__plus">+{row.scored}</i>}
          </span>
          <span className="ocrow__stat ocrow__stat--dim">{row.rebounds}</span>
          <span className="ocrow__stat ocrow__stat--dim">{row.assists}</span>
        </div>
      ))}
      {rows.length > 0 && (
        <div className="ocrow ocrow--total">
          <span />
          <span className="ocrow__name">On court</span>
          <span className="ocrow__stat">
            {rows.reduce((sum, r) => sum + r.points, 0)}
          </span>
          <span className="ocrow__stat ocrow__stat--dim">
            {rows.reduce((sum, r) => sum + r.rebounds, 0)}
          </span>
          <span className="ocrow__stat ocrow__stat--dim">
            {rows.reduce((sum, r) => sum + r.assists, 0)}
          </span>
        </div>
      )}
    </div>
  );
}

/**
 * The season-correct headshot, then the current one, then initials.
 *
 * Never a silhouette: a silhouette reads as "this player" while initials read
 * as "no photo", and only the second is true.
 */
function Face({ player }) {
  const [broken, setBroken] = useState(() => new Set());
  const url = [player?.headshot, player?.headshot_current].find(
    (candidate) => candidate && !broken.has(candidate)
  );
  if (!url) {
    return (
      <span className="ocface ocface--fallback" aria-hidden="true">
        {String(player?.name || "")
          .split(/\s+/)
          .filter(Boolean)
          .slice(0, 2)
          .map((part) => part[0])
          .join("")
          .toUpperCase()}
      </span>
    );
  }
  return (
    <img
      className="ocface"
      src={url}
      alt=""
      loading="lazy"
      onError={() => setBroken((prev) => new Set(prev).add(url))}
    />
  );
}
