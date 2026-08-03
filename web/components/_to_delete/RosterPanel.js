"use client";

import { useMemo, useState } from "react";
import PlayerRow from "./PlayerRow";
import { positionGroup, POSITION_GROUPS } from "@/lib/format";

/**
 * The right-hand roster panel: both teams' rosters for this game, searchable
 * and filterable by position, with the on-court five marked.
 *
 * "Add" is deliberately not a button that changes the probability. Who is on
 * the floor is a fact recorded in the play-by-play, not a control, and the
 * model that ships does not use lineup features anyway. Selecting a player
 * highlights them on the court and shows their card; it does not invent a
 * counterfactual the research does not support.
 */
export default function RosterPanel({
  players,
  celticsLineup,
  opponentLineup,
  side,
  onSide,
  selected,
  onSelect,
  positionFilter,
  onPositionFilter,
  embedded = false,
}) {
  const [query, setQuery] = useState("");
  const [metric, setMetric] = useState("minutes");

  const onCourt = new Set(
    side === "celtics" ? celticsLineup : opponentLineup
  );

  const roster = useMemo(() => {
    const wanted = side === "celtics";
    return Object.values(players)
      .filter((p) => p.is_celtics === wanted)
      .filter((p) =>
        !query
          ? true
          : p.name.toLowerCase().includes(query.toLowerCase().trim())
      )
      .filter((p) =>
        !positionFilter ? true : positionGroup(p) === positionFilter
      )
      .sort((a, b) => {
        const aOn = onCourt.has(String(a.person_id)) ? 0 : 1;
        const bOn = onCourt.has(String(b.person_id)) ? 0 : 1;
        if (aOn !== bOn) return aOn - bOn;
        if (metric === "value") {
          return (b.player_value ?? -99) - (a.player_value ?? -99);
        }
        return b.minutes - a.minutes;
      });
  }, [players, side, query, positionFilter, metric, onCourt]);

  return (
    <div className={embedded ? "" : "panel"}>
      <div className={`panel__head ${embedded ? "panel__head--bare" : ""}`}>
        {!embedded && <h2 className="panel__title">Roster</h2>}
        <div className="tabs">
          <button
            className={`tab ${side === "celtics" ? "tab--on tab--on-celtics" : ""}`}
            onClick={() => onSide("celtics")}
          >
            BOS
          </button>
          <button
            className={`tab ${side === "opponent" ? "tab--on tab--on-opponent" : ""}`}
            onClick={() => onSide("opponent")}
          >
            OPP
          </button>
        </div>
      </div>

      <div className="panel__body" style={{ display: "grid", gap: 11 }}>
        <input
          className="field"
          placeholder="Search players"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />

        <div className="row">
          <button
            className={`chip ${!positionFilter ? "chip--on" : ""}`}
            onClick={() => onPositionFilter(null)}
          >
            All
          </button>
          {POSITION_GROUPS.map((group) => (
            <button
              key={group.key}
              className={`chip ${positionFilter === group.key ? "chip--on" : ""}`}
              onClick={() =>
                onPositionFilter(positionFilter === group.key ? null : group.key)
              }
            >
              {group.label}
            </button>
          ))}
        </div>

        <div className="row" style={{ justifyContent: "space-between" }}>
          <span className="note">{roster.length} players</span>
          <div className="tabs">
            <button
              className={`tab ${metric === "minutes" ? "tab--on" : ""}`}
              onClick={() => setMetric("minutes")}
            >
              Minutes
            </button>
            <button
              className={`tab ${metric === "value" ? "tab--on" : ""}`}
              onClick={() => setMetric("value")}
            >
              Value
            </button>
          </div>
        </div>

        <div className="scroll">
          {roster.map((player) => (
            <PlayerRow
              key={player.person_id}
              player={player}
              opponent={side === "opponent"}
              active={
                onCourt.has(String(player.person_id)) ||
                selected === String(player.person_id)
              }
              metric={metric}
              onClick={() =>
                onSelect(
                  selected === String(player.person_id)
                    ? null
                    : String(player.person_id)
                )
              }
            />
          ))}
          {roster.length === 0 && (
            <p className="note">No player matches that search.</p>
          )}
        </div>

        <p className="note" style={{ margin: 0 }}>
          Highlighted rows are on the floor at the current moment. Who is on
          court comes from the reconstructed substitution log, which reproduces
          boxscore minutes to within 0.008 minutes across 13,546 player-games.
        </p>
      </div>
    </div>
  );
}
