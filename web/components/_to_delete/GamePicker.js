"use client";

import { useMemo, useState } from "react";
import TeamLogo from "./TeamLogo";
import { prettyDate, percent } from "@/lib/format";

/**
 * Choose a game.
 *
 * Sorting by "biggest comeback" is not decoration: the lowest out-of-fold
 * probability in a game Boston won is the most informative thing a viewer can
 * look at, because it is where the model was most confident and most wrong.
 */
const SORTS = [
  { key: "date", label: "Date" },
  { key: "comeback", label: "Comebacks" },
  { key: "close", label: "Closest" },
];

export default function GamePicker({ games, seasons, current, onPick,
                                    embedded = false }) {
  const [season, setSeason] = useState(current?.season || seasons[0]);
  const [sort, setSort] = useState("date");
  const [query, setQuery] = useState("");

  const list = useMemo(() => {
    const filtered = games
      .filter((g) => g.season === season)
      .filter((g) =>
        !query
          ? true
          : g.matchup.toLowerCase().includes(query.toLowerCase().trim()) ||
            g.date.includes(query.trim())
      );

    const sorted = [...filtered];
    if (sort === "comeback") {
      // Wins only: a low probability in a loss is the model being right.
      sorted.sort((a, b) => {
        if (a.celtics_won !== b.celtics_won) return a.celtics_won ? -1 : 1;
        return a.lowest_wp - b.lowest_wp;
      });
    } else if (sort === "close") {
      const gap = (g) => Math.abs(g.celtics_final - g.opponent_final);
      sorted.sort((a, b) => gap(a) - gap(b));
    } else {
      sorted.sort((a, b) => a.date.localeCompare(b.date));
    }
    return sorted;
  }, [games, season, sort, query]);

  return (
    <div className={embedded ? "" : "panel"}>
      <div className={`panel__head ${embedded ? "panel__head--bare" : ""}`}>
        {!embedded && <h2 className="panel__title">Games</h2>}
        <span className="note">{list.length} games</span>
      </div>
      <div className="panel__body" style={{ display: "grid", gap: 11 }}>
        <select
          className="field"
          value={season}
          onChange={(e) => setSeason(e.target.value)}
        >
          {seasons.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>

        <input
          className="field"
          placeholder="Search opponent or date"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />

        <div className="tabs">
          {SORTS.map((option) => (
            <button
              key={option.key}
              className={`tab ${sort === option.key ? "tab--on" : ""}`}
              onClick={() => setSort(option.key)}
            >
              {option.label}
            </button>
          ))}
        </div>

        <div className="scroll" style={{ maxHeight: 340 }}>
          {list.map((game) => {
            const active = game.game_id === current?.game_id;
            return (
              <button
                key={game.game_id}
                className={`player ${active ? "player--active" : ""}`}
                onClick={() => onPick(game.game_id)}
              >
                <TeamLogo
                  src={game.opponent_logo}
                  abbr={game.opponent}
                  size={34}
                />
                <div className="player__body">
                  <div className="player__name">{game.matchup}</div>
                  <div className="player__meta">
                    <span>{prettyDate(game.date)}</span>
                    <span
                      className="pill"
                      style={{
                        color: game.celtics_won ? "var(--celtics)" : "var(--text-faint)",
                      }}
                    >
                      {game.celtics_won ? "W" : "L"}{" "}
                      {game.celtics_final}-{game.opponent_final}
                    </span>
                    {game.periods > 4 && <span className="pill">OT</span>}
                  </div>
                </div>
                {sort === "comeback" && (
                  <div className="player__stat">
                    {percent(game.lowest_wp, 1)}
                    <small>low</small>
                  </div>
                )}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
