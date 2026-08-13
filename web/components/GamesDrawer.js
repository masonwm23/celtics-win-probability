"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import TeamLogo from "./TeamLogo";
import { jerseyNumber, percent, prettyDate } from "@/lib/format";
import { comebackSummary, comebackWins } from "@/lib/comebacks";
import BiggestSwings from "./BiggestSwings";

/**
 * Games and rosters, in a drawer over the dashboard.
 *
 * Both used to live at the bottom of the page, which meant scrolling away from
 * the thing you were watching to change the thing you were watching. The
 * dashboard stays visible and mounted behind this, so closing it returns you
 * to exactly the frame you left.
 *
 * Playback pauses while it is open. Changing the game underneath a running
 * clock would leave the cursor pointing into a different game's events for a
 * tick, and the pause is cheaper than defending against that.
 *
 * A right-side drawer on desktop, a full-screen sheet below 900px. That is one
 * CSS rule rather than two components: the content is the same either way.
 */
export default function GamesDrawer({
  open,
  onClose,
  tab,
  onTab,
  games,
  seasons,
  current,
  onPick,
  onWatchSwing,
  players,
  meta,
}) {
  const panelRef = useRef(null);
  const [season, setSeason] = useState(current?.season || seasons[0]);

  // Follow the loaded game when it changes underneath us, so reopening the
  // drawer lands on the season you are actually looking at.
  useEffect(() => {
    if (current?.season) setSeason(current.season);
  }, [current?.season]);

  // Escape closes. Focus moves into the panel so the keyboard is not still
  // driving the timeline behind it.
  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        onClose();
      }
    };
    window.addEventListener("keydown", onKey, true);
    panelRef.current?.focus();
    return () => window.removeEventListener("keydown", onKey, true);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="drawerroot" role="dialog" aria-modal="true" aria-label="Games and rosters">
      <div className="drawerroot__scrim" onClick={onClose} />
      <div className="drawerpanel" ref={panelRef} tabIndex={-1}>
        <header className="drawerpanel__head">
          <div className="drawerpanel__tabs" role="tablist">
            <button
              role="tab"
              aria-selected={tab === "games"}
              className={`drawertab ${tab === "games" ? "drawertab--on" : ""}`}
              onClick={() => onTab("games")}
            >
              Games
            </button>
            <button
              role="tab"
              aria-selected={tab === "rosters"}
              className={`drawertab ${tab === "rosters" ? "drawertab--on" : ""}`}
              onClick={() => onTab("rosters")}
            >
              Rosters
            </button>
          </div>
          <button className="drawerclose" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </header>

        {tab === "games" ? (
          <GamesTab
            games={games}
            seasons={seasons}
            season={season}
            onSeason={setSeason}
            current={current}
            onPick={(id) => {
              onPick(id);
              onClose();
            }}
            onWatchSwing={(swing) => {
              onWatchSwing(swing);
              onClose();
            }}
          />
        ) : (
          <RostersTab
            season={season}
            seasons={seasons}
            onSeason={setSeason}
            current={current}
            players={players}
            meta={meta}
            onGoToGames={() => onTab("games")}
          />
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Games
// ---------------------------------------------------------------------------

function GamesTab({ games, seasons, season, onSeason, current, onPick, onWatchSwing }) {
  const [query, setQuery] = useState("");
  const [view, setView] = useState("schedule");

  const list = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return games
      .filter((g) => g.season === season)
      .filter((g) =>
        !needle
          ? true
          : g.opponent.toLowerCase().includes(needle) ||
            g.matchup.toLowerCase().includes(needle) ||
            g.date.includes(needle) ||
            prettyDate(g.date).toLowerCase().includes(needle)
      )
      .sort((a, b) => a.date.localeCompare(b.date));
  }, [games, season, query]);

  const record = list.reduce(
    (acc, g) => (g.celtics_won ? { ...acc, w: acc.w + 1 } : { ...acc, l: acc.l + 1 }),
    { w: 0, l: 0 }
  );

  return (
    <div className="drawerbody">
      <div className="drawercontrols">
        <label className="drawerfield">
          <span>Season</span>
          <select
            className="field"
            value={season}
            onChange={(e) => onSeason(e.target.value)}
          >
            {seasons.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </label>
        {view === "schedule" && (
          <label className="drawerfield drawerfield--grow">
            <span>Search</span>
            <input
              className="field"
              placeholder="Opponent or date, e.g. DEN or 2024-03"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </label>
        )}
        <div className="drawerfield">
          <span>View</span>
          <div className="seg">
            <button
              className={`seg__btn ${view === "schedule" ? "seg__btn--on" : ""}`}
              onClick={() => setView("schedule")}
            >
              Schedule
            </button>
            <button
              className={`seg__btn ${view === "comebacks" ? "seg__btn--on" : ""}`}
              onClick={() => setView("comebacks")}
            >
              Comebacks
            </button>
            <button
              className={`seg__btn ${view === "swings" ? "seg__btn--on" : ""}`}
              onClick={() => setView("swings")}
            >
              Swings
            </button>
          </div>
        </div>
      </div>

      {view === "comebacks" ? (
        <ComebacksView
          games={games}
          season={season}
          current={current}
          onPick={onPick}
        />
      ) : view === "swings" ? (
        <BiggestSwings current={current} onPick={onPick} onWatch={onWatchSwing} />
      ) : (
        <Schedule list={list} record={record} query={query} season={season}
                  current={current} onPick={onPick} />
      )}
    </div>
  );
}

function Schedule({ list, record, query, season, current, onPick }) {
  return (
    <>
      <p className="drawercount">
        {list.length} game{list.length === 1 ? "" : "s"} · {record.w}-{record.l}
        {query && " matching"}
      </p>

      <div className="drawerscroll">
        {list.length === 0 && (
          <p className="note" style={{ padding: "16px 4px" }}>
            Nothing matches that search in {season}.
          </p>
        )}
        {list.map((game) => {
          const won = game.celtics_won;
          const bos = game.celtics_final;
          const opp = game.opponent_final;
          return (
            <button
              key={game.game_id}
              className={`gamerow ${
                current?.game_id === game.game_id ? "gamerow--on" : ""
              }`}
              onClick={() => onPick(game.game_id)}
            >
              <span className="gamerow__date">{prettyDate(game.date)}</span>
              <TeamLogo
                src={game.opponent_logo}
                abbr={game.opponent}
                size={24}
                className="gamerow__logo"
              />
              <span className="gamerow__team">
                <b>{game.opponent}</b>
                <span className="gamerow__venue">
                  {game.celtics_is_home ? "vs · home" : "at · away"}
                </span>
              </span>
              <span className={`gamerow__wl ${won ? "is-w" : "is-l"}`}>
                {won ? "W" : "L"}
              </span>
              <span className="gamerow__score">
                {bos}–{opp}
              </span>
              {game.periods > 4 && (
                <span className="gamerow__ot">
                  {game.periods === 5 ? "OT" : `${game.periods - 4}OT`}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </>
  );
}

/**
 * The season's comeback wins, longest odds first.
 *
 * The number that orders the list is the LOWEST out-of-fold win probability
 * the game ever reached, among games Boston won. Top of the list is the night
 * the model was most sure they were gone.
 *
 * Out of fold matters here more than anywhere else in the dashboard: each of
 * those probabilities comes from a model fitted on the other seven seasons, so
 * "0.18%" is a real prediction about a game the model had never seen, not a
 * model reciting an outcome it was trained on.
 *
 * The points deficit is shown beside it when the index carries it, because
 * "down 32, and the model had them at 0.9%" is the whole story in one line. It
 * does not affect the order.
 */
function ComebacksView({ games, season, current, onPick }) {
  const rows = useMemo(() => comebackWins(games, season), [games, season]);
  const summary = useMemo(() => comebackSummary(games, season), [games, season]);

  if (rows.length === 0) {
    return (
      <div className="drawerempty">
        <p className="note">
          Boston won {summary.wins} games in {season}, and the model never had
          them below even money in any of them, so there is no comeback to
          rank.
        </p>
      </div>
    );
  }

  return (
    <>
      <p className="drawercount">
        {rows.length} win{rows.length === 1 ? "" : "s"} in {season} after the
        model had Boston below 50% · longest odds{" "}
        {rows[0].odds || rows[0].lowestLabel} · {summary.neverBehind} win
        {summary.neverBehind === 1 ? "" : "s"} never dipped below
      </p>

      <div className="drawerscroll">
        {rows.map((game, i) => (
          <button
            key={game.game_id}
            className={`cbrow ${
              current?.game_id === game.game_id ? "cbrow--on" : ""
            }`}
            onClick={() => onPick(game.game_id)}
            title={`Model low ${game.lowestLabel}${
              game.odds ? ` (${game.odds})` : ""
            }, won ${game.celtics_final}-${game.opponent_final}`}
          >
            <span className="cbrow__rank">{i + 1}</span>
            <span className="cbrow__low">
              <b>{game.lowestLabel}</b>
              <small>model low</small>
            </span>
            <TeamLogo
              src={game.opponent_logo}
              abbr={game.opponent}
              size={24}
              className="cbrow__logo"
            />
            <span className="cbrow__team">
              <b>
                {game.celtics_is_home ? "vs" : "at"} {game.opponent}
              </b>
              <span className="cbrow__when">
                {prettyDate(game.date)}
                {game.odds ? ` · ${game.odds}` : ""}
                {game.moment ? ` · down ${game.largest_deficit} at ${game.moment}` : ""}
              </span>
            </span>
            <span className="cbrow__score">
              {game.celtics_final}–{game.opponent_final}
              {game.periods > 4 && (
                <i>{game.periods === 5 ? "OT" : `${game.periods - 4}OT`}</i>
              )}
            </span>
          </button>
        ))}
      </div>

      <p className="note drawerfoot">
        Ranked by the lowest out-of-fold win probability the game reached, in
        games the Celtics won. Out of fold means the model that produced it was
        fitted on the other seven seasons and never saw this game. Wins in
        which the probability never fell below 50% are not listed.
        {summary.largestDeficit > 0
          ? " The points deficit on each row is from the play-by-play score."
          : " Run scripts/42_comeback_index.py to also show the points deficit on each row."}
      </p>
    </>
  );
}

// ---------------------------------------------------------------------------
// Rosters
// ---------------------------------------------------------------------------

/**
 * The two rosters for the game that is loaded.
 *
 * Deliberately not a season-wide roster. The serving payload carries the
 * boxscore roster for each GAME, which is who actually dressed and played that
 * night; a season roster is a different thing and is not in this data. So when
 * the season selector points somewhere other than the loaded game, this says so
 * and offers the games list rather than showing a roster from the wrong season.
 */
function RostersTab({ season, seasons, onSeason, current, players, meta, onGoToGames }) {
  const [side, setSide] = useState("celtics");

  const roster = useMemo(() => {
    const wanted = side === "celtics";
    return Object.values(players || {})
      .filter((p) => Boolean(p.is_celtics) === wanted)
      .sort((a, b) => (b.minutes || 0) - (a.minutes || 0));
  }, [players, side]);

  const inSeason = current?.season === season;

  return (
    <div className="drawerbody">
      <div className="drawercontrols">
        <label className="drawerfield">
          <span>Season</span>
          <select className="field" value={season} onChange={(e) => onSeason(e.target.value)}>
            {seasons.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </label>
        <div className="drawerfield drawerfield--grow">
          <span>Team</span>
          <div className="seg">
            <button
              className={`seg__btn ${side === "celtics" ? "seg__btn--on" : ""}`}
              onClick={() => setSide("celtics")}
            >
              Celtics
            </button>
            <button
              className={`seg__btn ${side === "opponent" ? "seg__btn--on" : ""}`}
              onClick={() => setSide("opponent")}
            >
              {meta?.opponent || "Opponent"}
            </button>
          </div>
        </div>
      </div>

      {!inSeason ? (
        <div className="drawerempty">
          <p className="note">
            Rosters come from a game&apos;s own boxscore, which is who actually
            dressed and played that night. The loaded game is{" "}
            <strong>{current?.matchup}</strong> in{" "}
            <strong>{current?.season}</strong>, so there is nothing to show for{" "}
            <strong>{season}</strong> until you pick a game from it.
          </p>
          <button className="chip chip--on" onClick={onGoToGames}>
            Choose a {season} game
          </button>
        </div>
      ) : (
        <>
          <p className="drawercount">
            {meta?.matchup} · {prettyDate(current?.date)} · {roster.length} players
          </p>
          <div className="drawerscroll">
            {roster.map((player) => (
              <div className="rosterrow" key={player.person_id}>
                <Face player={player} />
                <span className="rosterrow__name">
                  {player.name}
                  {player.is_starter && <i className="rosterrow__st">ST</i>}
                </span>
                <span className="rosterrow__meta">
                  {player.position || player.coarse_position || "—"}
                </span>
                <span className="rosterrow__meta">{player.height || "—"}</span>
                <span className="rosterrow__meta">
                  {player.jersey ? `#${jerseyNumber(player.jersey)}` : "—"}
                </span>
                <span className="rosterrow__min">
                  {(player.minutes ?? 0).toFixed(1)}
                </span>
              </div>
            ))}
          </div>
          <p className="note drawerfoot">
            Position and height come from the season bio pull and are blank
            rather than guessed when that player has no bio row. Minutes are
            from this game&apos;s boxscore.
          </p>
        </>
      )}
    </div>
  );
}

function Face({ player }) {
  const [broken, setBroken] = useState(() => new Set());
  const url = [player?.headshot, player?.headshot_current].find(
    (candidate) => candidate && !broken.has(candidate)
  );
  if (!url) {
    return (
      <span className="rosterface rosterface--fallback" aria-hidden="true">
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
      className="rosterface"
      src={url}
      alt=""
      loading="lazy"
      onError={() => setBroken((prev) => new Set(prev).add(url))}
    />
  );
}

/**
 * The button that opens it.
 *
 * Always shows the loaded context, so the answer to "which game am I looking
 * at" is on screen without opening anything.
 */
export function GamesDrawerButton({ current, onOpen, compact = false }) {
  const label = current
    ? `${current.season} · ${current.matchup}`
    : "Games & rosters";
  return (
    <button
      className={`drawerbtn ${compact ? "drawerbtn--compact" : ""}`}
      onClick={onOpen}
      aria-haspopup="dialog"
    >
      <span className="drawerbtn__icon" aria-hidden="true">▦</span>
      <span className="drawerbtn__text">
        <span className="drawerbtn__title">Games &amp; rosters</span>
        {!compact && <span className="drawerbtn__ctx">{label}</span>}
      </span>
    </button>
  );
}
