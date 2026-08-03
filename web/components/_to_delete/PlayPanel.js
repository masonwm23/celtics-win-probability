"use client";

import { useMemo, useState } from "react";
import PlayCourt from "@/components/PlayCourt";
import MiniProbability from "@/components/MiniProbability";
import OnCourtBox from "@/components/OnCourtBox";
import { formatClock, jerseyNumber, periodLabel } from "@/lib/format";
import { DISCLOSURE, displayedChange } from "@/lib/reconstruct";
import { describePlay, reboundKindsForGame } from "@/lib/playby";
import { finalHeadline, finalSentence, gameOutcome } from "@/lib/outcome";
import {
  LEGEND_BOX,
  MADE_CLASS,
  MADE_RADIUS,
  MISS_CLASS,
  legendMissLines,
} from "@/lib/marks";

/**
 * Live play reconstruction.
 *
 * Court on the left, one card on the right, one control bar underneath. Every
 * part of it moves off the SAME cursor, so pressing Play advances the court,
 * the description, the big chart above and the small chart in the card in one
 * step. There is no second source of truth to fall out of sync with.
 *
 * The split that runs through the whole panel:
 *
 *   VERIFIED    lineup, clock, score, the event, the shot coordinate, and both
 *               probabilities. All recorded, all out of fold.
 *   SCHEMATIC   where the ten circles sit. A team sheet, not a position.
 *
 * That distinction is stated on screen rather than left to documentation,
 * because a diagram that looks like tracking data will be read as tracking
 * data whatever a README says.
 */
export default function PlayPanel({
  events,
  cursor,
  total,
  players,
  meta,
  celticsLineup,
  opponentLineup,
  playing,
  onPlayPause,
  onStep,
  onCursor,
  speed,
  speeds,
  onSpeed,
  ballMs,
}) {
  const [mode, setMode] = useState("current");
  const [howTo, setHowTo] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [fullHistory, setFullHistory] = useState(false);
  const [shotFilter, setShotFilter] = useState({
    team: "all",
    result: "all",
    value: "all",
    period: "all",
  });

  const player = players[events.person_id[cursor]] || null;

  // A rebound's type is recovered by differencing each player's own running
  // counters across the game, so it is computed once here rather than per
  // render. See lib/playby.js for why a single row cannot answer it.
  const reboundKinds = useMemo(() => reboundKindsForGame(events), [events]);

  // Null until the very last event, so nothing downstream can show a winner
  // part way through a game.
  const outcome = gameOutcome(events, meta, cursor);

  return (
    <section className="panel playpanel">
      <header className="playpanel__head">
        <h2 className="panel__title">Live play reconstruction</h2>
        <div className="row">
          <div className="seg" role="group" aria-label="What the court shows">
            <button
              className={`seg__btn ${mode === "current" ? "seg__btn--on" : ""}`}
              onClick={() => setMode("current")}
            >
              Current play
            </button>
            <button
              className={`seg__btn ${mode === "recent" ? "seg__btn--on" : ""}`}
              onClick={() => setMode("recent")}
            >
              Recent shots
            </button>
          </div>
          <button
            className={`ghostbtn ${howTo ? "ghostbtn--on" : ""}`}
            onClick={() => setHowTo((v) => !v)}
            aria-expanded={howTo}
          >
            <span className="ghostbtn__i">i</span> How to read
          </button>
        </div>
      </header>

      {howTo && <HowToRead />}

      <div className="playpanel__main">
        <div className="playpanel__left">
          <div className="playpanel__court">
            <PlayCourt
              events={events}
              cursor={cursor}
              players={players}
              celticsLineup={celticsLineup}
              opponentLineup={opponentLineup}
              opponentAbbrev={meta.opponent}
              mode={mode}
              shotFilter={shotFilter}
              fullHistory={fullHistory}
              ballMs={ballMs}
            />
          </div>
          <CourtLegend opponent={meta.opponent} />
          <p className="disclosure">
            <span className="disclosure__mark">i</span>
            {DISCLOSURE}
          </p>
        </div>

        {/* The right column, beside the court rather than below it. The live
            box sits FIRST because the whole point of it is being readable
            while playback runs; putting it under the card would have meant
            scrolling away from the thing you are watching. */}
        <div className="playpanel__right">
          <OnCourtBox
            events={events}
            cursor={cursor}
            players={players}
            meta={meta}
            celticsLineup={celticsLineup}
            opponentLineup={opponentLineup}
          />
          <CurrentPlayCard
            events={events}
            cursor={cursor}
            players={players}
            player={player}
            meta={meta}
            celticsLineup={celticsLineup}
            opponentLineup={opponentLineup}
            reboundType={reboundKinds.get(cursor) ?? null}
            outcome={outcome}
          />
        </div>
      </div>

      <Transport
        events={events}
        cursor={cursor}
        total={total}
        playing={playing}
        onPlayPause={onPlayPause}
        onStep={onStep}
        onCursor={onCursor}
        speed={speed}
        speeds={speeds}
        onSpeed={onSpeed}
      />

      <div className="drawer">
        <button
          className="drawer__toggle"
          onClick={() => setDrawerOpen((v) => !v)}
          aria-expanded={drawerOpen}
        >
          Shot filters
          <span className={`drawer__caret ${drawerOpen ? "drawer__caret--on" : ""}`}>
            ⌄
          </span>
        </button>
        {drawerOpen && (
          <ShotFilters
            mode={mode}
            value={shotFilter}
            onChange={setShotFilter}
            fullHistory={fullHistory}
            onFullHistory={setFullHistory}
          />
        )}
      </div>
    </section>
  );
}

/**
 * The four things on the court that mean something, named.
 *
 * Small and always visible rather than collapsed, because a court that needs a
 * hidden key is a court nobody reads correctly.
 */
function CourtLegend({ opponent }) {
  return (
    <div className="clegend">
      <span><i className="lg lg--bos" /> BOS player</span>
      <span><i className="lg lg--opp" /> {opponent} player</span>
      <span><MarkSwatch made /> Shot made</span>
      <span><MarkSwatch /> Shot missed</span>
      <span><i className="lg lg--lit" /> Involved in this play</span>
    </div>
  );
}

/**
 * The court's own shot marker, shrunk.
 *
 * Drawn from lib/marks.js in a box the same size as the court's arm length,
 * wearing the same class names, and scaled down only by the SVG viewport. The
 * colour, stroke weight, round caps and arm-to-stroke ratio therefore come out
 * identical to the marker on the floor rather than being re-specified in CSS
 * and drifting. The previous legend used a single CSS diagonal, so the key
 * showed a slash where the court showed a cross.
 */
function MarkSwatch({ made = false }) {
  return (
    <svg
      className="lgmark"
      viewBox={`0 0 ${LEGEND_BOX} ${LEGEND_BOX}`}
      aria-hidden="true"
    >
      {made ? (
        <circle
          cx={LEGEND_BOX / 2}
          cy={LEGEND_BOX / 2}
          r={MADE_RADIUS}
          className={MADE_CLASS}
        />
      ) : (
        <g className={MISS_CLASS}>
          {legendMissLines().map((line, i) => (
            <line key={i} {...line} />
          ))}
        </g>
      )}
    </svg>
  );
}

/**
 * The right-hand card: who, what, when, and what it was worth.
 *
 * `Before`, `After` and `Change` come from the same two points the small chart
 * draws its step between, so the number and the picture cannot disagree.
 */
function CurrentPlayCard({
  events,
  cursor,
  players,
  player,
  meta,
  celticsLineup,
  opponentLineup,
  reboundType,
  outcome,
}) {
  const action = events.action_type[cursor];
  const description = events.description[cursor] || "";
  // One rounding, shared with the small chart, so the step drawn there and the
  // three numbers below it are the same measurement.
  const change = displayedChange(events, cursor);

  const isFreeThrow = action === "Free Throw";
  const hasCoords =
    (action === "Made Shot" || action === "Missed Shot") &&
    !(events.loc_x[cursor] === 0 && events.loc_y[cursor] === 0);

  // The detailed reading of the feed's own text: "Defensive rebound" rather
  // than "Rebound", "Shooting foul" rather than "Foul", and the turnover's
  // recorded cause. Anything the feed does not state comes back as a note
  // instead of being filled in.
  const play = describePlay(
    {
      action_type: action,
      shot_value: events.shot_value[cursor],
      description,
    },
    { reboundType }
  );

  const assistName = assistLabel(description);
  const assister = assistName
    ? findByLabel(assistName, [...celticsLineup, ...opponentLineup], players)
    : null;

  const bos = events.celtics_score[cursor];
  const opp = events.opponent_score[cursor];
  const home = meta.celtics_is_home;

  const location = isFreeThrow
    ? ["Rule-defined", "warn"]
    : hasCoords
      ? ["Verified", "ok"]
      : ["No shot recorded", "none"];

  return (
    <aside className="playcard">
      <div className="playcard__eyebrow">Current play</div>

      {outcome?.isFinal && (
        <div className={`finalstate ${outcome.tie ? "finalstate--tie" : ""}`}>
          <div className="finalstate__head">{finalHeadline(outcome)}</div>
          <div className="finalstate__sub">{finalSentence(outcome)}</div>
          <div className="finalstate__meta">
            {outcome.periodEnd}
            {outcome.scoreCheck && !outcome.scoresAgree && (
              <span className="finalstate__warn">
                {" "}
                · the event score and the boxscore disagree
              </span>
            )}
          </div>
        </div>
      )}

      <div className="playcard__who">
        <PlayerFace player={player} />
        <div className="playcard__whotext">
          <h3 className="playcard__name">
            {player
              ? player.name
              : outcome?.isFinal
                ? "Game complete"
                : "No player recorded"}{" "}
            <span className="playcard__what">
              {outcome?.isFinal && !player
                ? outcome.tie
                  ? "recorded scores are tied"
                  : `${outcome.winnerName} defeat ${outcome.loserName}`
                : lowerFirst(play.label)}
              {play.detail && (
                <span className="playcard__cause"> — {play.detail}</span>
              )}
            </span>
          </h3>
          {play.note && <p className="playcard__unknown">{play.note}</p>}
          <p className="playcard__assist">
            {assistName ? (
              <>
                Assist: <b>{assister ? assister.name : assistName}</b>
              </>
            ) : (
              <span className="is-dim">No assist recorded</span>
            )}
          </p>
        </div>
      </div>

      <div className="playcard__chips">
        <span className="chipflat">
          {periodLabel(events.period[cursor])} {formatClock(events.clock[cursor])}
        </span>
        <span className="chipflat">
          {home
            ? `${meta.opponent} ${opp} – ${bos} BOS`
            : `BOS ${bos} – ${opp} ${meta.opponent}`}
        </span>
        {player?.jersey && (
          <span className="chipflat">#{jerseyNumber(player.jersey)}</span>
        )}
      </div>

      <MiniProbability events={events} cursor={cursor} />

      {change && (
        <>
          <p className={`movement movement--${change.direction}`}>
            Before <b>{change.before.toFixed(1)}%</b>
            <span className="movement__arrow"> → </span>
            After <b>{change.after.toFixed(1)}%</b>
            <span className="movement__dash"> — </span>
            Change{" "}
            <b>
              {change.points > 0 ? "+" : change.points < 0 ? "−" : "±"}
              {Math.abs(change.points).toFixed(1)} percentage points
            </b>
          </p>
          <div className="deltas">
            <div className="delta">
              <div className="delta__label">Before</div>
              <div className="delta__value">{change.before.toFixed(1)}%</div>
            </div>
            <div className="delta">
              <div className="delta__label">After</div>
              <div className="delta__value">{change.after.toFixed(1)}%</div>
            </div>
            <div className={`delta delta--${change.direction}`}>
              <div className="delta__label">Change</div>
              <div className="delta__value">
                {change.points > 0 ? "+" : change.points < 0 ? "−" : "±"}
                {Math.abs(change.points).toFixed(1)}
              </div>
              <div className="delta__unit">percentage points</div>
            </div>
          </div>
        </>
      )}

      <dl className="playcard__rows">
        <Row label="Shot location" value={location[0]} tone={location[1]} />
        <Row
          label="Lineup"
          value={
            celticsLineup.length + opponentLineup.length === 10
              ? "Verified"
              : `${celticsLineup.length + opponentLineup.length} of 10 on file`
          }
          tone={
            celticsLineup.length + opponentLineup.length === 10 ? "ok" : "warn"
          }
        />
      </dl>

      {description && <p className="playcard__feed">{description}</p>}
    </aside>
  );
}

/**
 * The player's face, or their initials.
 *
 * Never a generic silhouette. A silhouette reads as "this player" and initials
 * read as "no photo", and only the second one is true.
 */
function PlayerFace({ player }) {
  const [broken, setBroken] = useState(() => new Set());
  const url = [player?.headshot, player?.headshot_current].find(
    (candidate) => candidate && !broken.has(candidate)
  );
  if (!player) return <div className="face face--empty" aria-hidden="true" />;
  if (!url) {
    return (
      <div className="face face--fallback" aria-hidden="true">
        {initials(player.name)}
      </div>
    );
  }
  return (
    <img
      className="face"
      src={url}
      alt=""
      loading="lazy"
      onError={() => setBroken((prev) => new Set(prev).add(url))}
    />
  );
}

/** "Made 3PT" -> "made 3PT". Only the first letter, so 3PT keeps its case. */
function lowerFirst(text) {
  const value = String(text || "");
  return value ? value[0].toLowerCase() + value.slice(1) : value;
}

function initials(name) {
  return String(name || "")
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0])
    .join("")
    .toUpperCase();
}

function Row({ label, value, tone }) {
  return (
    <div className="playcard__row">
      <dt>{label}</dt>
      <dd className={tone ? `is-${tone}` : ""}>{value}</dd>
    </div>
  );
}

/** The assisting player as the feed wrote them, or null. */
function assistLabel(description) {
  const match = String(description || "").match(/\(([^()]+?)\s+\d+\s+AST\)/);
  return match ? match[1].trim() : null;
}

function findByLabel(label, lineup, players) {
  const wanted = label.toLowerCase().replace(/\./g, "").trim();
  const hits = lineup
    .map((id) => players?.[id])
    .filter(Boolean)
    .filter((p) => {
      const name = String(p.name || "").toLowerCase().replace(/\./g, "");
      return name === wanted || name.endsWith(` ${wanted}`);
    });
  return hits.length === 1 ? hits[0] : null;
}

function Transport({
  events,
  cursor,
  total,
  playing,
  onPlayPause,
  onStep,
  onCursor,
  speed,
  speeds,
  onSpeed,
}) {
  return (
    <div className="bar">
      <button className="bar__wide" onClick={() => onStep(-1)}>
        ◀ Previous
      </button>
      <button
        className={`bar__wide bar__wide--play ${playing ? "is-playing" : ""}`}
        onClick={onPlayPause}
        aria-label={playing ? "Pause" : "Play"}
      >
        {playing ? "❚❚ Pause" : "▶ Play"}
      </button>
      <button className="bar__wide" onClick={() => onStep(1)}>
        Next ▶
      </button>

      <div className="bar__track">
        <input
          className="scrub"
          type="range"
          min={0}
          max={Math.max(total - 1, 0)}
          value={cursor}
          onChange={(e) => onCursor(Number(e.target.value))}
          aria-label="Timeline"
        />
        <div className="bar__read">
          {periodLabel(events.period[cursor])} {formatClock(events.clock[cursor])}
          <span className="bar__count">{cursor + 1} / {total}</span>
        </div>
      </div>

      <div className="bar__speed">
        <span className="bar__speedlabel">Speed</span>
        <div className="seg seg--speed" role="group" aria-label="Playback speed">
          {speeds.map((value) => (
            <button
              key={value}
              className={`seg__btn ${speed === value ? "seg__btn--on" : ""}`}
              onClick={() => onSpeed(value)}
            >
              {value}&times;
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function HowToRead() {
  return (
    <div className="howto">
      <p>
        <b>The shot marker is measured.</b> It sits at the coordinate the
        play-by-play feed recorded, in tenths of a foot from the hoop. A green
        circle is a make, an amber cross a miss.
      </p>
      <p>
        <b>The ten circles are a team sheet, not positions.</b> They show who
        was on the floor, in fixed slots that never move. The attacking five are
        drawn brightly and the defending five are muted, which says who had the
        ball and nothing about where anybody stood. Public data carries one
        coordinate per shot attempt and nothing else, so no player location,
        pass or route exists to draw.
      </p>
      <p>
        <b>Everything else is recorded:</b> clock, score, the event, the lineups
        and both win probabilities. The probability is out of fold, from a model
        that never saw this season.
      </p>
    </div>
  );
}

const FILTER_GROUPS = [
  { key: "team", label: "Team", options: [["all", "Both"], ["bos", "BOS"], ["opp", "OPP"]] },
  { key: "result", label: "Result", options: [["all", "All"], ["made", "Made"], ["missed", "Missed"]] },
  { key: "value", label: "Type", options: [["all", "All"], ["2", "2PT"], ["3", "3PT"]] },
  { key: "period", label: "Period", options: [["all", "Game"], ["current", "This one"]] },
];

/**
 * Filters over the shot history.
 *
 * These narrow what is DRAWN, never what is counted or modelled. Nothing here
 * touches a probability. They apply to "Recent shots" only, and say so when
 * that mode is off rather than silently doing nothing.
 */
function ShotFilters({ mode, value, onChange, fullHistory, onFullHistory }) {
  const dirty = Object.values(value).some((v) => v !== "all");
  return (
    <div className="drawer__body">
      <div className="drawer__filters">
        {mode !== "recent" && (
          <p className="note drawer__hint">
            These apply to <b>Recent shots</b>. The court is on{" "}
            <b>Current play</b>, which draws this event&apos;s shot only.
          </p>
        )}
        {FILTER_GROUPS.map((group) => (
          <div className="filters__group" key={group.key}>
            <span className="filters__label">{group.label}</span>
            {group.options.map(([option, label]) => (
              <button
                key={option}
                className={`chip ${value[group.key] === option ? "chip--on" : ""}`}
                onClick={() => onChange({ ...value, [group.key]: option })}
              >
                {label}
              </button>
            ))}
          </div>
        ))}
        <div className="filters__group">
          <span className="filters__label">Depth</span>
          <button
            className={`chip ${fullHistory ? "chip--on" : ""}`}
            onClick={() => onFullHistory(!fullHistory)}
            title="Every shot so far, rather than the most recent eight"
          >
            Whole game
          </button>
          <button
            className="chip"
            onClick={() =>
              onChange({ team: "all", result: "all", value: "all", period: "all" })
            }
            disabled={!dirty}
          >
            Clear
          </button>
        </div>
      </div>
    </div>
  );
}
