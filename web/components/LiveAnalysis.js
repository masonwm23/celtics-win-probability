"use client";

import { useMemo, useState } from "react";
import PlayCourt from "@/components/PlayCourt";
import OnCourtBox from "@/components/OnCourtBox";
import TeamLogo from "@/components/TeamLogo";
import WinProbabilityChart from "@/components/WinProbabilityChart";
import { formatClock, jerseyNumber, periodLabel } from "@/lib/format";
import { DISCLOSURE, displayedChange } from "@/lib/reconstruct";
import { describePlay, reboundKindsForGame } from "@/lib/playby";
import { finalHeadline, finalSentence, gameOutcome } from "@/lib/outcome";
import { scoreLine, scoreSides } from "@/lib/teams";
import { settledWp } from "@/lib/settled";
import { biggestSwings, changeSummary, scoringRun, swingSize } from "@/lib/moments";
import {
  LEGEND_BOX,
  MADE_CLASS,
  MADE_RADIUS,
  MISS_CLASS,
  legendMissLines,
} from "@/lib/marks";

/**
 * The live section: court on the left, probability and detail on the right.
 *
 * This replaces a stacked layout where the full-height research chart sat
 * above the court. At 100% zoom on a 1440-wide laptop that chart used most of
 * the viewport on its own and the reconstruction started below the fold, so
 * watching the probability move and watching the play were mutually exclusive
 * unless you zoomed out. The full chart is not gone; it is folded away
 * underneath for the research view.
 *
 * Everything here runs off ONE cursor. The court, the chart's marker, the
 * current-play strip and the ten running point totals are four readings of the
 * same index, so they cannot drift apart.
 */
export default function LiveAnalysis({
  events,
  cursor,
  total,
  players,
  meta,
  celticsLineup,
  opponentLineup,
  showBaseline,
  onBaseline,
  onCursor,
  playing,
  onPlayPause,
  onStep,
  speed,
  speeds,
  onSpeed,
  ballMs,
  onOpenDrawer,
}) {
  const [mode, setMode] = useState("current");
  const [howTo, setHowTo] = useState(false);

  const reboundKinds = useMemo(() => reboundKindsForGame(events), [events]);
  const outcome = gameOutcome(events, meta, cursor);
  const player = players[events.person_id[cursor]] || null;
  const railWp = settledWp(events, meta, cursor);

  return (
    <section className="panel live">
      <header className="live__head">
        <h2 className="panel__title">Live analysis</h2>
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
          {/* The same drawer, reachable without going back to the header. */}
          <button
            className="ghostbtn"
            onClick={() => onOpenDrawer("games")}
            aria-haspopup="dialog"
          >
            <span className="ghostbtn__i">▦</span> Games &amp; rosters
          </button>
        </div>
      </header>

      {howTo && <HowToRead />}

      <div className="live__grid">
        {/* Left, about 65%. */}
        <div className="live__court">
          <div className="live__courthead">
            <span className="live__label">Live play reconstruction</span>
          </div>
          {/* The tour's "court" step points HERE, not at the whole left column.
              The column also holds the swing strip, which is its own later
              step, and one ring around both says "this is the court" while
              circling something else. */}
          <div data-tour="court">
            <PlayCourt
              events={events}
              cursor={cursor}
              players={players}
              celticsLineup={celticsLineup}
              opponentLineup={opponentLineup}
              opponentAbbrev={meta.opponent}
              mode={mode}
              ballMs={ballMs}
            />
            <div className="clegend">
              <span><i className="lg lg--bos" /> BOS player</span>
              <span><i className="lg lg--opp" /> {meta.opponent} player</span>
              <span><MarkSwatch made /> Shot made</span>
              <span><MarkSwatch /> Shot missed</span>
              <span><i className="lg lg--lit" /> Involved in this play</span>
            </div>
          </div>
          <div data-tour="swings">
            <SwingStrip events={events} cursor={cursor} onCursor={onCursor} />
          </div>
          <Reading events={events} cursor={cursor} meta={meta} />
          <p className="disclosure">
            <span className="disclosure__mark">i</span>
            {DISCLOSURE}
          </p>
        </div>

        {/* Right, about 35%. */}
        <div className="live__rail">
          <div className="railcard" data-tour="chart">
            <div className="railcard__head">
              <span className="live__label">Win probability</span>
              <div className="railcard__legend">
                <span><i className="swatch swatch--line" /> Celtics-specific</span>
                <button
                  className={`microchip ${showBaseline ? "microchip--on" : ""}`}
                  onClick={() => onBaseline(!showBaseline)}
                >
                  Generic baseline
                </button>
                {/* A scoreboard readout, so it resolves with the game. */}
                <b
                  className={`railcard__now ${
                    railWp.isSettled ? "railcard__now--settled" : ""
                  }`}
                >
                  {(railWp.value * 100).toFixed(1)}%
                </b>
              </div>
            </div>
            <WinProbabilityChart
              events={events}
              meta={meta}
              cursor={cursor}
              onCursor={onCursor}
              showBaseline={showBaseline}
              periods={meta.periods}
              variant="compact"
            />
          </div>

          <CurrentPlayStrip
            events={events}
            cursor={cursor}
            players={players}
            player={player}
            meta={meta}
            reboundType={reboundKinds.get(cursor) ?? null}
            outcome={outcome}
          />

          <OnCourtBox
            events={events}
            cursor={cursor}
            players={players}
            meta={meta}
            celticsLineup={celticsLineup}
            opponentLineup={opponentLineup}
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
    </section>
  );
}

/**
 * The current play, in one horizontal strip.
 *
 * The previous card was a column of blocks and was the single tallest thing in
 * the right rail. Same information, laid out across instead of down.
 */
function CurrentPlayStrip({
  events,
  cursor,
  players,
  player,
  meta,
  reboundType,
  outcome,
}) {
  const description = events.description[cursor] || "";
  const change = displayedChange(events, cursor);
  const play = describePlay(
    {
      action_type: events.action_type[cursor],
      shot_value: events.shot_value[cursor],
      description,
    },
    { reboundType }
  );

  const assistName = assistLabel(description);

  // Home team on the left, the same rule the scoreboard uses. This card used
  // to print Boston first regardless, so in Boston's 318 away games it read
  // the game backwards against the scoreboard directly above it.
  const sides = scoreSides({
    meta,
    celticsScore: events.celtics_score[cursor],
    opponentScore: events.opponent_score[cursor],
  });

  return (
    <div className="railcard strip" data-tour="play">
      <div className="railcard__head">
        <span className="live__label">Current play</span>
        <span className="strip__count">
          {cursor + 1} / {events.wp.length}
        </span>
      </div>

      {outcome.isFinal && (
        <div className={`finalstate ${outcome.tie ? "finalstate--tie" : ""}`}>
          <div className="finalstate__head">{finalHeadline(outcome)}</div>
          <div className="finalstate__sub">{finalSentence(outcome)}</div>
          <div className="finalstate__meta">
            {outcome.periodEnd}
            {outcome.scoreCheck && !outcome.scoresAgree && (
              <span className="finalstate__warn">
                {" "}· the event score and the boxscore disagree
              </span>
            )}
          </div>
        </div>
      )}

      <div className="strip__row">
        <div className="strip__who">
          <Face player={player} />
          <div>
            <div className="strip__name">
              {player
                ? player.name
                : outcome.isFinal
                  ? "Game complete"
                  : "No player recorded"}
            </div>
            <div className="strip__what">
              {outcome.isFinal && !player
                ? outcome.tie
                  ? "recorded scores are tied"
                  : `${outcome.winnerName} defeat ${outcome.loserName}`
                : play.label}
              {play.detail && <span> — {play.detail}</span>}
            </div>
          </div>
        </div>

        <Cell label="Assist" value={assistName || "None"} dim={!assistName} />
        <ScoreCell
          sides={sides}
          clock={`${periodLabel(events.period[cursor])} ${formatClock(events.clock[cursor])}`}
        />
        <Cell
          label="Win prob change"
          value={
            change
              ? `${change.points > 0 ? "+" : change.points < 0 ? "−" : "±"}${Math.abs(change.points).toFixed(1)}`
              : "—"
          }
          tone={change?.direction}
          note="percentage points"
        />
      </div>

      {play.note && <p className="strip__unknown">{play.note}</p>}
      {description && <p className="strip__feed">{description}</p>}
    </div>
  );
}

/**
 * The score, with each team's own logo under its own number.
 *
 * The numbers alone were ambiguous: nothing on the card said which side was
 * which, and the order is not fixed because Boston are away in half their
 * games. The logos are the existing payload assets and fall back to the
 * tricode, which is always correct.
 */
function ScoreCell({ sides, clock }) {
  return (
    <div className="strip__cell strip__score" title={scoreLine(sides)}>
      {/* The clock sits with the score rather than in the far corner of the
          court, so "when" and "what" are read in one place. */}
      <div className="strip__clock">{clock}</div>
      <div className="strip__cellvalue">
        {sides.left.score}–{sides.right.score}
      </div>
      <div className="strip__teams">
        {[sides.left, sides.right].map((side) => (
          <span
            className={`strip__team ${side.isCeltics ? "is-bos" : ""}`}
            key={side.abbrev}
          >
            <TeamLogo
              src={side.logo}
              abbr={side.abbrev}
              size={21}
              className="strip__logo"
            />
            <i>{side.abbrev}</i>
          </span>
        ))}
      </div>
    </div>
  );
}

/**
 * The biggest probability moves in this game, as one row of chips.
 *
 * Clicking one seeks there. Every figure is the real out-of-fold change on a
 * real event, so this is a way into the game rather than a summary of it.
 */
function SwingStrip({ events, cursor, onCursor }) {
  const rows = useMemo(() => biggestSwings(events, { limit: 6 }), [events]);
  if (!rows.length) return null;
  return (
    <div className="swings">
      <span className="swings__label">Biggest swings</span>
      {rows.map((row) => (
        <button
          key={row.index}
          className={`swingchip swingchip--${row.towards} ${
            row.index === cursor ? "swingchip--on" : ""
          }`}
          onClick={() => onCursor(row.index)}
          title={row.description}
        >
          <i>{row.label}</i>
          {row.points > 0 ? "+" : "−"}
          {Math.abs(row.points).toFixed(1)}
        </button>
      ))}
    </div>
  );
}

/**
 * Three readings of the current moment, all measured.
 *
 * Swing size is the swing that ACTUALLY happened, ranked within this game. It
 * is deliberately not called leverage: leverage is the swing a moment could
 * produce, which needs an estimator built out of fold that does not exist yet.
 *
 * The explanation states what changed alongside the probability. The model
 * does not expose a reason and none is invented here.
 */
function Reading({ events, cursor, meta }) {
  const size = swingSize(events, cursor);
  const run = scoringRun(events, cursor, meta.opponent);
  const why = changeSummary(events, cursor, meta.opponent);

  return (
    <div className="reading">
      <div className="reading__row">
        <div className="reading__cell">
          <div className="strip__celllabel">Swing size</div>
          <div className="reading__value">
            {size ? size.label : "\u2014"}
            {size && (
              <i className="reading__sub">
                bigger than {Math.round(size.percentile * 100)}% of this
                game&apos;s {size.of} plays
              </i>
            )}
          </div>
        </div>

        <div className="reading__cell">
          <div className="strip__celllabel">Scoring run</div>
          <div className="reading__value">
            {run ? (
              <>
                <b className={run.team === "bos" ? "is-bos" : "is-opp"}>
                  {run.abbrev} {run.points}-0
                </b>
                <i className="reading__sub">since {run.since}</i>
              </>
            ) : (
              <span className="is-dim">None yet</span>
            )}
          </div>
        </div>

        {why && (
          <div className="reading__why">
            <div className="strip__celllabel">What changed</div>
            <p className="reading__text">{why.text}</p>
            <p className="reading__caveat">{why.caveat}</p>
          </div>
        )}
      </div>
    </div>
  );
}

function Cell({ label, value, tone, dim, note }) {
  return (
    <div className={`strip__cell ${tone ? `is-${tone}` : ""}`}>
      <div className="strip__celllabel">{label}</div>
      <div className={`strip__cellvalue ${dim ? "is-dim" : ""}`}>{value}</div>
      {note && <div className="strip__cellnote">{note}</div>}
    </div>
  );
}

function Face({ player }) {
  const [broken, setBroken] = useState(() => new Set());
  const url = [player?.headshot, player?.headshot_current].find(
    (candidate) => candidate && !broken.has(candidate)
  );
  if (!player || !url) {
    return <span className="stripface stripface--fallback" aria-hidden="true" />;
  }
  return (
    <img
      className="stripface"
      src={url}
      alt=""
      loading="lazy"
      onError={() => setBroken((prev) => new Set(prev).add(url))}
    />
  );
}

/** The court's own marker, shrunk. Same geometry, same class names. */
function MarkSwatch({ made = false }) {
  return (
    <svg className="lgmark" viewBox={`0 0 ${LEGEND_BOX} ${LEGEND_BOX}`} aria-hidden="true">
      {made ? (
        <circle cx={LEGEND_BOX / 2} cy={LEGEND_BOX / 2} r={MADE_RADIUS} className={MADE_CLASS} />
      ) : (
        <g className={MISS_CLASS}>
          {legendMissLines().map((line, i) => <line key={i} {...line} />)}
        </g>
      )}
    </svg>
  );
}

function assistLabel(description) {
  const match = String(description || "").match(/\(([^()]+?)\s+\d+\s+AST\)/);
  return match ? match[1].trim() : null;
}

function Transport({
  events, cursor, total, playing, onPlayPause, onStep, onCursor,
  speed, speeds, onSpeed,
}) {
  return (
    <div className="bar" data-tour="transport">
      <button className="bar__wide" onClick={() => onStep(-1)}>◀ Previous</button>
      <button
        className={`bar__wide bar__wide--play ${playing ? "is-playing" : ""}`}
        onClick={onPlayPause}
        aria-label={playing ? "Pause" : "Play"}
      >
        {playing ? "❚❚ Pause" : "▶ Play"}
      </button>
      <button className="bar__wide" onClick={() => onStep(1)}>Next ▶</button>

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

/**
 * The "How to read" panel.
 *
 * The wording here is the author's own, supplied Aug 4 and used verbatim apart
 * from spelling and agreement fixes. Do not "improve" it: the voice is
 * deliberate, and the four items map one-to-one onto the four he numbered.
 *
 * Each point still carries the actual mark it describes, drawn from the same
 * geometry the court uses (MarkSwatch, the .lg swatches), so the reader matches
 * a symbol to a symbol instead of holding a description in their head while
 * they go looking for it.
 */
function HowToRead() {
  return (
    <div className="howto">
      <p className="howto__lede">
        The information displayed is from an actual game played back from the
        official record of game events. The large green percentage shown is the
        Celtics&apos; chance of winning the game for the exact moment during the
        game derived from the score, the time remaining and what players are on
        the floor at that same moment.
      </p>

      <ul className="howto__list">
        <li>
          <span className="howto__cue">
            <MarkSwatch made />
            <MarkSwatch />
          </span>
          <span>
            These symbols on the court represent the location where the shot was
            taken from. The green circle means the basket went in, the amber
            cross indicates the shot missed. The indicated shooting position is
            the actual spot the shot occurred from based on league records and
            is not an estimate or prediction.
          </span>
        </li>

        <li>
          <span className="howto__cue">
            <i className="lg lg--bos" />
            <i className="lg lg--bos howto__dot--muted" />
          </span>
          <span>
            The ten circles show the players currently in the game and playing
            and are not a representation of their exact position on the court.
            Consider this visual representation as a team sheet as the actual
            position slots do not move. The five players highlighted represent
            the team on offense, while the five players faded are the team on
            defense.
          </span>
        </li>

        <li>
          <span className="howto__cue">
            <i className="swatch swatch--line howto__line" />
          </span>
          <span>
            No data input into this model has been artificially created. All
            data used, such as the game clock time, the score, the basketball
            play at any time during the game and both team lineups have been
            used directly, without change, from the actual NBA game record of
            the specified game represented.
          </span>
        </li>
      </ul>
    </div>
  );
}
