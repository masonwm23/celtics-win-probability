"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import GamesDrawer, { GamesDrawerButton } from "@/components/GamesDrawer";
import LiveAnalysis from "@/components/LiveAnalysis";
import ScoringRun from "@/components/ScoringRun";
import LineupImpact from "@/components/LineupImpact";
import MatchupRibbon from "@/components/MatchupRibbon";
import ModelQuality from "@/components/ModelQuality";
import OpponentContext from "@/components/OpponentContext";
import PlayerRow from "@/components/PlayerRow";
import Tour, {
  TourNudge,
  advanceNudge,
  nudgeStage,
  silenceNudge,
} from "@/components/Tour";
import Welcome, { rememberWelcome } from "@/components/Welcome";
import WhatIf from "@/components/WhatIf";
import WinProbabilityChart from "@/components/WinProbabilityChart";
import { fetchGame, fetchGames, API_BASE } from "@/lib/api";
import { prettyDate } from "@/lib/format";

// Replay speed. Events are not evenly spaced in game time, so this steps
// through events rather than seconds: a dense scoring run plays out at the same
// rate as a quiet stretch, which is what makes the trace readable.
//
// The base tick was 90ms, which is a scrub rather than a replay: the
// description and the probability change were gone before either could be
// read. 1x is now 700ms, about the time it takes to read one line of
// play-by-play, so 0.5x is 1.4 seconds per event and 2x is 350ms. Three
// settings, because a longer list of speeds is a worse control than a sensible
// default.
const BASE_TICK_MS = 700;
const SPEEDS = [0.5, 1, 2];

export default function Dashboard() {
  const [index, setIndex] = useState(null);
  const [gameId, setGameId] = useState(null);
  const [game, setGame] = useState(null);
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [showBaseline, setShowBaseline] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [selected, setSelected] = useState(null);
  // The drawer. Opening it pauses playback: changing the game underneath a
  // running clock would leave the cursor pointing into another game's events
  // for a tick, and pausing is cheaper than defending against that.
  // The orientation panel. Starts closed and is switched on after mount by the
  // effect below: reading localStorage during render would make the server and
  // the client disagree about the first paint, which React reports as a
  // hydration error.
  const [welcomeOpen, setWelcomeOpen] = useState(false);
  const [tourOpen, setTourOpen] = useState(false);
  // null = say nothing, "centre" = the large card, "side" = the small one.
  const [nudge, setNudge] = useState(null);
  // Folds the tour has forced open. A fold stays controllable only while its
  // name is in here, so anything the tour did not touch keeps its own state and
  // the reader's manual opens and closes are never overridden.
  const [forcedFolds, setForcedFolds] = useState({});
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerTab, setDrawerTab] = useState("games");
  const openDrawer = useCallback((tab = "games") => {
    setPlaying(false);
    setDrawerTab(tab);
    setDrawerOpen(true);
  }, []);
  // Two kinds of failure, deliberately separated. If the index cannot load the
  // dashboard has nothing to show and the fatal screen is right. If ONE game
  // fails, replacing the whole page would throw away a working view over a
  // single bad request, so that surfaces as an inline banner and the game
  // already on screen stays put.
  const [error, setError] = useState(null);
  const [gameError, setGameError] = useState(null);
  const timer = useRef(null);

  // First visit only. A browser that has dismissed it never sees it again
  // unless the header button asks for it.
  // The tour offer owns the first-run slot. The welcome panel used to open
  // itself here too, which meant a first-time visitor met two centred panels in
  // a row saying overlapping things. It is still one click away under "What is
  // this?", where somebody who wants the longer explanation will look for it.
  useEffect(() => {
    const stage = nudgeStage();
    if (stage) {
      setNudge(stage);
      advanceNudge();
    }
  }, []);

  // Demo overrides, read from the address bar. The first-run experience is by
  // design something a browser sees twice and then never again, which is right
  // for a visitor and useless for the person demonstrating this to a room: by
  // the third rehearsal it has gone quiet and there is no way back to it short
  // of clearing storage from the console.
  //
  //   ?intro   the welcome panel
  //   ?new     the large first-time tour card
  //   ?tour    starts the tour immediately
  //
  // Nothing is written to storage by any of these, so using one does not spend
  // a stage or change what a real visitor would see next.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.has("intro")) setWelcomeOpen(true);
    else if (params.has("new")) setNudge("centre");
    else if (params.has("tour")) setTourOpen(true);
  }, []);

  const dismissNudge = useCallback(() => {
    silenceNudge();
    setNudge(null);
  }, []);

  const closeWelcome = useCallback(() => {
    rememberWelcome();
    setWelcomeOpen(false);
  }, []);

  const startTour = useCallback(() => {
    rememberWelcome();
    silenceNudge();
    setNudge(null);
    setWelcomeOpen(false);
    setForcedFolds({});
    setTourOpen(true);
  }, []);

  // The tour opens a fold it is about to point at. It never closes one, so a
  // reader who opened something themselves does not have it shut under them.
  const openFoldForTour = useCallback((name) => {
    setForcedFolds((current) =>
      current[name] ? current : { ...current, [name]: true });
  }, []);

  // Leaves every fold exactly as the tour left it: the reader has just been
  // shown those panels and collapsing them on exit would undo the point.
  const closeTour = useCallback(() => setTourOpen(false), []);

  const foldProps = useCallback((name) => (
    forcedFolds[name]
      ? { open: true, onOpenChange: (next) => setForcedFolds((c) => ({ ...c, [name]: next })) }
      : {}
  ), [forcedFolds]);

  // The game index, once.
  useEffect(() => {
    fetchGames()
      .then((data) => {
        setIndex(data);
        // Open on the biggest comeback Boston won, because a flat game teaches
        // a first-time viewer nothing about what the model does.
        const wins = data.games.filter((g) => g.celtics_won);
        wins.sort((a, b) => a.lowest_wp - b.lowest_wp);
        setGameId(wins[0]?.game_id || data.games[0].game_id);
      })
      .catch((err) => setError(err.message));
  }, []);

  // The selected game.
  useEffect(() => {
    if (!gameId) return;
    let cancelled = false;
    setPlaying(false);
    setGameError(null);
    fetchGame(gameId)
      .then((data) => {
        if (cancelled) return;
        setGame(data);
        setCursor(0);
        setSelected(null);
      })
      .catch((err) => {
        if (cancelled) return;
        // Keep whatever game is already loaded rather than blanking the page.
        setGameError(`${gameId}: ${err.message}`);
      });
    return () => {
      cancelled = true;
    };
  }, [gameId]);

  const total = game ? game.events.wp.length : 0;

  // Playback.
  useEffect(() => {
    if (!playing || !game) return undefined;
    timer.current = setInterval(() => {
      setCursor((c) => {
        if (c >= total - 1) {
          setPlaying(false);
          return c;
        }
        return c + 1;
      });
    }, BASE_TICK_MS / speed);
    return () => clearInterval(timer.current);
  }, [playing, game, total, speed]);

  const step = useCallback(
    (delta) => {
      setPlaying(false);
      setCursor((c) => Math.min(Math.max(c + delta, 0), Math.max(total - 1, 0)));
    },
    [total]
  );

  // Keyboard: space to play, arrows to step, shift for a bigger jump.
  useEffect(() => {
    function onKey(e) {
      if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
      if (drawerOpen || welcomeOpen || tourOpen) return;
      if (e.code === "Space") {
        e.preventDefault();
        setPlaying((p) => !p);
      } else if (e.code === "ArrowRight") {
        step(e.shiftKey ? 25 : 1);
      } else if (e.code === "ArrowLeft") {
        step(e.shiftKey ? -25 : -1);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [step, drawerOpen, welcomeOpen, tourOpen]);

  const lineups = useMemo(() => {
    if (!game) return { celtics: [], opponent: [] };
    const table = game.lineup_table;
    return {
      celtics: table[game.events.celtics_lineup[cursor]] || [],
      opponent: table[game.events.opponent_lineup[cursor]] || [],
    };
  }, [game, cursor]);

  if (error) return <FatalError message={error} />;
  if (!index || !game) return <Loading />;

  const meta = game.meta;
  const events = game.events;

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand__mark">BOS</div>
          <div>
            <h1 className="brand__title">Celtics Live Win Probability</h1>
            <p className="brand__sub">
              MSBA Directed Research · Brandeis University · 636 games,
              2016-17 to 2023-24
            </p>
          </div>
        </div>
        <div className="row">
          <span className="badge badge--oof">
            <span className="dot" />
            every probability out of fold
          </span>
          <span className="badge">{prettyDate(meta.date)}</span>
          <span className="badge">{meta.season}</span>
          <button
            className="ghostbtn"
            data-tour="whatisthis"
            onClick={() => setWelcomeOpen(true)}
            aria-haspopup="dialog"
          >
            <span className="ghostbtn__i">i</span> What is this?
          </button>
          <button className="ghostbtn ghostbtn--go" onClick={startTour}>
            <span className="ghostbtn__i">&rarr;</span> Take the tour
          </button>
          <GamesDrawerButton
            current={index.games.find((g) => g.game_id === gameId)}
            onOpen={() => openDrawer("games")}
          />
        </div>
      </header>

      {gameError && (
        <div
          className="panel"
          style={{ padding: "12px 16px", marginBottom: 14 }}
        >
          <div className="row" style={{ justifyContent: "space-between" }}>
            <span className="badge badge--warn">could not load that game</span>
            <button className="chip" onClick={() => setGameError(null)}>
              Dismiss
            </button>
          </div>
          <p className="note" style={{ margin: "8px 0 0" }}>
            {gameError}. Still showing{" "}
            <strong>
              {meta.matchup}, {prettyDate(meta.date)}
            </strong>
            .
          </p>
        </div>
      )}

      <Welcome
        open={welcomeOpen}
        onClose={closeWelcome}
        onStartTour={startTour}
        meta={meta}
      />
      <Tour open={tourOpen} onClose={closeTour} onOpenFold={openFoldForTour} />
      <TourNudge
        open={Boolean(nudge) && !welcomeOpen && !tourOpen}
        variant={nudge || "side"}
        onStart={startTour}
        onDismiss={dismissNudge}
        onSoftClose={() => setNudge(null)}
      />

      <div data-tour="scoreboard">
        <MatchupRibbon meta={meta} events={events} cursor={cursor} />
      </div>

      {/* The live section owns the fold. Everything a viewer needs while a
          game is playing sits here: court, probability, current play, the ten
          running totals and the controls. Everything else is folded below. */}
      <LiveAnalysis
        events={events}
        cursor={cursor}
        total={total}
        players={game.players}
        meta={meta}
        celticsLineup={lineups.celtics}
        opponentLineup={lineups.opponent}
        showBaseline={showBaseline}
        onBaseline={setShowBaseline}
        onCursor={(i) => {
          setPlaying(false);
          setCursor(i);
        }}
        playing={playing}
        onPlayPause={() => setPlaying((p) => !p)}
        onStep={step}
        speed={speed}
        speeds={SPEEDS}
        onSpeed={setSpeed}
        ballMs={Math.min(520, (BASE_TICK_MS / speed) * 0.65)}
        onOpenDrawer={openDrawer}
      />

      <div className="folds">
        {/* The research view. Not deleted, just no longer occupying the fold. */}
        <Fold tour="fullchart" title="Detailed probability chart" hint="full research view">
          <div className="panel__body">
            <WinProbabilityChart
              events={events}
              meta={meta}
              cursor={cursor}
              onCursor={(i) => {
                setPlaying(false);
                setCursor(i);
              }}
              showBaseline={showBaseline}
              periods={meta.periods}
            />
            <p className="note" style={{ marginTop: 12, marginBottom: 0 }}>
              <strong>Solid line:</strong> gradient boosting on thirteen
              validated game-state features.{" "}
              <strong>Dashed line:</strong> a logistic regression on score
              margin and time remaining alone. Across 636 games the two are
              statistically indistinguishable, bootstrap difference +0.0011
              with a 95% interval of [−0.0029, +0.0048]. That null result is
              the paper&apos;s headline, and it is visible here rather than
              merely asserted.
            </p>
          </div>
        </Fold>

        <Fold tour="scoring" title="Points as the game runs" hint="every player, both teams">
          <div className="panel__body">
            <ScoringRun
              events={events}
              cursor={cursor}
              players={game.players}
              meta={meta}
              celticsLineup={lineups.celtics}
              opponentLineup={lineups.opponent}
            />
            <p className="note" style={{ marginTop: 12, marginBottom: 0 }}>
              Totals are summed from made field goals and made free throws. The
              feed also writes a running total into each description, and that
              counter disagrees with the boxscore in 4 of the 636 games, so it
              is not used.
            </p>
          </div>
        </Fold>

        {game.opponent_context && (
          <Fold
            tour="opponent"
            title="Who they were playing"
            hint={`${meta.opponent} form before this game`}
          >
            <OpponentContext
              context={game.opponent_context}
              opponent={meta.opponent}
              opponentName={meta.opponent_name}
            />
          </Fold>
        )}

        {/* Placed ABOVE the lineup fold on purpose. The lineup note argues
            from a Brier score, so the reader should have met one by the time
            they get there. */}
        <Fold tour="model" {...foldProps("model")} title="How good is this model?" hint="and what a Brier score is">
          <ModelQuality />
        </Fold>

        <Fold tour="lineup" title="Lineup on the floor" hint="descriptive only, not a model input">
          <LineupImpact
            players={game.players}
            lineup={lineups.celtics}
            opponentLineup={lineups.opponent}
            wp={events.wp[cursor]}
          />
        </Fold>

        <Fold tour="whatif" {...foldProps("whatif")} title="What if the margin were different" hint="in-sample">
          <WhatIf
            gameId={meta.game_id}
            eventIndex={events.event_index[cursor]}
            actualMargin={events.margin[cursor]}
            actualWp={events.wp[cursor]}
          />
        </Fold>

        {selected && game.players[selected] && (
          <Fold title="Selected player" hint={game.players[selected].name}>
            <div className="panel__body">
              <PlayerRow player={game.players[selected]} active metric="value" />
              <button
                className="chip"
                style={{ marginTop: 10 }}
                onClick={() => setSelected(null)}
              >
                Clear
              </button>
            </div>
          </Fold>
        )}
      </div>

      <GamesDrawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        tab={drawerTab}
        onTab={setDrawerTab}
        games={index.games}
        seasons={index.seasons}
        current={index.games.find((g) => g.game_id === gameId)}
        onPick={setGameId}
        players={game.players}
        meta={meta}
      />
    </div>
  );
}

/**
 * What this play did to the win probability.
 *
 * Every number here is real: both probabilities come from the out-of-fold
 * series already on screen in the chart. This is the single most useful thing
 * the court view can say, because it connects a play to the research question
 * rather than just illustrating it.
 */
/**
 * A section that starts closed.
 *
 * The child is UNMOUNTED while closed rather than hidden, so the what-if panel
 * does not fire its request until somebody actually opens it. That is the
 * opposite of the sidebar tabs, where staying mounted preserves a search box;
 * here there is no state worth preserving and there is a network call worth
 * deferring.
 */
function Fold({ title, hint, children, tour, open: openProp, onOpenChange }) {
  const [openState, setOpen] = useState(false);
  // A tour step that points at a fold has to be able to open it, so the
  // open flag becomes controllable when a caller supplies one.
  const open = openProp === undefined ? openState : openProp;
  const setBoth = (next) => {
    setOpen(next);
    onOpenChange?.(next);
  };
  return (
    <div className="panel" data-tour={tour}>
      <button
        className="fold__toggle"
        onClick={() => setBoth(!open)}
        aria-expanded={open}
      >
        {title}
        {hint && <span className="fold__hint">{hint}</span>}
        <span className={`fold__caret ${open ? "fold__caret--on" : ""}`}>⌄</span>
      </button>
      {open && children}
    </div>
  );
}

function Loading() {
  return (
    <div className="shell">
      <div className="state">
        <div className="spinner" />
        <p>Loading game data</p>
      </div>
    </div>
  );
}

function FatalError({ message }) {
  return (
    <div className="shell">
      <div className="state">
        <span className="badge badge--warn">cannot reach the API</span>
        <p style={{ maxWidth: 460 }}>{message}</p>
        <p className="note" style={{ maxWidth: 460 }}>
          The dashboard reads from the Python API at <code>{API_BASE}</code>.
          Start it in a Terminal, from the project root:
        </p>
        <div className="code">python scripts/21_serve_api.py</div>
        <p className="note" style={{ maxWidth: 460 }}>
          If that reports missing serving data, run{" "}
          <code>scripts/20_build_serving.py</code> first.
        </p>
      </div>
    </div>
  );
}
