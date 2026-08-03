"use client";

import TeamLogo from "./TeamLogo";
import { formatClock, periodLabel, percent } from "@/lib/format";
import { finalHeadline, gameOutcome } from "@/lib/outcome";
import { settledWp } from "@/lib/settled";

/**
 * The scoreboard strip: who is playing, the score at the cursor, the clock, and
 * the live win probability.
 *
 * Layout is decided by SIDE, not by team. An earlier version keyed the markup
 * off "is this Boston", which meant that in Boston's 318 away games the blocks
 * rendered mirrored: scores drifted to the outer edges and logos to the middle,
 * the opposite of every scoreboard ever built. Logos belong on the outside and
 * scores next to the clock.
 *
 * The probability here is OUT OF FOLD, which the badge states. It is the most
 * prominent number on the page, so it had better be the honest one.
 *
 * With one exception, and the badge changes to declare it. At the final event
 * the clock reads 0:00 and the result is already in the score columns, so this
 * shows 100% or 0% rather than the model's residual 99.7% or 0.2%. A headline
 * reading "FINAL, TOR win 113-101" directly above "0.2% chance of winning" was
 * two claims on screen at once with only one of them true. See lib/settled for
 * why this is display-only and what it refuses to do.
 */
function Team({ abbr, logo, score, isHome, side, leading }) {
  const parts = [
    <TeamLogo key="logo" src={logo} abbr={abbr} />,
    <div key="name">
      <div className="ribbon__abbr">{abbr}</div>
      <div className="ribbon__meta">{isHome ? "Home" : "Away"}</div>
    </div>,
    <div
      key="score"
      className={`ribbon__score ${leading ? "ribbon__score--lead" : ""}`}
    >
      {score}
    </div>,
  ];
  return (
    <div className={`ribbon__team ${side === "right" ? "ribbon__team--away" : ""}`}>
      {side === "right" ? [...parts].reverse() : parts}
    </div>
  );
}

export default function MatchupRibbon({ meta, events, cursor }) {
  const bos = events.celtics_score[cursor];
  const opp = events.opponent_score[cursor];
  const celticsHome = meta.celtics_is_home;

  const celtics = {
    abbr: "BOS",
    logo: meta.celtics_logo,
    score: bos,
    isHome: celticsHome,
    leading: bos > opp,
  };
  const opponent = {
    abbr: meta.opponent,
    logo: meta.opponent_logo,
    score: opp,
    isHome: !celticsHome,
    leading: false,
  };

  // The home team sits on the left, as a broadcast graphic would show it.
  const left = celticsHome ? celtics : opponent;
  const right = celticsHome ? opponent : celtics;

  const outcome = gameOutcome(events, meta, cursor);
  const wp = settledWp(events, meta, cursor);

  return (
    <div className="panel ribbon">
      <Team {...left} side="left" />
      <div className="ribbon__centre">
        <div className="ribbon__clock">{formatClock(events.clock[cursor])}</div>
        {/* At the last event the status is the RESULT, not the period. Anything
            else reads as though the game might continue. */}
        <div className={`ribbon__period ${outcome.isFinal ? "is-final" : ""}`}>
          {outcome.isFinal
            ? finalHeadline(outcome)
            : `${periodLabel(events.period[cursor])}${
                events.is_clutch[cursor] ? " · CLUTCH" : ""
              }`}
        </div>
        <div className={`ribbon__wp ${wp.isSettled ? "ribbon__wp--settled" : ""}`}>
          {percent(wp.value)}
        </div>
        <div className="ribbon__wplabel">
          Boston win probability
          {/* The badge names the SOURCE of the number above it. Once that
              number comes from the final score rather than from the model,
              saying "out of fold" would be claiming a provenance it no longer
              has. */}
          {wp.isSettled ? (
            <span className="ribbon__oof ribbon__oof--settled">
              <span className="dot" /> final result · game over
            </span>
          ) : (
            <span className="ribbon__oof">
              <span className="dot" /> out of fold
            </span>
          )}
        </div>
      </div>
      <Team {...right} side="right" />
    </div>
  );
}
