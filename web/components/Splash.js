"use client";

import { useState } from "react";

/**
 * The boot screen.
 *
 * The percentage is not decoration. It is driven by `progress`, which the
 * dashboard computes as the smaller of two things: how much of the data has
 * actually downloaded, and how far a four-second floor has run. So the bar can
 * be held back by a slow connection but it can never claim a file arrived
 * before it did, and on a fast one it still plays out long enough to be read.
 *
 * Everything stated on this screen is a real number from the project: 800
 * games, ten seasons, 387,320 events, thirteen model features. A splash that
 * rounds its own project up is a bad first impression for a research
 * dashboard, so none of these are approximate.
 *
 * `leaving` is set for the fade-out. The component stays mounted through it so
 * the dashboard does not appear underneath a half-faded overlay.
 */
export default function Splash({ progress = 0, label = "Loading", leaving = false }) {
  const pct = Math.round(Math.min(Math.max(progress, 0), 1) * 100);
  return (
    <div
      className={`splash ${leaving ? "splash--leaving" : ""}`}
      role="status"
      aria-live="polite"
      aria-busy={!leaving}
    >
      <div className="splash__inner">
        <SplashMark />

        <h1 className="splash__title">
          Celtics <span>Live</span> Win Probability
        </h1>
        <p className="splash__tagline">
          Real-time analytics. Every possession matters.
        </p>

        <div className="splash__rule">
          <span />
          <i />
          <span />
        </div>

        <p className="splash__by">A project by</p>
        <p className="splash__name">Mason Marathias</p>
        <p className="splash__affil">
          MSBA Directed Research <b>·</b> Brandeis University
        </p>

        <div className="splash__bar">
          <div
            className="splash__fill"
            style={{ width: `${pct}%` }}
            role="progressbar"
            aria-valuenow={pct}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label="Loading the dashboard"
          />
        </div>
        <p className="splash__status">{label}...</p>
        <p className="splash__pct">{pct}%</p>

        <ul className="splash__features">
          {FEATURES.map((f) => (
            <li key={f.title}>
              <span className="splash__icon">{f.icon}</span>
              <b>{f.title}</b>
              <span>{f.body}</span>
            </li>
          ))}
        </ul>
      </div>

      <footer className="splash__foot">
        <span>{IconCap} MSBA Directed Research</span>
        <i />
        <span>{IconShield} Brandeis University</span>
        <i />
        <span>{IconBall} 800 games, 2016-17 to 2025-26</span>
      </footer>
    </div>
  );
}

/**
 * The badge. The Celtics roundel comes from the same NBA CDN the scoreboard
 * uses, and fails the same way: to the tricode, never to a broken-image icon.
 */
function SplashMark() {
  const [failed, setFailed] = useState(false);
  return (
    <div className="splash__mark" aria-hidden="true">
      <span className="splash__ring" />
      <span className="splash__ticks" />
      {failed ? (
        <span className="splash__fallback">BOS</span>
      ) : (
        <img
          className="splash__logo"
          src="https://cdn.nba.com/logos/nba/1610612738/primary/L/logo.svg"
          alt=""
          onError={() => setFailed(true)}
        />
      )}
    </div>
  );
}

/* ------------------------------------------------------------------- icons */

const stroke = {
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.6,
  strokeLinecap: "round",
  strokeLinejoin: "round",
};

const IconTrend = (
  <svg viewBox="0 0 24 24" width="26" height="26" {...stroke}>
    <path d="M3 20h18" />
    <path d="M6 16l4-5 3 3 5-7" />
    <path d="M14 7h4v4" />
  </svg>
);

const IconCourt = (
  <svg viewBox="0 0 24 24" width="26" height="26" {...stroke}>
    <rect x="2.5" y="5.5" width="19" height="13" rx="1.5" />
    <path d="M12 5.5v13" />
    <circle cx="12" cy="12" r="2.6" />
    <path d="M2.5 9.2h3v5.6h-3M21.5 9.2h-3v5.6h3" />
  </svg>
);

const IconPulse = (
  <svg viewBox="0 0 24 24" width="26" height="26" {...stroke}>
    <path d="M2 13h3.5l2-6 3 12 2.5-8 2 4H22" />
  </svg>
);

const IconLineup = (
  <svg viewBox="0 0 24 24" width="26" height="26" {...stroke}>
    <circle cx="8.5" cy="8" r="2.8" />
    <circle cx="16.5" cy="9" r="2.2" />
    <path d="M3 19c0-3 2.5-5 5.5-5s5.5 2 5.5 5" />
    <path d="M15 14.4c2.6.2 4.5 2.1 4.5 4.6" />
  </svg>
);

const IconDoc = (
  <svg viewBox="0 0 24 24" width="26" height="26" {...stroke}>
    <path d="M6 3h8l4 4v14H6z" />
    <path d="M14 3v4h4" />
    <path d="M9 12h6M9 15.5h6M9 9h2" />
  </svg>
);

const IconCap = (
  <svg viewBox="0 0 24 24" width="13" height="13" {...stroke}>
    <path d="M12 5l9 4-9 4-9-4 9-4z" />
    <path d="M6 11v4c0 1.4 2.7 2.5 6 2.5s6-1.1 6-2.5v-4" />
  </svg>
);

const IconShield = (
  <svg viewBox="0 0 24 24" width="13" height="13" {...stroke}>
    <path d="M12 3l7 3v5.5c0 4.2-2.9 7.6-7 9-4.1-1.4-7-4.8-7-9V6l7-3z" />
  </svg>
);

const IconBall = (
  <svg viewBox="0 0 24 24" width="13" height="13" {...stroke}>
    <circle cx="12" cy="12" r="8.5" />
    <path d="M12 3.5v17M3.5 12h17" />
    <path d="M6 6c3.5 2.2 5.5 5.6 5.5 10.5M18 6c-3.5 2.2-5.5 5.6-5.5 10.5" />
  </svg>
);

/**
 * Five claims, each one true of what loads behind this screen.
 *
 * "Thirteen features" is the model's real input count and "387,320 game
 * states" is the real size of the corpus, not a rounded boast. The lineup line
 * is worded as a view rather than a driver on purpose: lineup strength is
 * descriptive in this dashboard, not a model input, and the panel it refers to
 * says so too.
 */
const FEATURES = [
  {
    icon: IconTrend,
    title: "Real-time analysis",
    body: "Win probability after every play",
  },
  {
    icon: IconCourt,
    title: "Advanced modeling",
    body: "13 features, 387,320 game states",
  },
  {
    icon: IconPulse,
    title: "Interactive insights",
    body: "Replay momentum swings and runs",
  },
  {
    icon: IconLineup,
    title: "Lineup impact",
    body: "See who was on the floor for it",
  },
  {
    icon: IconDoc,
    title: "Research driven",
    body: "Every probability is out of fold",
  },
];
