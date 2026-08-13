"use client";

import { useEffect, useRef } from "react";

/**
 * What this is, for somebody who has just opened it.
 *
 * The dashboard used to drop a first-time viewer straight onto a court diagram
 * with ten circles, a percentage and a scrubbing timeline, with nothing saying
 * what any of it was for. "How to read" answers what the marks MEAN, which is a
 * legend and a different job from orientation. This answers what the thing IS,
 * what you are looking at, and what to do first.
 *
 * The honest headline is in here rather than buried three folds down. The
 * research answer is a null result, and a viewer who spends ten minutes
 * exploring before discovering that has been misled by omission.
 *
 * SHOWN ONCE, REOPENABLE ON PURPOSE. Remembering the dismissal keeps it out of
 * the way day to day, but on a laptop that has already dismissed it the panel
 * would be invisible at exactly the moment it matters most, which is a
 * committee or a stranger opening it cold. So the header carries a permanent
 * "Help" button and this is reachable forever.
 *
 * localStorage is read defensively. Safari in private mode throws on access
 * rather than returning null, and a dashboard that crashes on load because it
 * could not remember a dismissal would be a poor trade.
 */

export const WELCOME_STORAGE_KEY = "celtics-wp-welcome-seen";

/** Has this browser seen it? False on any error, so the panel still shows. */
export function hasSeenWelcome() {
  try {
    return window.localStorage.getItem(WELCOME_STORAGE_KEY) === "1";
  } catch {
    return false;
  }
}

/** Remember the dismissal. Silently gives up if storage is unavailable. */
export function rememberWelcome() {
  try {
    window.localStorage.setItem(WELCOME_STORAGE_KEY, "1");
  } catch {
    /* private browsing, or storage disabled. Not worth breaking the page. */
  }
}

export default function Welcome({ open, onClose, onStartTour, meta }) {
  const closeRef = useRef(null);
  const panelRef = useRef(null);

  // Escape closes, and focus moves into the panel so a keyboard user is not
  // left tabbing around a dashboard they cannot see.
  useEffect(() => {
    if (!open) return undefined;
    closeRef.current?.focus();

    function onKey(e) {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
        return;
      }
      // A crude focus trap: two focusable elements, so wrap between them.
      if (e.key === "Tab") {
        const nodes = panelRef.current?.querySelectorAll(
          "button, [href], input, select, textarea, [tabindex]:not([tabindex='-1'])"
        );
        if (!nodes || nodes.length === 0) return;
        const first = nodes[0];
        const last = nodes[nodes.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    }

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  const opponent = meta?.opponent || "the opponent";

  return (
    <div className="welcome" role="dialog" aria-modal="true" aria-labelledby="welcome-title">
      <div className="welcome__scrim" onClick={onClose} />
      <div className="welcome__panel" ref={panelRef}>
        <div className="welcome__tag">MSBA Directed Research · Brandeis University</div>
        <h2 className="welcome__title" id="welcome-title">
          Celtics Live Win Probability
        </h2>

        <p className="welcome__lede">
          This replays real Boston Celtics games, one play at a time, and shows
          what a forecasting model thought their chances were at every moment.
          Ten seasons, 800 games, all of it from the league&apos;s official
          record of what happened.
        </p>

        <div className="welcome__block">
          <div className="welcome__h">What you are looking at</div>
          <p>
            A game is already loaded &mdash; Boston against {opponent}. The
            court on the left shows the play happening right now. The
            probability chart on the right is the Celtics&apos; chance of
            winning, moving as the game goes. Press <kbd>space</kbd> to play it,
            or drag anywhere on the chart to jump to a moment.
          </p>
        </div>

        <div className="welcome__block">
          <div className="welcome__h">Three things worth trying</div>
          <ol className="welcome__list">
            <li>
              Hit play and watch the probability move as the game turns.
            </li>
            <li>
              Open <b>Games &amp; rosters</b> to pick a different night &mdash;
              the list is ranked by the biggest comebacks Boston won.
            </li>
            <li>
              Open <b>What if the margin were different</b> and drag the slider
              to ask the model what it would have said about a game that never
              happened.
            </li>
          </ol>
        </div>

        <div className="welcome__block welcome__block--finding">
          <div className="welcome__h">The honest headline</div>
          <p>
            The question behind this project was whether a model built
            specifically around the Celtics forecasts better than a simple one
            that only watches the score and the clock.{" "}
            <b>It does not.</b> Across 800 games the two are too close to call.
            That is the finding, and it is stated here rather than buried &mdash;
            the <b>How good is this model?</b> panel has the numbers.
          </p>
          <p style={{ marginBottom: 0 }}>
            Every probability on the timeline is <b>out of fold</b>: predicted by
            a version of the model that never saw that season. It is forecasting
            these games, not remembering them.
          </p>
        </div>

        <div className="welcome__actions">
          {onStartTour && (
            <button className="welcome__go" onClick={onStartTour} ref={closeRef}>
              Take the tour
            </button>
          )}
          <button
            className={onStartTour ? "welcome__plain" : "welcome__go"}
            onClick={onClose}
            ref={onStartTour ? undefined : closeRef}
          >
            Skip, I&apos;ll look around myself
          </button>
          <span className="welcome__hint">
            The tour takes about a minute and points at each part of the page in
            turn. Both this and the tour stay available from the header.
          </span>
        </div>
      </div>
    </div>
  );
}
