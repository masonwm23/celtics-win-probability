"use client";

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";

/**
 * A guided tour that points at the REAL dashboard.
 *
 * The ask was screenshots of each section with an explanation. This does the
 * same job without the failure mode: it measures the live element and cuts a
 * hole in a dimmed overlay around it, so the reader is looking at the actual
 * court, the actual chart and the actual numbers rather than a picture of them.
 * Five sections changed today. A set of captured images would already be out of
 * date, and a walkthrough showing a panel that no longer exists is worse than
 * no walkthrough at all. This cannot go stale, because there is nothing to keep
 * in step.
 *
 * Steps target `data-tour` attributes rather than CSS classes. A class is a
 * styling decision and gets renamed the moment somebody restyles a panel; a
 * data attribute exists only for this, so a refactor that moves a section keeps
 * the tour pointing at it.
 *
 * MISSING TARGETS ARE SKIPPED, NOT FATAL. Two folds are conditional: opponent
 * context is absent when the payload has none, and the selected-player fold
 * only exists once somebody clicks a player. A tour that threw, or that
 * spotlighted the top-left corner of the page because it measured null, would
 * be a worse bug than the one it is explaining. Steps whose element is not on
 * the page are dropped when the tour starts.
 */

/**
 * The script.
 *
 * THE TITLES AND BODIES ARE THE AUTHOR'S OWN, supplied Aug 4, used verbatim
 * apart from three typo fixes ("until the come finishes" -> "the game", "the
 * arrow keys prove" -> "provide", "This button bring back" -> "brings"). Do not
 * rewrite them.
 *
 * Note that step 3 is titled "The Probability Chart", not "The line". He asked
 * for "the line" to be dropped as a name for it everywhere it appeared, so
 * step 4's body and the Welcome panel were changed to match.
 *
 * `opens` names a fold that has to be open before the step can point at
 * anything, because a collapsed fold is a 44-pixel button and spotlighting it
 * teaches nobody anything.
 */
export const TOUR_STEPS = [
  {
    target: "scoreboard",
    title: "The Score and Win Probability Percentage",
    body:
      "This is the score and clock information at the exact time point in the "
      + "game shown. The large number in green at the center of the highlighted "
      + "display is the Celtic's win percentage at this exact moment of the game.",
  },
  {
    target: "court",
    title: "On The Court",
    body:
      "Indicated who took the shot. The bold green circle indicated if the "
      + "basket was made and the amber cross indicated a missed basket. The ten "
      + "players faces indicated who is on the floor, but not where they "
      + "actually stood during the game.",
  },
  {
    target: "chart",
    title: "The Probability Chart",
    body:
      "Every point in the chart represents one play in the game. Points are "
      + "added to the chart as plays develop during the game until the game "
      + "finishes. The larger the value on the y-axis, the higher the Celtics "
      + "win probability. Drag the mouse along the x-axis to see how the win "
      + "probability changes as the game progresses. This identifies key plays "
      + "that significantly influence the game outcome.",
  },
  {
    target: "play",
    title: "What Just Happened - Real Time Game Assessment",
    body:
      "The detailed description of the current point (or play) on the "
      + "probability chart. The numbers to the right of the box provides data "
      + "on what this play caused to the Celtics' probability to win. Green "
      + "indicates the probability to win went up, while red lowers the chances "
      + "to win.",
  },
  {
    target: "swings",
    title: "The Big Moments - Decisive Play",
    body:
      "The play in the game that caused the game win probability significantly "
      + "change. Click on one to jump to that play.",
  },
  {
    target: "transport",
    title: "Watch It Play",
    body:
      "Press the play button and the model runs the game. The speed control is "
      + "located on the right side to adjust the game pace. 0.5x allows for the "
      + "best view on every play, while 2x provides the fastest option for "
      + "general overview. The space bar starts and stops the game and the "
      + "arrow keys provide a step wise \u201cone play at a time\u201d option.",
  },
  {
    target: "model",
    opens: "model",
    title: "The Value of the CLWP Model",
    body:
      "The accuracy and the value of the model is explained in this section.",
  },
  {
    target: "whatif",
    opens: "whatif",
    title: "Change the Score",
    body:
      "Drag the bar to change the score and the CLWP model will adapt and "
      + "provide an updated win probability based on the new score.",
  },
  {
    target: "whatisthis",
    title: "Help",
    body:
      "This button brings back the tour to help the user understand the various "
      + "features and utilities.",
  },
];

const PAD = 8;

/** Gap between the card and the thing it points at. */
const GAP = 14;
/** Minimum gap between the card and the edge of the window. */
const EDGE = 16;

/**
 * Where to put the caption card.
 *
 * The first version tried below the target, then above it, and clamped only the
 * horizontal position. That works until the target is taller than the window,
 * which the court is on a laptop: "below" fell off the bottom, so it flipped to
 * "above", and because the target's top was already off-screen the card was
 * positioned above THAT — half of it above the top edge, unreadable, with the
 * Next button barely reachable.
 *
 * So this tries four sides in order of preference and takes the first that
 * genuinely fits, measuring the card rather than assuming a height. If nothing
 * fits, which happens when a target fills the window, it pins to a corner. The
 * final clamp is unconditional: whatever else happens, the card is on screen.
 */
function place(rect, card) {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const clampLeft = (l) => Math.max(EDGE, Math.min(l, vw - card.w - EDGE));
  const clampTop = (t) => Math.max(EDGE, Math.min(t, vh - card.h - EDGE));

  if (!rect) {
    return { top: Math.max(EDGE, (vh - card.h) / 2), left: clampLeft((vw - card.w) / 2) };
  }

  const below = rect.top + rect.height + GAP;
  if (below + card.h + EDGE <= vh) return { top: below, left: clampLeft(rect.left) };

  const above = rect.top - GAP - card.h;
  if (above >= EDGE) return { top: above, left: clampLeft(rect.left) };

  const right = rect.left + rect.width + GAP;
  if (right + card.w + EDGE <= vw) return { top: clampTop(rect.top), left: right };

  const left = rect.left - GAP - card.w;
  if (left >= EDGE) return { top: clampTop(rect.top), left };

  // The target fills the window. Sit in the bottom-left, which on this layout
  // is the quietest corner of the court.
  return { top: clampTop(vh - card.h - EDGE), left: clampLeft(EDGE) };
}

export default function Tour({ open, onClose, onOpenFold }) {
  const [index, setIndex] = useState(0);
  const [steps, setSteps] = useState([]);
  const [rect, setRect] = useState(null);
  const cardRef = useRef(null);
  const [cardSize, setCardSize] = useState({ w: 380, h: 210 });

  // Drop steps whose element is not on the page, once, when the tour starts.
  useEffect(() => {
    if (!open) return;
    setSteps(
      TOUR_STEPS.filter((s) => document.querySelector(`[data-tour="${s.target}"]`))
    );
    setIndex(0);
  }, [open]);

  const step = steps[index] || null;

  // Opening a fold changes the page height, so this runs before paint and the
  // measurement below sees the expanded panel rather than the collapsed one.
  useLayoutEffect(() => {
    if (!open || !step?.opens) return;
    onOpenFold?.(step.opens);
  }, [open, step, onOpenFold]);

  const measure = useCallback(() => {
    if (!step) return;
    const node = document.querySelector(`[data-tour="${step.target}"]`);
    if (!node) {
      setRect(null);
      return;
    }
    const box = node.getBoundingClientRect();
    setRect({
      top: box.top - PAD,
      left: box.left - PAD,
      width: box.width + PAD * 2,
      height: box.height + PAD * 2,
    });
  }, [step]);

  // Scroll the target into view, then measure. The timeout lets a smooth scroll
  // and any fold animation settle; measuring immediately would spotlight where
  // the element used to be.
  useEffect(() => {
    if (!open || !step) return undefined;
    const node = document.querySelector(`[data-tour="${step.target}"]`);
    // Centring a target taller than the window puts its top off-screen, which
    // leaves the caption nowhere good to go and the reader looking at the
    // middle of a court with no idea what is being pointed at. Tall targets get
    // their top aligned instead, so the highlight starts where the eye does.
    const tall = node && node.getBoundingClientRect().height > window.innerHeight * 0.7;
    node?.scrollIntoView({ behavior: "smooth", block: tall ? "start" : "center" });
    const settle = setTimeout(measure, 380);

    window.addEventListener("resize", measure);
    window.addEventListener("scroll", measure, true);
    return () => {
      clearTimeout(settle);
      window.removeEventListener("resize", measure);
      window.removeEventListener("scroll", measure, true);
    };
  }, [open, step, measure]);

  // The card's height depends on how long the caption is, so it is measured
  // rather than guessed. Guarded on a real change, because this effect has no
  // dependency array and would otherwise loop.
  useLayoutEffect(() => {
    const node = cardRef.current;
    if (!node) return;
    const box = node.getBoundingClientRect();
    setCardSize((prev) =>
      Math.abs(prev.w - box.width) < 1 && Math.abs(prev.h - box.height) < 1
        ? prev
        : { w: box.width, h: box.height });
  });

  const next = useCallback(() => {
    setIndex((i) => (i + 1 < steps.length ? i + 1 : i));
  }, [steps.length]);
  const back = useCallback(() => setIndex((i) => Math.max(0, i - 1)), []);

  useEffect(() => {
    if (!open) return undefined;
    function onKey(e) {
      if (e.key === "Escape") { e.preventDefault(); onClose(); }
      else if (e.key === "ArrowRight" || e.key === "Enter") { e.preventDefault(); if (index + 1 < steps.length) next(); else onClose(); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); back(); }
    }
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [open, onClose, next, back, index, steps.length]);

  if (!open || !step) return null;

  const last = index === steps.length - 1;
  const pos = place(rect, cardSize);

  return (
    <div className="tour" role="dialog" aria-modal="true" aria-label="Dashboard tour">
      {/* Four panes around the target rather than one box-shadow, so the hole
          is a real gap: clicks inside it still reach the dashboard and the
          reader can try the thing being described without leaving the tour. */}
      {rect ? (
        <>
          <div className="tour__mask" style={{ top: 0, left: 0, right: 0, height: `${Math.max(0, rect.top)}px` }} onClick={onClose} />
          <div className="tour__mask" style={{ top: `${rect.top + rect.height}px`, left: 0, right: 0, bottom: 0 }} onClick={onClose} />
          <div className="tour__mask" style={{ top: `${rect.top}px`, left: 0, width: `${Math.max(0, rect.left)}px`, height: `${rect.height}px` }} onClick={onClose} />
          <div className="tour__mask" style={{ top: `${rect.top}px`, left: `${rect.left + rect.width}px`, right: 0, height: `${rect.height}px` }} onClick={onClose} />
          <div
            className="tour__ring"
            style={{ top: `${rect.top}px`, left: `${rect.left}px`, width: `${rect.width}px`, height: `${rect.height}px` }}
          />
        </>
      ) : (
        <div className="tour__mask tour__mask--full" onClick={onClose} />
      )}

      <div
        className="tour__card"
        ref={cardRef}
        style={{ top: `${pos.top}px`, left: `${pos.left}px` }}
      >
        <div className="tour__count">
          Step {index + 1} of {steps.length}
        </div>
        <div className="tour__title">{step.title}</div>
        <p className="tour__body">{step.body}</p>
        <div className="tour__actions">
          <button className="tour__skip" onClick={onClose}>
            Skip tour
          </button>
          <div className="tour__nav">
            {index > 0 && (
              <button className="tour__btn" onClick={back}>
                Back
              </button>
            )}
            <button className="tour__btn tour__btn--go" onClick={last ? onClose : next}>
              {last ? "Done" : "Next"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}


export const TOUR_NUDGE_KEY = "celtics-wp-tour-nudge-stage";

/**
 * Decided once per page load, then remembered for the rest of it.
 *
 * React Strict Mode, which this app enables, mounts every component in
 * development, unmounts it, and mounts it again. That made the effect asking
 * this question run twice: the first call found no key, decided to show the
 * card and wrote "shown" to storage; the second call read that key back and
 * concluded the visitor had already been here. React discards the state from
 * the first mount, so the card never rendered and the whole first-run
 * experience was invisible in development.
 *
 * Module scope survives that remount. Component state and refs do not, which
 * is why the guard lives here rather than in the component.
 */
let decidedThisLoad;

/**
 * Whether to offer the tour at all.
 *
 *   first visit   "centre"  a large card in the middle of the screen
 *   after that    null      nothing; the header button is the only offer
 *
 * One showing. It is a large panel over somebody's first view of the page, and
 * a second one would be an interruption rather than an introduction: by then
 * they have seen the header and know the tour exists.
 *
 * Reading storage throws outright in Safari's private mode, so a failure falls
 * back to showing it. For a one-off visitor, seeing the offer is a much smaller
 * cost than never seeing it — and a private window is exactly how somebody
 * opens a link they were sent.
 */
export function nudgeStage() {
  if (decidedThisLoad !== undefined) return decidedThisLoad;
  try {
    decidedThisLoad = window.localStorage.getItem(TOUR_NUDGE_KEY) ? null : "centre";
  } catch {
    decidedThisLoad = "centre";
  }
  return decidedThisLoad;
}

/**
 * Record that it has been shown, so it never opens itself again.
 *
 * Safe to call more than once, which Strict Mode guarantees it will be.
 */
export function advanceNudge() {
  try {
    window.localStorage.setItem(TOUR_NUDGE_KEY, "shown");
  } catch {
    /* nothing to do; the offer simply will not persist */
  }
}

/** Same thing here: shown once is shown for good. */
export function silenceNudge() {
  advanceNudge();
}

/**
 * The prompt telling somebody the tour exists.
 *
 * The welcome panel offers the tour too, but only on a true first visit.
 * Anybody returning, which includes the person presenting this on their own
 * laptop, would otherwise never learn the tour is there.
 *
 * `variant` decides how much room it takes. Neither variant traps focus or
 * blocks the page: the dashboard is readable without it, and somebody who wants
 * to get straight to the basketball should not have to dismiss anything.
 */
export function TourNudge({ open, variant = "side", onStart, onDismiss, onSoftClose }) {
  if (!open) return null;
  const centre = variant === "centre";
  return (
    <>
      {/* The centred stage gets a backdrop. Without one the card sits on the
          same surface colour as every panel behind it and reads as just
          another panel that happens to be floating -- which is exactly what it
          looked like. It dims rather than blacks out, because the point is
          still to show somebody the dashboard, not to hide it. */}
      {centre && (
        <div
          className="nudge__scrim"
          onClick={onSoftClose || onDismiss}
          aria-hidden="true"
        />
      )}
      <div
        className={`nudge ${centre ? "nudge--centre" : "nudge--side"}`}
        role="complementary"
      >
        {centre && <div className="nudge__eyebrow">New here?</div>}
        <div className="nudge__text">
          {centre ? (
            <>
              This page replays real Celtics games and shows what a model
              thought their chances were.{" "}
              <b>A short tour points out what everything means.</b>
            </>
          ) : (
            <>
              <b>First time here?</b> A short tour points out what everything on
              this page means.
            </>
          )}
        </div>
        <div className="nudge__row">
          <button className="nudge__go" onClick={onStart}>
            Take the tour
          </button>
          <button className="nudge__no" onClick={onDismiss}>
            No thanks
          </button>
        </div>
        {centre && (
          <div className="nudge__hint">
            It is always available from <b>Take the tour</b> at the top.
          </div>
        )}
      </div>
    </>
  );
}
