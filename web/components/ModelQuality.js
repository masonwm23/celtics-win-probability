"use client";

import { useEffect, useState } from "react";
import { fetchModel } from "@/lib/api";

/**
 * How good is this model, and what does the number mean.
 *
 * Written because the dashboard had a gap. "Brier" appeared exactly once in the
 * whole interface, inside the lineup note, where it was carrying an argument:
 * lineup strength moved it 0.1630 to 0.1823 and was therefore dropped. A reader
 * who does not know the metric cannot tell whether that is a rounding error or
 * a disaster, so the evidence was unreadable to exactly the person it was meant
 * to convince. Meanwhile the model's OWN headline numbers appeared nowhere at
 * all: the interface said everything was out of fold, which is a claim about
 * honesty, and never said whether it was any good, which is a claim about
 * skill.
 *
 * This panel leads with the null result rather than burying it. The research
 * question was whether a Celtics-specific model beats a generic margin-and-time
 * one, and the answer on 636 games is that it does not. A panel that opened
 * with 0.163 and mentioned the tie in a footnote would be technically true and
 * would leave the reader with the wrong impression.
 *
 * Where the numbers come from:
 *
 *   - the shipped model's own metrics are FETCHED from /api/model, which
 *     returns models/model_metadata.json verbatim. Nothing here is retyped, so
 *     a retrain cannot leave the interface quoting stale figures.
 *   - the tier COMPARISON is not in that file, because the metadata describes
 *     one model and a comparison needs two. Those four numbers are pinned
 *     below, cited to the report they came out of.
 */

/**
 * The tier comparison, copied from reports/phase4_results.txt.
 *
 * This is the one block in this component that is not fetched, and it is
 * separated out and named so that it is obvious what has to change if the tier
 * comparison is ever re-run. The report's own line reads:
 *
 *   tier2_generic vs tier3_celtics   +0.0011  [-0.0029, +0.0048]   no
 *
 * where "no" is the report's answer to "is this difference real?", meaning the
 * interval includes zero. The bootstrap resamples GAMES, n = 636, not events:
 * events inside one game share an outcome, so resampling 308,975 correlated
 * rows as if independent would give an interval far too narrow.
 */
const TIER_COMPARISON = {
  source: "reports/phase4_results.txt",
  genericBrier: 0.1641,
  celticsBrier: 0.163,
  diff: 0.0011,
  ciLow: -0.0029,
  ciHigh: 0.0048,
  distinguishable: false,
};

const dp4 = (x) => x.toFixed(4);
const signed4 = (x) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(4)}`;
const pct1 = (x) => `${(x * 100).toFixed(1)}%`;

export default function ModelQuality() {
  const [meta, setMeta] = useState(null);
  const [error, setError] = useState(null);

  // The fold renders its children only once opened, so this fires the first
  // time somebody actually asks the question rather than on every page load.
  useEffect(() => {
    let alive = true;
    fetchModel()
      .then((m) => alive && setMeta(m))
      .catch((e) => alive && setError(e.message));
    return () => {
      alive = false;
    };
  }, []);

  if (error) {
    return (
      <div className="panel__body">
        <span className="badge badge--warn">could not load model metrics</span>
        <p className="note" style={{ margin: "9px 0 0" }}>
          {error}. The API has to be running for this panel; everything here
          comes from <code>/api/model</code> rather than from numbers typed into
          the interface.
        </p>
      </div>
    );
  }

  if (!meta) {
    return (
      <div className="panel__body">
        <p className="note" style={{ margin: 0 }}>Loading model metrics…</p>
      </div>
    );
  }

  const p = meta.out_of_fold_performance;
  const { genericBrier, celticsBrier, diff, ciLow, ciHigh } = TIER_COMPARISON;

  return (
    <div className="panel__body">
      {/* ---- the verdict, first, before any number is explained ---------- */}
      <div className="verdict">
        <div className="verdict__tag">The question this project asked</div>
        <p className="verdict__q">
          Does a model that knows about <em>these</em> players and{" "}
          <em>this</em> team predict the game better than a simple one that
          only watches the score and the clock?
        </p>
        <p className="verdict__a">
          <b>No. The two are too close to call, and that is the finding.</b>{" "}
          Across {meta.trained_on_games} games the Celtics-specific model came
          out very slightly ahead. But when the comparison is re-run thousands
          of times on different samples of those same games, the Celtics model
          wins some of those runs and loses the rest. That is what a tie looks
          like. Saying so is the honest answer; quoting the tiny lead on its own
          would have dressed a tie up as a win.
        </p>
        <p className="verdict__fine">
          For the record: {dp4(celticsBrier)} against {dp4(genericBrier)}, a
          difference of {signed4(diff)}, and re-running the comparison puts that
          difference anywhere between {signed4(ciLow)} and {signed4(ciHigh)} —
          a range that contains zero, meaning &ldquo;no difference&rdquo; is
          well within what the data supports.
        </p>
      </div>

      {/* ---- what the number actually is -------------------------------- */}
      <h3 className="mq__h">So what is a Brier score?</h3>
      <p className="mq__p">
        Think of grading a weather forecaster. If they say a 90% chance of rain
        and it rains, they did well. If they say 90% and the day stays dry, that
        is a bad miss. A Brier score is that idea totalled up: how far the
        forecast sat from what actually happened, averaged over every moment of
        every game. <b>Lower is better</b>, and zero would mean never being
        wrong.
      </p>
      <p className="mq__p">
        On its own, {dp4(p.brier)} means nothing to anybody. The way to read it
        is against someone who does not really forecast at all. The Celtics won{" "}
        {pct1(p.base_rate)} of these games, so a person who says{" "}
        &ldquo;{pct1(p.base_rate)}&rdquo; at every single moment and never
        reacts to anything that happens scores {dp4(p.baseline_brier)}. This
        model scores <b>{dp4(p.brier)}</b>. It has closed about{" "}
        {pct1(p.brier_skill)} of the distance between not trying and being
        perfect.
      </p>

      {/* ---- the scale. The two markers overlapping IS the point. -------- */}
      <div className="mqscale">
        <div className="mqscale__track">
          <span
            className="mqscale__mark mqscale__mark--generic"
            style={{ left: `${(genericBrier / p.baseline_brier) * 100}%` }}
            title={`Generic margin-and-time model, Brier ${dp4(genericBrier)}`}
          />
          <span
            className="mqscale__mark mqscale__mark--ours"
            style={{ left: `${(p.brier / p.baseline_brier) * 100}%` }}
            title={`This model, Brier ${dp4(p.brier)}`}
          />
        </div>
        <div className="mqscale__ends">
          <span>0 · perfect</span>
          <span>
            {dp4(p.baseline_brier)} · never move off {pct1(p.base_rate)}
          </span>
        </div>
        <p className="mqscale__cap">
          <i className="mqscale__key mqscale__key--ours" /> this model{" "}
          {dp4(p.brier)}
          <i className="mqscale__key mqscale__key--generic" /> the simple model{" "}
          {dp4(genericBrier)} &mdash; both are plotted above, and they land so
          close together that they overlap. You are looking at the answer to the
          research question: nobody can tell these two apart.
        </p>
      </div>

      {/* ---- the rest of the scorecard ---------------------------------- */}
      <h3 className="mq__h">The rest of the scorecard, if you want it</h3>
      <div className="impact">
        <div className="impact__card">
          <div className="impact__label">Brier score</div>
          <div className="impact__value" style={{ color: "var(--celtics)" }}>
            {dp4(p.brier)}
          </div>
          <div className="impact__sub">how wrong it was · lower is better</div>
        </div>

        <div className="impact__card">
          <div className="impact__label">Skill</div>
          <div className="impact__value">{pct1(p.brier_skill)}</div>
          <div className="impact__sub">of the way to a perfect score</div>
        </div>

        <div className="impact__card">
          <div className="impact__label">AUC</div>
          <div className="impact__value">{p.auc.toFixed(4)}</div>
          <div className="impact__sub">how often it backs the actual winner</div>
        </div>

        <div className="impact__card">
          <div className="impact__label">Log loss</div>
          <div className="impact__value">{p.logloss.toFixed(4)}</div>
          <div className="impact__sub">
            extra penalty for being sure and wrong
          </div>
        </div>

        <div className="impact__card">
          <div className="impact__label">Calibration error</div>
          <div className="impact__value">{p.ece.toFixed(4)}</div>
          <div className="impact__sub">
            when it says 70%, it happens ~70% of the time
          </div>
        </div>
      </div>

      <p className="note" style={{ marginTop: 14, marginBottom: 0 }}>
        <strong>Nothing above was graded on games the model had already
        seen.</strong> The model was trained eight separate times, each time
        with one full season hidden from it, and every season was then scored by
        the version that had never read it. That is what &ldquo;out of
        fold&rdquo; means on the rest of this page, and it is why{" "}
        {p.n.toLocaleString()} forecasts across {meta.trained_on_games} games
        count as a real test rather than a memory exercise. Two honest caveats,
        both stated in the paper rather than hidden: the AUC of{" "}
        {p.auc.toFixed(4)} is flattered by the fourth quarter, when a big lead
        really does decide the game and any model looks clever, so the
        early-game and close-game rows are the ones worth reading; and a further
        check suggests these figures are a touch optimistic, by about 0.0025
        Brier, compared with forecasting a genuinely unseen future season. The
        head-to-head comparison at the top comes from{" "}
        <code>{TIER_COMPARISON.source}</code>; every other figure here is read
        live from the deployed model&apos;s own record rather than typed in.
      </p>
    </div>
  );
}
