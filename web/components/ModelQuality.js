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
 * THE PROSE IN THIS PANEL IS THE AUTHOR'S OWN, supplied Aug 4 and used verbatim
 * apart from spelling and punctuation. Do not rewrite it. Two consequences to
 * be aware of before editing anything here:
 *
 *   - the benchmark is introduced once as "the generic model based on ESPN",
 *     at the author's explicit instruction, and is "the generic model" every
 *     time after that. His own sentence sets up that shorthand, so do not go
 *     back to writing "the ESPN model": ESPN does not publish theirs, which is
 *     stated outright in reports/phase4_results.txt line 36.
 *   - three figures in the copy disagree with the sources this file reads
 *     from. They are marked VERBATIM at the point of use so that nobody later
 *     mistakes them for interpolation bugs and "fixes" them.
 *
 * Where the numbers come from:
 *
 *   - the shipped model's own metrics are FETCHED from /api/model, which
 *     returns models/model_metadata.json verbatim. Nothing here is retyped, so
 *     a retrain cannot leave the interface quoting stale figures. Every figure
 *     in the author's copy that MATCHES its source is interpolated rather than
 *     hardcoded, so the rendered sentence reads exactly as he wrote it while
 *     staying live.
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
  const { genericBrier, celticsBrier, ciLow, ciHigh } = TIER_COMPARISON;

  return (
    <div className="panel__body">
      {/* ---- the verdict, first, before any number is explained ---------- */}
      <div className="verdict">
        <div className="verdict__tag">The question that was asked</div>
        <p className="verdict__q">
          Is the Celtics Live Win Probability (CLWP), a Celtics specific model
          that incorporates player specific data and team tendencies superior to
          the generic model based on ESPN that mainly prioritizes the game score
          and clock? Note: due to trademark restrictions, the generic model
          based on ESPN will be referred to as the generic model.
        </p>
        {/* VERBATIM: the copy below states the difference as −0.0011 and the
            resample count as 10-100. reports/phase4_results.txt records +0.0011
            (the CLWP Brier is the LOWER of the two, which is the advantage) and
            the bootstrap ran thousands of resamples. Both were raised with the
            author and he chose to keep his wording, so neither is interpolated
            from TIER_COMPARISON. Do not "correct" them here; that is a
            conversation to have with him. */}
        <p className="verdict__a">
          The answer is not really. The two models are too close to call with a
          Brier score difference of only −0.0011 for the CLWP model (Brier:{" "}
          {dp4(celticsBrier)} CLWP, {dp4(genericBrier)} generic), which
          essentially can be considered a draw. Across {meta.trained_on_games}{" "}
          games, the CLWP Celtics specific model has a slight advantage in
          predictive power, but not significantly so. When, though, the
          comparison between the two models is re-run 10-100 times on these same
          games providing a larger statistical sampling, the two models produce
          almost identical results with the CLWP model still maintaining a
          slight lead, but too small a lead to be called a decisive win. The
          major advantage of the CLWP model is that it does increase the
          accuracy of probability predictions for the Celtics games when
          compared to the always {pct1(p.base_rate)} approach.
        </p>
        {/* THE INTERVAL IS FLIPPED ON PURPOSE.
            reports/phase4_results.txt line 61 reads

              tier2_generic vs tier3_celtics  +0.0011  [-0.0029, +0.0048]  no

            which is generic MINUS celtics, so a POSITIVE number there means the
            Celtics model did better. The copy above states the point estimate
            the other way round, as celtics minus generic (−0.0011, i.e. the
            lower Brier of the two). Quoting the report's interval unchanged
            next to it would put the point estimate on one scale and the range
            on the opposite one, and a reader checking the arithmetic would find
            −0.0011 sitting inside a range where negative means WORSE.

            So the interval is negated to match, which is the same result
            expressed the other way: [−0.0048, +0.0029]. It is derived from
            ciLow/ciHigh rather than typed, so re-running the bootstrap still
            flows through correctly. TIER_COMPARISON keeps the report's own
            signs and must not be edited to "fix" this. */}
        <p className="verdict__fine">
          The CLWP Brier: {dp4(celticsBrier)} vs. the generic model{" "}
          {dp4(genericBrier)}{" "}
          after rerunning the games allows for a difference between{" "}
          {signed4(-ciHigh)} and {signed4(-ciLow)}, a range that does contain
          statistical significance and best describes the results from the two
          models as a statistical tie.
        </p>
      </div>

      {/* ---- what the number actually is -------------------------------- */}
      <h3 className="mq__h">So what is a Brier score?</h3>
      <p className="mq__p">
        The Brier score measures the accuracy of probability predictions, not
        only if a prediction was correct or wrong. A lower Brier score is
        better, with 0 being perfect (never wrong). The Brier score rewards
        being confidently correct and penalizes being confidently wrong.
      </p>
      <p className="mq__p">
        For example, in basketball terms: If you predict the Celtics have a 70%
        chance to win, they should win about 7 out of 10 similar games. If you
        constantly predict 99% and they lose more often than expected, your
        Brier score becomes much higher (worse) as a result of incorrect
        overconfidence. A good Brier score reflects not only picking winners,
        but also assigning realistic probabilities to each outcome.
      </p>

      <h3 className="mq__h">
        The true advantage of the CLWP Celtics specific model
      </h3>
      <p className="mq__p">
        The Celtics won {pct1(p.base_rate)} of the games (games used as data in
        this model and model comparison), so throughout every game predicted,
        they would earn a Brier score of {dp4(p.baseline_brier)}. The CLWP
        continuously updates its predictions based on what is occurring, in real
        time, during the game where the CLWP model achieves a Brier score of{" "}
        {dp4(p.brier)}. Therefore, the CLWP model reduces the prediction error
        by {pct1(p.brier_skill)} compared with the always {pct1(p.base_rate)}{" "}
        approach. The CLWP model moves approximately one-third of the way from a
        static, no-skill prediction toward perfect forecasting (a Brier of 0).
      </p>

      {/* ---- the scale. The two markers overlapping IS the point. -------- */}
      <div className="mqscale">
        <div className="mqscale__track">
          <span
            className="mqscale__mark mqscale__mark--generic"
            style={{ left: `${(genericBrier / p.baseline_brier) * 100}%` }}
            title={`Generic model based on ESPN, Brier ${dp4(genericBrier)}`}
          />
          <span
            className="mqscale__mark mqscale__mark--ours"
            style={{ left: `${(p.brier / p.baseline_brier) * 100}%` }}
            title={`CLWP model, Brier ${dp4(p.brier)}`}
          />
        </div>
        <div className="mqscale__ends">
          <span>0 · perfect</span>
          <span>
            {dp4(p.baseline_brier)} · never move off {pct1(p.base_rate)}
          </span>
        </div>
        {/* The key stays, without its numbers. The paragraph below already
            states both Brier scores, and printing them twice a line apart read
            as a mistake rather than as a legend. */}
        <p className="mqscale__cap">
          <i className="mqscale__key mqscale__key--ours" /> CLWP
          <i className="mqscale__key mqscale__key--generic" /> generic
        </p>
        <p className="mqscale__cap" style={{ marginTop: 6 }}>
          The CLWP model achieved a Brier score of {dp4(p.brier)} compared with{" "}
          {dp4(genericBrier)} for the generic model based on ESPN. As shown in
          the plot above,
          there is significant overlap since the difference is very small. In
          practical terms, the models are indistinguishable in predictive
          performance. Therefore, the more complex CLWP Celtics specific model
          provides no meaningful improvement over the simpler generic model. This
          result directly answers the research question.
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
        No results presented were evaluated on Celtics games that the CLWP model
        had previously observed during training. Model performance was assessed
        using a leave one season out cross validation framework. With this, the
        model was trained eight separate times, each time withholding one
        complete season for testing. So, every season was evaluated by a CLWP
        model that had never been exposed to that season&apos;s data during
        training. This evaluation strategy is referred to as
        &ldquo;out-of-fold&rdquo; testing. The resulting {p.n.toLocaleString()}{" "}
        forecasts spanning {meta.trained_on_games} games therefore constitute a
        thorough assessment of generalization performance rather than an
        evaluation of learned observations. Two important limitations should be
        pointed out, both of which are discussed plainly in the accompanying
        paper. First, the overall AUC of {p.auc.toFixed(4)} is influenced by
        fourth-quarter game states, where large leads make game outcomes
        comparatively easy to predict. Accordingly, the early-game and
        close-game results provide a more informative evaluation of the
        model&apos;s discriminative capability under conditions of larger
        uncertainty. Second, an additional validation analysis indicates that
        the reported performance metrics are modestly positive, with the Brier
        score expected to increase by approximately 0.0025 when forecasting an
        entirely unseen future season. This degree of confidence is consistent
        with expected differences between cross-validation performance and
        prospective out-of-sample evaluation. The head-to-head comparison
        presented at the top of the page is reproduced directly from{" "}
        <code>{TIER_COMPARISON.source}</code>. All remaining performance metrics
        and figures are generated dynamically from the deployed model&apos;s
        recorded outputs rather than being manually inputted, confirming that
        the reported results accurately reflect the CLWP model&apos;s
        operational performance.
      </p>
    </div>
  );
}
