/**
 * The deployment model, evaluated in the browser.
 *
 * WHY THIS EXISTS
 *   Everything the timeline shows is a precomputed out-of-fold probability read
 *   from a static JSON file. Only the what-if slider needs a live prediction,
 *   and needing it was the only reason the dashboard required a Python process
 *   at all. A gradient boosted tree is a pile of if-statements, so the saved
 *   model is exported to JSON and evaluated here instead. The app becomes a
 *   static site: no server, no cold start, nothing to keep running.
 *
 * IT IS THE SAME MODEL, NOT AN APPROXIMATION OF IT
 *   The trees are the trees XGBoost saved. The arithmetic below reproduces
 *   XGBClassifier.predict_proba to float32 precision on all 308,975 rows of the
 *   dataset; the check lives in tools/verify_js_model.mjs and is part of the
 *   test suite. Two details make that true rather than nearly true:
 *
 *     1. XGBoost compares in float32. Doing it in JavaScript's float64 puts
 *        rows whose feature value sits within a rounding error of a split
 *        threshold down the wrong branch. Math.fround on both sides fixes it.
 *     2. base_score is stored in probability space and enters the sum in
 *        margin space, so it is the logit of the stored value.
 *
 * THE IN-SAMPLE CAVEAT STILL APPLIES
 *   This is the model fitted on all eight seasons. For every game in this
 *   dataset it is in-sample, so its number is not a fair estimate of accuracy.
 *   That was true when the prediction came from Python and it is true here.
 *   Callers must keep showing the caveat.
 */

export const FEATURE_ORDER = [
  "celtics_margin",
  "seconds_remaining_period",
  "seconds_remaining_game",
  "seconds_elapsed_game",
  "period",
  "is_overtime",
  "celtics_is_home",
  "celtics_has_possession",
  "momentum_120s",
  "momentum_300s",
  "is_clutch",
  "margin_per_minute_remaining",
  "possession_number",
];

// The NBA definition, matching src/features.py. Kept as named constants so the
// two implementations can be diffed by eye rather than by trust.
export const CLUTCH_SECONDS = 300;
export const CLUTCH_MARGIN = 5;

/** Features the model is given but that are computed from other features. */
export const DERIVED_FEATURES = ["is_clutch", "margin_per_minute_remaining"];

/**
 * Recompute the derived features after an override.
 *
 * This is the JavaScript half of src/features.py's recompute_derived, and it
 * exists because of a real bug: overriding celtics_margin while leaving
 * margin_per_minute_remaining and is_clutch at their original values made the
 * model report a HIGHER win probability for a bigger deficit. The override has
 * to propagate. `skip` holds any feature the caller set explicitly, which is
 * never overwritten.
 */
export function recomputeDerived(row, skip = []) {
  const out = { ...row };
  if (!skip.includes("margin_per_minute_remaining")) {
    const minutesLeft = Math.max(out.seconds_remaining_game / 60, 1 / 60);
    out.margin_per_minute_remaining = out.celtics_margin / minutesLeft;
  }
  if (!skip.includes("is_clutch")) {
    out.is_clutch =
      out.period >= 4 &&
      out.seconds_remaining_period <= CLUTCH_SECONDS &&
      Math.abs(out.celtics_margin) <= CLUTCH_MARGIN
        ? 1
        : 0;
  }
  return out;
}

const f32 = Math.fround;

/** Turn a named row into the ordered float32 vector the trees index into. */
export function toVector(row) {
  return FEATURE_ORDER.map((name) => {
    const v = row[name];
    if (v === null || v === undefined) return NaN;
    return f32(typeof v === "boolean" ? (v ? 1 : 0) : Number(v));
  });
}

/**
 * Probability that Boston win, from a named feature row.
 *
 * `model` is the parsed model_trees.json. Pass `skip` through from the caller's
 * override list so an explicitly set derived feature is respected.
 */
export function predict(model, row, { recompute = true, skip = [] } = {}) {
  const prepared = recompute ? recomputeDerived(row, skip) : row;
  return predictVector(model, toVector(prepared));
}

/** The tree walk itself. Exported so the verifier can drive it directly. */
export function predictVector(model, x) {
  let margin = Math.log(model.base_score / (1 - model.base_score));

  for (const tree of model.trees) {
    const { f, t, l, r, d } = tree;
    let i = 0;
    while (l[i] !== -1) {
      const v = x[f[i]];
      // NaN takes the default branch; otherwise strictly-less-than goes left,
      // compared in float32 exactly as XGBoost does it.
      i = Number.isNaN(v)
        ? d[i]
          ? l[i]
          : r[i]
        : v < f32(t[i])
          ? l[i]
          : r[i];
    }
    margin += t[i]; // leaf value lives in split_conditions for leaf nodes
  }

  return 1 / (1 + Math.exp(-margin));
}

/** Convenience for the what-if panel: base row + overrides -> probability. */
export function whatIf(model, baseRow, overrides) {
  const merged = { ...baseRow, ...overrides };
  return {
    probability: predict(model, merged, { skip: Object.keys(overrides) }),
    derivedRecomputed: DERIVED_FEATURES.filter(
      (name) => !Object.keys(overrides).includes(name)
    ),
    caveat:
      "This number comes from the deployment model, which was fitted on all " +
      "eight seasons. For any game in this dataset that is in-sample, so it is " +
      "not a fair estimate of accuracy. The timeline probabilities are out of " +
      "fold; this one is not.",
  };
}
