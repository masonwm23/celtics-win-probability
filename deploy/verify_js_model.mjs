/**
 * Does the browser model give the same answer as XGBoost?
 *
 * Two checks, both against predictions produced by XGBClassifier.predict_proba
 * in Python and written to disk. The first covers every row in the dataset. The
 * second covers the what-if path specifically, where a margin override has to
 * propagate into the two derived features before the trees see it — the case
 * that was a real bug in the Python API and would be just as easy to get wrong
 * a second time here.
 *
 * The bar is float32 precision, because that is the precision XGBoost itself
 * evaluates in. Anything looser would be hiding a branch disagreement.
 */
import fs from "fs";
import { predictVector, predict, FEATURE_ORDER } from "./wp-model.js";

const TOL = 1e-6;
const here = new URL(".", import.meta.url).pathname;
const model = JSON.parse(fs.readFileSync(here + "model_trees.json", "utf8"));

let failed = false;
const report = (name, max, mean, n, over) => {
  const pass = max <= TOL && over === 0;
  if (!pass) failed = true;
  console.log(
    `${pass ? "PASS" : "FAIL"}  ${name}\n` +
      `      rows ${n.toLocaleString()}  max |diff| ${max.toExponential(3)}` +
      `  mean ${mean.toExponential(3)}  rows over ${TOL}: ${over}`
  );
};

// ---- 1. every row in the dataset -------------------------------------------
{
  const X = new Float32Array(fs.readFileSync(here + "_X.f32").buffer);
  const P = new Float64Array(fs.readFileSync(here + "_p.f64").buffer);
  const k = FEATURE_ORDER.length;
  const n = P.length;
  if (X.length !== n * k) throw new Error("feature matrix does not match predictions");

  let max = 0, sum = 0, over = 0;
  const row = new Array(k);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < k; j++) row[j] = X[i * k + j];
    const d = Math.abs(predictVector(model, row) - P[i]);
    if (d > max) max = d;
    if (d > TOL) over++;
    sum += d;
  }
  report("all events, vector path", max, sum / n, n, over);
}

// ---- 2. the what-if path, including derived-feature recomputation -----------
{
  const cases = JSON.parse(fs.readFileSync(here + "_whatif_cases.json", "utf8"));
  const n = cases.rows.length;
  let max = 0, sum = 0, over = 0;
  for (let i = 0; i < n; i++) {
    // recompute=true on purpose: the JS must derive is_clutch and
    // margin_per_minute_remaining itself and land on what Python derived.
    const got = predict(model, cases.rows[i], { recompute: true, skip: [] });
    const d = Math.abs(got - cases.expected[i]);
    if (d > max) max = d;
    if (d > TOL) over++;
    sum += d;
  }
  report("what-if overrides, derived features recomputed in JS", max, sum / n, n, over);
}

// ---- 3. the derived features actually propagate -----------------------------
// The original bug was that overriding celtics_margin left
// margin_per_minute_remaining stale, so the model never saw the change. This
// asserts the override reaches the trees, which is what the fix guaranteed.
// It deliberately does NOT assert that probability rises monotonically with
// margin: it does not, and that is a property of the deployment model rather
// than of this port. See tools/characterise_whatif.md — at large deficits the
// deployment model is recalling training games, not forecasting, which is why
// the what-if panel constrains its slider and carries the in-sample caveat.
{
  const cases = JSON.parse(fs.readFileSync(here + "_whatif_cases.json", "utf8"));
  const base = cases.rows[0];
  const at = (margin) =>
    predict(model, { ...base, celtics_margin: margin }, { skip: ["celtics_margin"] });

  const distinct = new Set([-20, -10, 0, 10, 20].map((m) => at(m).toFixed(6)));
  const propagates = distinct.size === 5;
  if (!propagates) failed = true;
  console.log(
    `${propagates ? "PASS" : "FAIL"}  a margin override reaches the trees ` +
      `(${distinct.size}/5 distinct probabilities)`
  );

  // Over the range the UI exposes, the curve should at least trend upward.
  const lo = at(-15), hi = at(15);
  const trends = hi > lo;
  if (!trends) failed = true;
  console.log(
    `${trends ? "PASS" : "FAIL"}  within the slider range, +15 beats -15 ` +
      `(${(lo * 100).toFixed(1)}% vs ${(hi * 100).toFixed(1)}%)`
  );
}

process.exit(failed ? 1 : 0);
