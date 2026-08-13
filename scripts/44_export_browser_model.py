"""
Phase 9c: export the trained booster to the browser what-if model.

WHY THIS EXISTS
---------------
The dashboard timeline replays precomputed out-of-fold probabilities from static
JSON. The ONLY live prediction is the what-if slider, and it runs the saved
model in the browser by reading model_trees.json (see web/lib/wp-model.js). No
other step regenerated that file, so after a retrain it silently went stale:
the trees, and base_score, still described the previous model.

This script closes that gap. It is the deterministic exporter, run AFTER
scripts/11_train_model.py, and it is the reason a retrain no longer leaves the
what-if panel on an old model.

WHAT IT WRITES
    web/public/data/model_trees.json   the copy the app actually loads
    deploy/model_trees.json            the self-contained kit copy
    deploy/_X.f32, _p.f64, _whatif_cases.json   fixtures for verify_js_model.mjs

HOW IT PROVES ITSELF
--------------------
Before writing anything it re-derives XGBClassifier.predict_proba with a float32
tree walk that mirrors web/lib/wp-model.js line for line, and compares against
the real predict_proba over EVERY row in the model frame. The bar is float32
precision (max |diff| <= 1e-6), which is the precision XGBoost itself evaluates
in and the same bar tools/verify_js_model.mjs uses. If the export is not exact
the script raises and writes nothing.

Two details make the port exact rather than nearly exact, both handled here and
in wp-model.js:
  1. XGBoost compares in float32. A float64 comparison sends rows whose value
     sits within a rounding error of a split threshold down the wrong branch.
  2. base_score is stored in probability space and enters the margin sum as its
     logit.

HOW TO RUN
    Open this file in Spyder and press F5.
    Then, to cross-check in the actual JavaScript, run in a terminal:
        node deploy/verify_js_model.mjs
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

# Must match web/lib/wp-model.js FEATURE_ORDER exactly. The saved model's
# metadata carries the same list; this is asserted against it below.
FEATURE_ORDER = [
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
]

# NBA clutch definition, matching src/features.py and wp-model.js.
CLUTCH_SECONDS = 300
CLUTCH_MARGIN = 5

TOL = 1e-6

MODEL_PATH = config.MODELS_DIR / "win_probability_model.joblib"
METADATA_PATH = config.MODELS_DIR / "model_metadata.json"
DEPLOY_DIR = config.PROJECT_ROOT / "deploy"
WEB_DATA_DIR = config.PROJECT_ROOT / "web" / "public" / "data"


def read_base_score(booster) -> float:
    """base_score in probability space, from the booster config.

    Newer XGBoost stores it as a bracketed JSON array string, e.g. "[6.57539E-1]".
    """
    cfg = json.loads(booster.save_config())
    raw = cfg["learner"]["learner_model_param"]["base_score"]
    text = str(raw).strip()
    return float(json.loads(text)[0]) if text.startswith("[") else float(text)


def build_model(booster) -> dict:
    """Booster -> {base_score, trees:[{f,t,l,r,d}]}, the format wp-model.js reads.

    Per node: f split-feature index, t threshold (or leaf value when l == -1),
    l/r the yes/no child node indices, d True when a missing value takes the
    yes (left) branch. Feature names are f0..f12, which map directly to
    FEATURE_ORDER indices.
    """
    df = booster.trees_to_dataframe()
    trees = []
    for _, group in df.groupby("Tree", sort=True):
        group = group.sort_values("Node")
        size = int(group["Node"].max()) + 1
        f = [0] * size
        t = [0.0] * size
        left = [-1] * size
        right = [-1] * size
        default_left = [False] * size
        for row in group.itertuples():
            node = int(row.Node)
            if row.Feature == "Leaf":
                t[node] = float(row.Gain)          # leaf output value
            else:
                f[node] = int(str(row.Feature)[1:])   # 'f7' -> 7
                t[node] = float(row.Split)
                yes = int(str(row.Yes).split("-")[1])
                no = int(str(row.No).split("-")[1])
                miss = int(str(row.Missing).split("-")[1])
                left[node] = yes
                right[node] = no
                default_left[node] = (miss == yes)
        trees.append({"f": f, "t": t, "l": left, "r": right, "d": default_left})
    return {"base_score": read_base_score(booster), "trees": trees}


def predict_all(model: dict, X: np.ndarray) -> np.ndarray:
    """Vectorised float32 tree walk mirroring wp-model.js predictVector."""
    n = X.shape[0]
    margin = np.full(
        n, np.log(model["base_score"] / (1 - model["base_score"])), dtype=np.float64)
    row_ids = np.arange(n)
    for tree in model["trees"]:
        f = np.asarray(tree["f"], np.int64)
        t = np.asarray(tree["t"], np.float32)
        left = np.asarray(tree["l"], np.int64)
        right = np.asarray(tree["r"], np.int64)
        default_left = np.asarray(tree["d"], bool)
        idx = np.zeros(n, np.int64)
        active = left[idx] != -1
        while active.any():
            cur = idx[active]
            value = X[row_ids[active], f[cur]]           # float32
            go_left = np.where(np.isnan(value), default_left[cur], value < t[cur])
            idx[active] = np.where(go_left, left[cur], right[cur])
            active = left[idx] != -1
        margin += t[idx].astype(np.float64)
    return 1.0 / (1.0 + np.exp(-margin))


def recompute_derived(vec: np.ndarray) -> np.ndarray:
    """JS recomputeDerived, for the what-if fixtures. Indices per FEATURE_ORDER."""
    out = vec.copy()
    minutes_left = max(out[2] / 60.0, 1 / 60.0)          # seconds_remaining_game
    out[11] = out[0] / minutes_left                      # margin_per_minute_remaining
    out[10] = 1.0 if (out[4] >= 4                        # period
                      and out[1] <= CLUTCH_SECONDS       # seconds_remaining_period
                      and abs(out[0]) <= CLUTCH_MARGIN) else 0.0
    return out


def main():
    model_obj = joblib.load(MODEL_PATH)
    booster = model_obj.get_booster()

    # The feature order is load-bearing. Refuse to export against a mismatch.
    meta = json.loads(METADATA_PATH.read_text())
    if meta.get("feature_order") != FEATURE_ORDER:
        raise SystemExit(
            "feature_order in model_metadata.json does not match FEATURE_ORDER "
            "in this script and web/lib/wp-model.js. Reconcile before exporting.")

    model = build_model(booster)
    print(f"base_score {model['base_score']:.6f}   trees {len(model['trees'])}")

    frame = pd.read_parquet(config.MODEL_FRAME_PARQUET)
    X = frame[FEATURE_ORDER].to_numpy(np.float32)
    reference = model_obj.predict_proba(X)[:, 1].astype(np.float64)
    ported = predict_all(model, X)
    diff = np.abs(reference - ported)
    over = int((diff > TOL).sum())
    print(f"VERIFY all rows n={len(X):,}  max|diff| {diff.max():.3e}  "
          f"mean {diff.mean():.3e}  over {TOL:g}: {over}")
    if diff.max() > TOL or over:
        raise SystemExit("export does NOT reproduce predict_proba; nothing written.")

    # ---- write the model to both locations --------------------------------
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(model, separators=(",", ":"))
    (DEPLOY_DIR / "model_trees.json").write_text(payload)
    (WEB_DATA_DIR / "model_trees.json").write_text(payload)

    # ---- regenerate the node-verifier fixtures ----------------------------
    np.ascontiguousarray(X, np.float32).tofile(DEPLOY_DIR / "_X.f32")
    np.ascontiguousarray(reference, np.float64).tofile(DEPLOY_DIR / "_p.f64")

    rng = np.random.default_rng(config.RANDOM_SEED)
    case_idx = rng.choice(len(X), size=2000, replace=False)
    overrides = rng.integers(-30, 31, size=2000)
    rows, prepared = [], []
    for k, i in enumerate(case_idx):
        raw = X[i].astype(np.float64).copy()
        raw[0] = float(overrides[k])                     # override celtics_margin
        rows.append({name: (None if np.isnan(raw[j]) else float(raw[j]))
                     for j, name in enumerate(FEATURE_ORDER)})
        # Mirror wp-model.js exactly: recomputeDerived runs in float64, and only
        # toVector casts to float32. Recomputing in float32 here would round a
        # derived value differently and, on a row sitting at a split threshold,
        # send it down the other branch.
        prepared.append(recompute_derived(raw).astype(np.float32))
    expected = model_obj.predict_proba(
        np.array(prepared, np.float32))[:, 1].astype(float).tolist()
    (DEPLOY_DIR / "_whatif_cases.json").write_text(
        json.dumps({"rows": rows, "expected": expected}))

    print()
    print("wrote:")
    print(f"  {WEB_DATA_DIR / 'model_trees.json'}   (the app loads this)")
    print(f"  {DEPLOY_DIR / 'model_trees.json'}")
    print(f"  {DEPLOY_DIR / '_X.f32'}, _p.f64, _whatif_cases.json")
    print()
    print("Cross-check in the real JavaScript with:")
    print("    node deploy/verify_js_model.mjs")


if __name__ == "__main__":
    main()
