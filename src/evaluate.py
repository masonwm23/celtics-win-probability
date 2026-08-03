"""
Phase 4: evaluation, including calibration.

A win probability model is judged on two different things, and reporting only the
first is a common and serious mistake.

DISCRIMINATION asks whether the model ranks winning situations above losing ones.
AUC measures this.

CALIBRATION asks whether the numbers mean what they say. When the model says 70
percent, does Boston win about 70 percent of those situations? A model can rank
perfectly and still be badly calibrated, and for this project calibration is the
more important property: the deliverable is a live probability shown on a
dashboard, and a number that reads 70 percent while meaning 90 is worse than
useless to someone watching.

Brier score and log loss capture both properties at once. Expected calibration
error isolates calibration alone.

Everything here also reports the BASELINE, meaning the score you would get by
always predicting the base rate. A Brier of 0.17 is meaningless in isolation; a
Brier of 0.17 against a base rate that scores 0.25 is a 31 percent improvement.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

DEFAULT_BINS = 10

# Game phases used for the breakdown. Late-game performance is easy and early
# performance is hard, so a single pooled number hides the interesting part.
PHASES = [
    ("first 6 min", lambda f: f["seconds_elapsed_game"].le(360)),
    ("period 1", lambda f: f["period"].eq(1)),
    ("period 2", lambda f: f["period"].eq(2)),
    ("period 3", lambda f: f["period"].eq(3)),
    ("period 4", lambda f: f["period"].eq(4)),
    ("overtime", lambda f: f["period"].gt(4)),
    ("clutch", lambda f: f["is_clutch"]),
    ("last 2 min", lambda f: f["seconds_remaining_game"].le(120)),
]


def brier_skill_score(y_true, probabilities):
    """
    Improvement in Brier score over always predicting the base rate.

    1.0 is perfect, 0.0 is no better than the base rate, negative is worse than
    doing nothing. This is the honest way to report a Brier score, because the
    raw value depends on how lopsided the outcomes are.
    """
    y_true = np.asarray(y_true, dtype=float)
    base_rate = y_true.mean()
    baseline = base_rate * (1 - base_rate)
    if baseline <= 0:
        return float("nan")
    return float(1 - brier_score_loss(y_true, probabilities) / baseline)


def expected_calibration_error(y_true, probabilities, bins=DEFAULT_BINS):
    """
    Mean absolute gap between predicted probability and observed frequency,
    weighted by how many predictions fall in each bin.

    0.02 means that on average the stated probability is off by two points.
    """
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(probabilities, edges[1:-1], right=False),
                    0, bins - 1)
    total, error = len(y_true), 0.0
    for b in range(bins):
        mask = index == b
        count = int(mask.sum())
        if count == 0:
            continue
        error += (count / total) * abs(probabilities[mask].mean()
                                       - y_true[mask].mean())
    return float(error)


def max_calibration_error(y_true, probabilities, bins=DEFAULT_BINS):
    """The worst bin, which is what a user would actually notice."""
    table = calibration_table(y_true, probabilities, bins)
    if table.empty:
        return float("nan")
    return float((table["mean_predicted"] - table["observed"]).abs().max())


def calibration_table(y_true, probabilities, bins=DEFAULT_BINS) -> pd.DataFrame:
    """Reliability table: predicted versus observed frequency, by bin."""
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(probabilities, edges[1:-1], right=False),
                    0, bins - 1)
    rows = []
    for b in range(bins):
        mask = index == b
        count = int(mask.sum())
        if count == 0:
            continue
        rows.append({
            "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
            "n": count,
            "share": count / len(y_true),
            "mean_predicted": float(probabilities[mask].mean()),
            "observed": float(y_true[mask].mean()),
            "gap": float(probabilities[mask].mean() - y_true[mask].mean()),
        })
    return pd.DataFrame(rows)


def score_all(y_true, probabilities, bins=DEFAULT_BINS) -> dict:
    """Every metric for one set of predictions."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    base_rate = float(y_true.mean())
    out = {
        "n": int(len(y_true)),
        "base_rate": base_rate,
        "brier": float(brier_score_loss(y_true, probabilities)),
        "baseline_brier": float(base_rate * (1 - base_rate)),
        "brier_skill": brier_skill_score(y_true, probabilities),
        "ece": expected_calibration_error(y_true, probabilities, bins),
        "max_ce": max_calibration_error(y_true, probabilities, bins),
    }
    if len(np.unique(y_true)) > 1:
        out["auc"] = float(roc_auc_score(y_true, probabilities))
        out["logloss"] = float(log_loss(y_true, probabilities, labels=[0, 1]))
        base = np.clip(base_rate, 1e-9, 1 - 1e-9)
        out["baseline_logloss"] = float(
            -(base_rate * np.log(base) + (1 - base_rate) * np.log(1 - base)))
    else:
        out["auc"] = float("nan")
        out["logloss"] = float("nan")
        out["baseline_logloss"] = float("nan")
    return out


def phase_table(frame: pd.DataFrame, y_true, probabilities,
                bins=DEFAULT_BINS) -> pd.DataFrame:
    """Metrics broken out by game phase."""
    y_true = pd.Series(np.asarray(y_true, dtype=int), index=frame.index)
    probabilities = pd.Series(np.asarray(probabilities, dtype=float),
                              index=frame.index)
    rows = []
    for label, predicate in PHASES:
        mask = predicate(frame) & probabilities.notna()
        if int(mask.sum()) < 100:
            continue
        result = score_all(y_true[mask], probabilities[mask], bins)
        result["phase"] = label
        rows.append(result)
    return pd.DataFrame(rows)


def compare_tiers(results: dict, y_true) -> pd.DataFrame:
    """
    One row per tier, pooled over all out-of-fold predictions.

    `results` maps a tier key to its out-of-fold probability Series.
    """
    rows = []
    for key, probabilities in results.items():
        mask = pd.Series(probabilities).notna().to_numpy()
        result = score_all(np.asarray(y_true)[mask],
                           np.asarray(probabilities)[mask])
        result["tier"] = key
        rows.append(result)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cluster bootstrap
# ---------------------------------------------------------------------------

def _per_game_brier_parts(game_ids, y_true, probabilities):
    """Squared-error sum and row count per game, for fast resampling."""
    frame = pd.DataFrame({
        "game_id": np.asarray(game_ids),
        "sse": (np.asarray(probabilities, dtype=float)
                - np.asarray(y_true, dtype=float)) ** 2,
    })
    grouped = frame.groupby("game_id")["sse"].agg(["sum", "count"])
    return grouped["sum"].to_numpy(), grouped["count"].to_numpy()


def bootstrap_brier_difference(game_ids, y_true, probabilities_a,
                               probabilities_b, n_boot=2000, seed=None,
                               alpha=0.05):
    """
    Confidence interval on the Brier difference between two models.

    RESAMPLES GAMES, NOT EVENTS. Events within a game share an outcome and are
    highly dependent, so resampling events would treat 309,000 correlated rows as
    309,000 independent ones and produce an interval far too narrow. The correct
    unit is the game, and there are 636 of them.

    Returns a dict with the observed difference (a minus b, so positive means b
    is better), the bootstrap interval, and whether it excludes zero.

    Implementation note: the per-game squared-error sums are precomputed once, so
    each resample is O(number of games) rather than O(number of events).
    """
    rng = np.random.default_rng(seed)
    sse_a, n_a = _per_game_brier_parts(game_ids, y_true, probabilities_a)
    sse_b, n_b = _per_game_brier_parts(game_ids, y_true, probabilities_b)
    n_games = len(sse_a)

    observed = (sse_a.sum() / n_a.sum()) - (sse_b.sum() / n_b.sum())

    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        pick = rng.integers(0, n_games, n_games)
        draws[i] = ((sse_a[pick].sum() / n_a[pick].sum())
                    - (sse_b[pick].sum() / n_b[pick].sum()))

    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {
        "observed_difference": float(observed),
        "ci_low": float(low),
        "ci_high": float(high),
        "excludes_zero": bool(low > 0 or high < 0),
        "n_games": int(n_games),
        "n_boot": int(n_boot),
    }
