"""
Tests for Phase 4: evaluation metrics and model tier definitions.

The metric tests use cases with a known right answer, because a metric that is
subtly wrong produces a plausible-looking result table that nobody catches.
"""

import numpy as np
import pandas as pd
import pytest

from src import evaluate, models


# ---------------------------------------------------------------------------
# Brier skill score
# ---------------------------------------------------------------------------

def test_perfect_predictions_have_skill_one():
    y = np.array([1, 1, 0, 0])
    p = np.array([1.0, 1.0, 0.0, 0.0])
    assert evaluate.brier_skill_score(y, p) == pytest.approx(1.0)


def test_predicting_the_base_rate_has_zero_skill():
    """This is the whole point of the metric: the base rate is the zero line."""
    y = np.array([1, 1, 1, 0])
    p = np.full(4, 0.75)
    assert evaluate.brier_skill_score(y, p) == pytest.approx(0.0, abs=1e-9)


def test_worse_than_the_base_rate_is_negative():
    y = np.array([1, 1, 1, 0])
    p = np.array([0.1, 0.1, 0.1, 0.9])
    assert evaluate.brier_skill_score(y, p) < 0


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def test_perfectly_calibrated_predictions_have_near_zero_ece():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 40000)
    y = (rng.uniform(size=40000) < p).astype(int)
    assert evaluate.expected_calibration_error(y, p) < 0.02


def test_systematically_overconfident_predictions_are_caught():
    """Claiming 90 percent while winning 50 percent must show a large ECE."""
    y = np.array([1, 0] * 500)
    p = np.full(1000, 0.9)
    assert evaluate.expected_calibration_error(y, p) == pytest.approx(0.4, abs=0.01)


def test_calibration_table_reports_gap_direction():
    y = np.array([1, 0] * 500)
    p = np.full(1000, 0.9)
    table = evaluate.calibration_table(y, p)
    assert len(table) == 1
    row = table.iloc[0]
    assert row["mean_predicted"] == pytest.approx(0.9)
    assert row["observed"] == pytest.approx(0.5)
    assert row["gap"] > 0            # positive gap means overconfident


def test_calibration_table_bins_sum_to_all_rows():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=5000)
    y = (rng.uniform(size=5000) < p).astype(int)
    table = evaluate.calibration_table(y, p)
    assert int(table["n"].sum()) == 5000
    assert table["share"].sum() == pytest.approx(1.0)


def test_max_calibration_error_is_at_least_the_mean():
    rng = np.random.default_rng(2)
    p = rng.uniform(size=3000)
    y = (rng.uniform(size=3000) < p * 0.5).astype(int)
    assert (evaluate.max_calibration_error(y, p)
            >= evaluate.expected_calibration_error(y, p))


# ---------------------------------------------------------------------------
# score_all
# ---------------------------------------------------------------------------

def test_score_all_reports_the_baseline_alongside_the_score():
    """A Brier score without its baseline is uninterpretable."""
    y = np.array([1] * 65 + [0] * 35)
    p = np.full(100, 0.65)
    result = evaluate.score_all(y, p)
    assert result["baseline_brier"] == pytest.approx(0.65 * 0.35)
    assert result["brier"] == pytest.approx(result["baseline_brier"])
    assert result["brier_skill"] == pytest.approx(0.0, abs=1e-9)
    assert "baseline_logloss" in result


def test_score_all_handles_a_single_class_without_crashing():
    y = np.ones(50, dtype=int)
    p = np.full(50, 0.8)
    result = evaluate.score_all(y, p)
    assert np.isnan(result["auc"])
    assert result["n"] == 50


# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------

def test_tiers_are_nested_so_the_comparison_isolates_each_addition():
    keys = [t["key"] for t in models.TIERS]
    assert keys == ["tier1_pregame", "tier2_generic",
                    "tier3_celtics", "tier4_lineup"]
    game_state = set(models.GAME_STATE_FEATURES)
    with_lineup = set(models.LINEUP_FEATURES)
    assert game_state.issubset(with_lineup)
    assert len(with_lineup) == len(game_state) + 3


def test_generic_baseline_uses_only_margin_and_time():
    """
    Tier 2 must stay minimal. Adding features would stop it being a baseline,
    and it is the stand-in for a generic public model.
    """
    assert models.GENERIC_FEATURES == ["celtics_margin", "seconds_remaining_game"]


def test_pregame_tier_has_no_in_game_information():
    banned = {"celtics_margin", "seconds_remaining_game", "momentum_120s",
              "celtics_has_possession", "period", "is_clutch"}
    assert not set(models.PREGAME_FEATURES) & banned


def test_generic_interaction_adds_an_urgency_term():
    """A five-point lead means different things at 2 minutes and 30 minutes."""
    x = np.array([[5.0, 120.0], [5.0, 1800.0]])
    out = models.add_generic_interaction(x)
    assert out.shape == (2, 3)
    assert out[0, 2] > out[1, 2]


def test_generic_interaction_survives_zero_seconds():
    x = np.array([[5.0, 0.0]])
    out = models.add_generic_interaction(x)
    assert np.isfinite(out).all()


def test_no_tier_name_claims_to_be_espn():
    """
    ESPN's model is not published, so no tier may be labelled as it. This is a
    representation issue in a graded paper, not a style preference.
    """
    for tier in models.TIERS:
        assert "espn" not in tier["name"].lower()
        assert "espn" not in tier["key"].lower()


def test_only_the_lineup_tier_declares_it_needs_lineup_features():
    needs = [t["key"] for t in models.TIERS if t["needs_lineup"]]
    assert needs == ["tier4_lineup"]


# ---------------------------------------------------------------------------
# Phase table
# ---------------------------------------------------------------------------

def test_phase_table_skips_phases_with_too_few_events():
    frame = pd.DataFrame({
        "period": [1] * 500 + [5] * 3,
        "seconds_elapsed_game": [10.0] * 503,
        "seconds_remaining_game": [2800.0] * 503,
        "is_clutch": [False] * 503,
    })
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 503)
    p = rng.uniform(size=503)
    table = evaluate.phase_table(frame, y, p)
    assert "overtime" not in set(table["phase"])
    assert "period 1" in set(table["phase"])


# ---------------------------------------------------------------------------
# Cluster bootstrap. Resampling the wrong unit is the classic way to get an
# interval that is far too narrow and then over-claim a result.
# ---------------------------------------------------------------------------

def test_bootstrap_detects_a_genuinely_better_model():
    rng = np.random.default_rng(0)
    n_games, per_game = 400, 300
    game_ids, y, good, bad = [], [], [], []
    for g in range(n_games):
        outcome = int(rng.integers(0, 2))
        game_ids += [f"g{g}"] * per_game
        y += [outcome] * per_game
        good += list(np.clip(outcome * 0.75 + 0.12
                             + rng.normal(0, 0.05, per_game), 0.01, 0.99))
        bad += list(rng.uniform(0.2, 0.8, per_game))
    result = evaluate.bootstrap_brier_difference(
        game_ids, y, bad, good, n_boot=300, seed=1)
    assert result["observed_difference"] > 0     # second model better
    assert result["excludes_zero"]
    assert result["n_games"] == n_games


def test_bootstrap_reports_no_difference_for_identical_models():
    rng = np.random.default_rng(2)
    game_ids = [f"g{i // 100}" for i in range(20000)]
    y = rng.integers(0, 2, 20000)
    p = rng.uniform(0.2, 0.8, 20000)
    result = evaluate.bootstrap_brier_difference(
        game_ids, y, p, p, n_boot=200, seed=3)
    assert result["observed_difference"] == pytest.approx(0.0, abs=1e-12)
    assert not result["excludes_zero"]


def _clustered_fixture(n_games, rows_per_game, seed):
    """Games with one shared outcome and heterogeneous prediction quality."""
    rng = np.random.default_rng(seed)
    game_ids, y, a, b = [], [], [], []
    for g in range(n_games):
        outcome = int(rng.integers(0, 2))
        skill = rng.uniform(0.0, 0.45)          # how good model A is this game
        for _ in range(rows_per_game):
            game_ids.append(f"g{g}")
            y.append(outcome)
            a.append(np.clip(0.5 + skill * (1 if outcome else -1)
                             + rng.normal(0, 0.05), 0.01, 0.99))
            b.append(float(rng.uniform(0.3, 0.7)))
    return game_ids, np.array(y), np.array(a), np.array(b)


def test_bootstrap_width_is_driven_by_games_not_rows():
    """
    THE decisive property. Hold the number of games fixed and multiply the rows
    per game tenfold. If the code resampled EVENTS the interval would shrink by
    roughly the square root of ten; because it resamples GAMES it should barely
    move.

    Getting this wrong is how a 0.6 percent difference gets reported as
    conclusive when it is not.
    """
    thin = evaluate.bootstrap_brier_difference(
        *_clustered_fixture(60, 30, seed=11), n_boot=600, seed=1)
    thick = evaluate.bootstrap_brier_difference(
        *_clustered_fixture(60, 300, seed=11), n_boot=600, seed=1)

    thin_width = thin["ci_high"] - thin["ci_low"]
    thick_width = thick["ci_high"] - thick["ci_low"]
    assert thin["n_games"] == thick["n_games"] == 60
    # Ten times the rows must NOT meaningfully tighten the interval.
    assert thick_width > 0.7 * thin_width


def test_bootstrap_width_shrinks_when_there_are_more_games():
    """More independent games is the only thing that should tighten it."""
    few = evaluate.bootstrap_brier_difference(
        *_clustered_fixture(25, 100, seed=12), n_boot=600, seed=2)
    many = evaluate.bootstrap_brier_difference(
        *_clustered_fixture(400, 100, seed=12), n_boot=600, seed=2)
    assert (few["ci_high"] - few["ci_low"]) > (many["ci_high"] - many["ci_low"])


def test_lineup_variants_are_pre_registered_and_distinct():
    """Three fixed alternatives, each changing exactly one thing."""
    keys = [v["key"] for v in models.LINEUP_VARIANTS]
    assert keys == ["variant_a_diff_only", "variant_b_heavy_shrinkage",
                    "variant_c_time_interaction"]
    assert len({v["key"] for v in models.LINEUP_VARIANTS}) == 3
    shrinkages = {v["shrinkage_minutes"] for v in models.LINEUP_VARIANTS}
    assert 2000.0 in shrinkages          # variant B changes shrinkage
    a = models.LINEUP_VARIANTS[0]
    assert a["features"].count("lineup_strength_diff") == 1
    assert "celtics_lineup_strength" not in a["features"]


def test_lineup_time_interaction_decays_as_the_clock_runs_down():
    frame = pd.DataFrame({
        "lineup_strength_diff": [0.10, 0.10],
        "seconds_remaining_game": [2880.0, 60.0],
    })
    out = models.add_lineup_interactions(frame)
    assert out["lineup_diff_x_time"].iloc[0] > out["lineup_diff_x_time"].iloc[1]
    assert out["lineup_diff_x_time"].iloc[0] == pytest.approx(0.10)
