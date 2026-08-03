"""
Tests for Phase 7: the dose-response ladders, the clean lineup re-test, and the
deliverable switching rule.

The ladders only prove something if they are genuinely nested and genuinely
matched in resolution. The switching rule only means something if it can refuse
to switch. Both are tested here.
"""

import numpy as np
import pandas as pd
import pytest

from src import clean_tests, models


def toy_frame(n_games=200, events_per_game=10, seed=2):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_games):
        game_id = f"00216{g:05d}"
        won = int(rng.integers(0, 2))
        diff = float(rng.uniform(-9.0, 10.0))
        for e in range(events_per_game):
            rows.append({
                "game_id": game_id,
                "event_index": e,
                "celtics_won": won,
                "opponent_point_diff_prior": diff,
                "strength_diff_prior": -diff,
                "lineup_strength_diff": float(rng.normal()),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The opponent ladder
# ---------------------------------------------------------------------------

def test_opponent_ladder_reduces_cardinality_monotonically():
    """Coarser rungs must have fewer distinct values, or it is not a ladder."""
    frame = clean_tests.add_opponent_ladder(toy_frame())
    per_game = frame.groupby("game_id").first()
    counts = [per_game[name].nunique() for name, _s, _l
              in clean_tests.OPPONENT_LADDER]
    counts.append(per_game[clean_tests.OPPONENT_SOURCE].nunique())
    assert counts == sorted(counts)
    assert counts[0] < counts[-1]


def test_opponent_ladder_is_constant_within_a_game():
    frame = clean_tests.add_opponent_ladder(toy_frame())
    for name, _step, _label in clean_tests.OPPONENT_LADDER:
        assert (frame.groupby("game_id")[name].nunique() == 1).all()


def test_opponent_ladder_preserves_ordering():
    """Rounding may not scramble which opponent is stronger."""
    frame = clean_tests.add_opponent_ladder(toy_frame())
    per_game = frame.groupby("game_id").first().sort_values(
        clean_tests.OPPONENT_SOURCE)
    for name, _step, _label in clean_tests.OPPONENT_LADDER:
        assert np.all(np.diff(per_game[name].to_numpy()) >= 0)


def test_opponent_ladder_uses_no_data_dependent_parameter():
    """
    Rounding by a fixed step gives the same edges whatever the data does, so no
    part of the held-out season can influence the binning.
    """
    small = clean_tests.add_opponent_ladder(toy_frame(n_games=20, seed=4))
    large = clean_tests.add_opponent_ladder(toy_frame(n_games=400, seed=4))
    for name, step, _label in clean_tests.OPPONENT_LADDER:
        for frame in (small, large):
            values = frame[name].to_numpy()
            assert np.allclose(values / step, np.round(values / step))


def test_opponent_ladder_raises_without_the_source_column():
    with pytest.raises(KeyError, match="opponent_point_diff_prior"):
        clean_tests.add_opponent_ladder(pd.DataFrame({"game_id": ["a"]}))


# ---------------------------------------------------------------------------
# The random ladder
# ---------------------------------------------------------------------------

def test_random_ladder_hits_its_target_cardinalities():
    """
    Bounded above by construction, and close to the target in practice.

    An exact count is the wrong assertion: with 500 games and 100 bins a bin can
    come up empty by chance. The property that matters is that the rung cannot
    EXCEED its intended resolution, which is what makes the ladder a ladder.
    """
    frame = clean_tests.add_random_ladder(toy_frame(n_games=500))
    per_game = frame.groupby("game_id").first()
    for name, k in clean_tests.RANDOM_LADDER:
        achieved = per_game[name].nunique()
        assert achieved <= k
        assert achieved >= 0.9 * k
        values = per_game[name].to_numpy()
        assert values.min() >= 0 and values.max() < k


def test_random_ladder_is_nested():
    """
    Every rung must be a coarsening of the SAME draw. Independent draws at each
    rung would compare different random variables, not different resolutions.
    """
    frame = clean_tests.add_random_ladder(toy_frame(n_games=300))
    per_game = frame.groupby("game_id").first()
    fine = per_game["rand_bins100"].to_numpy()
    coarse = per_game["rand_bins5"].to_numpy()
    # A finer bin index must determine the coarser one.
    mapping = {}
    for f, c in zip(fine, coarse):
        if f in mapping:
            assert mapping[f] == c
        mapping[f] = c


def test_random_ladder_raw_is_near_unique():
    frame = clean_tests.add_random_ladder(toy_frame(n_games=250))
    per_game = frame.groupby("game_id").first()
    assert per_game[clean_tests.RANDOM_RAW].nunique() == 250


def test_random_ladder_is_constant_within_a_game():
    frame = clean_tests.add_random_ladder(toy_frame())
    columns = [clean_tests.RANDOM_RAW] + [n for n, _k
                                          in clean_tests.RANDOM_LADDER]
    for column in columns:
        assert (frame.groupby("game_id")[column].nunique() == 1).all()


def test_random_ladder_carries_no_signal():
    frame = clean_tests.add_random_ladder(toy_frame(n_games=400, seed=11))
    per_game = frame.groupby("game_id").first()
    correlation = np.corrcoef(per_game[clean_tests.RANDOM_RAW],
                              per_game["celtics_won"])[0, 1]
    assert abs(correlation) < 0.15


def test_random_ladder_is_reproducible():
    a = clean_tests.add_random_ladder(toy_frame(), seed=42)
    b = clean_tests.add_random_ladder(toy_frame(), seed=42)
    assert np.array_equal(a[clean_tests.RANDOM_RAW], b[clean_tests.RANDOM_RAW])


def test_the_two_ladders_have_comparable_resolutions():
    """
    The dose-response argument compares rung to rung. If the rungs were at very
    different cardinalities the comparison would be meaningless.
    """
    frame = clean_tests.add_random_ladder(
        clean_tests.add_opponent_ladder(toy_frame(n_games=636, seed=6)))
    per_game = frame.groupby("game_id").first()
    for (opp_name, _s, _l), (rand_name, _k) in zip(clean_tests.OPPONENT_LADDER,
                                                   clean_tests.RANDOM_LADDER):
        opp = per_game[opp_name].nunique()
        rand = per_game[rand_name].nunique()
        assert 0.2 <= opp / rand <= 5.0, (
            f"{opp_name} has {opp} values but {rand_name} has {rand}")


# ---------------------------------------------------------------------------
# Fold-safe lineup binning
# ---------------------------------------------------------------------------

def test_lineup_bins_come_only_from_training_values():
    """
    The held-out season must not influence where the cut points sit. Changing
    the test values alone must not change the training bins.
    """
    train = np.linspace(-1.0, 1.0, 500)
    test_a = np.linspace(-1.0, 1.0, 100)
    test_b = np.linspace(-50.0, 50.0, 100)
    bins_a, _ = clean_tests.bin_by_training_quantiles(train, test_a)
    bins_b, _ = clean_tests.bin_by_training_quantiles(train, test_b)
    assert np.array_equal(bins_a, bins_b)


def test_lineup_bins_produce_the_requested_count():
    train = np.random.default_rng(0).normal(size=2000)
    bins, _ = clean_tests.bin_by_training_quantiles(train, train[:10])
    assert len(np.unique(bins)) == clean_tests.LINEUP_BINS


def test_lineup_bins_are_roughly_equal_sized():
    train = np.random.default_rng(1).normal(size=5000)
    bins, _ = clean_tests.bin_by_training_quantiles(train, train[:10])
    counts = np.bincount(bins.astype(int))
    assert counts.min() > 0.8 * counts.max()


def test_lineup_bins_handle_a_degenerate_feature():
    """A constant feature must not raise; it simply cannot be split."""
    train = np.zeros(100)
    bins, test_bins = clean_tests.bin_by_training_quantiles(train, np.zeros(10))
    assert len(np.unique(bins)) >= 1
    assert len(test_bins) == 10


def test_lineup_bins_collapse_cardinality():
    values = np.random.default_rng(3).normal(size=1000)
    bins, _ = clean_tests.bin_by_training_quantiles(values, values[:5])
    assert len(np.unique(bins)) < len(np.unique(values))


# ---------------------------------------------------------------------------
# The specifications
# ---------------------------------------------------------------------------

def test_feature_comparisons_never_confound_the_model_with_the_feature():
    for base_key, candidate_key in clean_tests.FEATURE_COMPARISONS:
        base = clean_tests.SPEC_BY_KEY[base_key]
        candidate = clean_tests.SPEC_BY_KEY[candidate_key]
        assert base["factory"] is candidate["factory"], (
            f"{base_key} and {candidate_key} use different models, so their "
            "difference would not isolate the feature")


def test_deliverable_comparisons_are_deliberately_cross_family():
    """
    These DO span model families, on purpose, because the question is what to
    ship rather than what a feature is worth. Keeping them in a separate list is
    what stops that being an accident.
    """
    for base_key, candidate_key in clean_tests.DELIVERABLE_COMPARISONS:
        base = clean_tests.SPEC_BY_KEY[base_key]
        candidate = clean_tests.SPEC_BY_KEY[candidate_key]
        assert base["factory"] is not candidate["factory"]
    overlap = (set(clean_tests.DELIVERABLE_COMPARISONS)
               & set(clean_tests.FEATURE_COMPARISONS))
    assert not overlap


def test_every_spec_has_a_written_prediction():
    for spec in clean_tests.CLEAN_SPECS:
        assert spec["prediction"].strip()


def test_spec_keys_are_unique():
    keys = [s["key"] for s in clean_tests.CLEAN_SPECS]
    assert len(keys) == len(set(keys))


def test_references_match_the_published_tiers():
    assert (clean_tests.SPEC_BY_KEY["p7_tier3"]["features"]
            == list(models.GAME_STATE_FEATURES))
    assert (clean_tests.SPEC_BY_KEY["p7_tier2"]["features"]
            == list(models.GENERIC_FEATURES))


def test_the_linear_null_control_exists_and_is_compared():
    """Without this control the linear opponent result cannot be trusted."""
    assert "p7_linear_random" in clean_tests.SPEC_BY_KEY
    assert ("p7_tier2", "p7_linear_random") in clean_tests.FEATURE_COMPARISONS


def test_both_ladders_are_present_at_matched_rungs():
    for _label, opp_key, rand_key in clean_tests.LADDER_PAIRS:
        assert opp_key in clean_tests.SPEC_BY_KEY
        assert rand_key in clean_tests.SPEC_BY_KEY


# ---------------------------------------------------------------------------
# The switching rule
# ---------------------------------------------------------------------------

def result(difference, low, high):
    return {"observed_difference": difference, "ci_low": low, "ci_high": high,
            "excludes_zero": bool(low > 0 or high < 0)}


def test_rule_switches_only_on_a_decisive_win():
    text = clean_tests.deliverable_verdict(
        result(0.0040, 0.0010, 0.0075), "challenger", "tier 3")
    assert text.startswith("SWITCH")


def test_rule_refuses_to_switch_on_a_point_estimate():
    """
    The interval spans zero, so the models are not distinguishable. This is the
    case the rule exists for.
    """
    text = clean_tests.deliverable_verdict(
        result(0.0040, -0.0005, 0.0090), "challenger", "tier 3")
    assert text.startswith("KEEP")


def test_rule_refuses_to_switch_when_the_challenger_is_behind():
    text = clean_tests.deliverable_verdict(
        result(-0.0040, -0.0090, 0.0005), "challenger", "tier 3")
    assert text.startswith("KEEP")


def test_rule_is_the_same_standard_that_rejected_tier_4():
    """
    Tier 3 vs tier 4 measured -0.0193 with [-0.0252, -0.0133]: decisive, and
    against the challenger. The rule must keep the incumbent there too.
    """
    text = clean_tests.deliverable_verdict(
        result(-0.0193, -0.0252, -0.0133), "tier 4", "tier 3")
    assert text.startswith("KEEP")


# ---------------------------------------------------------------------------
# The dose-response reading
# ---------------------------------------------------------------------------

def test_dose_response_confirmed_when_both_ladders_rise_together():
    text = clean_tests.dose_response_verdict(
        opponent_gaps=[0.000, 0.010, 0.025, 0.037],
        random_gaps=[0.001, 0.009, 0.022, 0.041])
    assert "CONFIRMED, AND MATCHED" in text


def test_dose_response_rejected_when_damage_does_not_grow():
    text = clean_tests.dose_response_verdict(
        opponent_gaps=[0.030, 0.030, 0.029, 0.028],
        random_gaps=[0.001, 0.001, 0.000, 0.000])
    assert "NOT FOUND" in text


def test_dose_response_partial_when_the_ladders_diverge():
    text = clean_tests.dose_response_verdict(
        opponent_gaps=[0.000, 0.020, 0.050, 0.090],
        random_gaps=[0.000, 0.002, 0.005, 0.010])
    assert "PARTIALLY MATCHED" in text


def test_dose_response_needs_the_random_ladder_to_rise_too():
    """
    A real feature getting worse with resolution, while a random one does not,
    would point at overfitting a genuine signal rather than at memorisation.
    """
    text = clean_tests.dose_response_verdict(
        opponent_gaps=[0.000, 0.020, 0.050, 0.090],
        random_gaps=[0.000, 0.000, 0.000, 0.000])
    assert "NOT FOUND" in text


# ---------------------------------------------------------------------------
# End-to-end smoke
# ---------------------------------------------------------------------------

def test_every_clean_spec_fits_and_predicts():
    frame = clean_tests.add_random_ladder(
        clean_tests.add_opponent_ladder(toy_frame(n_games=60,
                                                  events_per_game=20)))
    rng = np.random.default_rng(8)
    for column in models.GAME_STATE_FEATURES:
        if column not in frame:
            frame[column] = rng.normal(size=len(frame))
    frame[clean_tests.LINEUP_BINNED] = rng.integers(0, 5, len(frame)).astype(float)

    target = frame["celtics_won"].astype(int)
    train, test = frame.iloc[:800], frame.iloc[800:]
    for spec in clean_tests.CLEAN_SPECS:
        probabilities, model = models.fit_predict(spec, train,
                                                  target.iloc[:800], test)
        assert len(probabilities) == len(test)
        x_train = train[spec["features"]].astype(float).to_numpy()
        if spec["transform"] is not None:
            x_train = spec["transform"](x_train)
        assert len(model.predict_proba(x_train)[:, 1]) == len(train)
