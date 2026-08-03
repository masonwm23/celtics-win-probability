"""
Tests for Phase 6: the memorisation controls.

The point of a control is that it is genuinely uninformative. If the "random"
column leaked any real signal, or if the bucketed feature still identified
individual games, the control would prove nothing. These tests check exactly
those properties, plus the arithmetic of the verdict rule.
"""

import numpy as np
import pandas as pd
import pytest

from src import controls, models, opponent_strength


def toy_frame(n_games=40, events_per_game=50, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_games):
        game_id = f"00216{g:05d}"
        won = int(rng.integers(0, 2))
        for e in range(events_per_game):
            rows.append({
                "game_id": game_id,
                "event_index": e,
                "celtics_won": won,
                "opponent_point_diff_prior": float(g) - 20.0,
                "strength_diff_prior": 20.0 - float(g),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The random control must be constant within a game and carry no signal
# ---------------------------------------------------------------------------

def test_random_constant_is_constant_within_each_game():
    """
    If it varied within a game it would not be the same shape as the opponent
    feature, and would not reproduce the mechanism under test.
    """
    frame = controls.add_random_game_constants(toy_frame())
    per_game = frame.groupby("game_id")["random_game_constant"].nunique()
    assert (per_game == 1).all()


def test_random_constant_is_near_unique_across_games():
    """It has to be able to identify a game, exactly as the real feature can."""
    frame = controls.add_random_game_constants(toy_frame(n_games=100))
    distinct = frame.groupby("game_id")["random_game_constant"].first().nunique()
    assert distinct == 100


def test_random_bucket_cannot_identify_a_game():
    """The coarse control must take few values, or it is not a coarse control."""
    frame = controls.add_random_game_constants(toy_frame(n_games=100))
    assert frame["random_game_bucket"].nunique() <= 5


def test_random_bucket_is_also_constant_within_a_game():
    frame = controls.add_random_game_constants(toy_frame())
    per_game = frame.groupby("game_id")["random_game_bucket"].nunique()
    assert (per_game == 1).all()


def test_random_constant_is_uncorrelated_with_the_outcome():
    """
    A control that accidentally predicted the target would not be a control.
    Across many games the correlation must be indistinguishable from zero.
    """
    frame = controls.add_random_game_constants(toy_frame(n_games=300, seed=7))
    per_game = frame.groupby("game_id").first()
    correlation = np.corrcoef(per_game["random_game_constant"],
                              per_game["celtics_won"])[0, 1]
    assert abs(correlation) < 0.2


def test_random_constant_is_reproducible_for_a_fixed_seed():
    a = controls.add_random_game_constants(toy_frame(), seed=99)
    b = controls.add_random_game_constants(toy_frame(), seed=99)
    assert np.array_equal(a["random_game_constant"], b["random_game_constant"])


def test_random_constant_differs_for_a_different_seed():
    a = controls.add_random_game_constants(toy_frame(), seed=1)
    b = controls.add_random_game_constants(toy_frame(), seed=2)
    assert not np.array_equal(a["random_game_constant"],
                              b["random_game_constant"])


def test_random_constant_attaches_to_every_row():
    frame = controls.add_random_game_constants(toy_frame())
    assert frame[controls.RANDOM_FEATURES].notna().all().all()
    assert len(frame) == len(toy_frame())


# ---------------------------------------------------------------------------
# The bucketed opponent feature must lose its ability to identify a game
# ---------------------------------------------------------------------------

def test_bucketing_collapses_cardinality():
    """
    The whole point. The raw feature can name a game; the bucketed one cannot.
    """
    frame = toy_frame(n_games=100)
    raw = frame.groupby("game_id")["opponent_point_diff_prior"].first().nunique()
    bucketed = controls.add_opponent_buckets(frame)
    coarse = bucketed.groupby("game_id")["opponent_quality_bucket"].first().nunique()
    assert raw == 100
    assert coarse <= 5
    assert coarse < raw


def test_bucketing_preserves_the_ordering_of_opponent_quality():
    """Coarse, but not scrambled: a stronger opponent must not get a lower bucket."""
    frame = toy_frame(n_games=60)
    bucketed = controls.add_opponent_buckets(frame)
    per_game = bucketed.groupby("game_id").first().sort_values(
        "opponent_point_diff_prior")
    buckets = per_game["opponent_quality_bucket"].to_numpy()
    assert np.all(np.diff(buckets) >= 0)


def test_bucket_edges_produce_five_buckets():
    values = pd.DataFrame({
        "game_id": [f"g{i}" for i in range(5)],
        "opponent_point_diff_prior": [-9.0, -3.0, 0.0, 3.0, 9.0],
        "strength_diff_prior": [-9.0, -3.0, 0.0, 3.0, 9.0],
    })
    out = controls.add_opponent_buckets(values)
    assert list(out["opponent_quality_bucket"]) == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_bucketing_raises_when_opponent_features_are_absent():
    with pytest.raises(KeyError, match="opponent features absent"):
        controls.add_opponent_buckets(pd.DataFrame({"game_id": ["a"]}))


# ---------------------------------------------------------------------------
# The game-aware floor
# ---------------------------------------------------------------------------

def test_game_aware_floor_exceeds_the_events_in_one_game():
    """
    486 events per game is the number that makes min_child_weight = 20 useless.
    The floor must be far above one game, or the control does not control.
    """
    events_per_game = 486
    assert controls.GAME_AWARE_MIN_CHILD_WEIGHT > events_per_game * 4


def test_game_aware_factory_actually_sets_the_parameter():
    model = controls.make_gb_game_aware()
    assert model.min_child_weight == controls.GAME_AWARE_MIN_CHILD_WEIGHT


def test_default_factory_is_unchanged():
    """Phase 4 numbers must remain reproducible, so the default cannot move."""
    assert models.DEFAULT_MIN_CHILD_WEIGHT == 20
    assert models.make_gradient_boosting().min_child_weight == 20


# ---------------------------------------------------------------------------
# The specs themselves
# ---------------------------------------------------------------------------

def test_every_spec_has_a_prediction_written_in_advance():
    """
    Pre-registration is the difference between a control and a fishing trip.
    """
    for spec in controls.CONTROL_SPECS:
        assert spec["prediction"].strip()


def test_every_spec_has_resolved_features():
    for spec in controls.CONTROL_SPECS:
        assert spec["features"], f"{spec['key']} has no features"
        assert isinstance(spec["features"], list)


def test_spec_keys_are_unique():
    keys = [s["key"] for s in controls.CONTROL_SPECS]
    assert len(keys) == len(set(keys))


def test_the_fair_opponent_test_compares_like_with_like():
    """
    C4 must be compared against C5, not against tier 3. Both carry the same
    min_child_weight, so the difference between them isolates opponent context
    rather than the parameter change.
    """
    c4 = controls.SPEC_BY_KEY["c4_opponent_game_aware"]
    c5 = controls.SPEC_BY_KEY["c5_tier3_game_aware"]
    assert c4["factory"] is c5["factory"]
    assert ("c5_tier3_game_aware", "c4_opponent_game_aware") \
        in controls.CONTROL_COMPARISONS


def test_no_control_is_compared_against_a_different_configuration():
    """Every pre-registered pair must share a model factory."""
    for base_key, candidate_key in controls.CONTROL_COMPARISONS:
        base = controls.SPEC_BY_KEY[base_key]
        candidate = controls.SPEC_BY_KEY[candidate_key]
        assert base["factory"] is candidate["factory"], (
            f"{base_key} and {candidate_key} use different model settings, so "
            "their difference would not isolate the feature")


def test_the_reference_specs_match_the_phase_4_and_5_tiers_exactly():
    """
    If the references drifted, the controls would be measured against a
    different baseline than the result they are testing.
    """
    assert (controls.SPEC_BY_KEY["ref_tier3"]["features"]
            == list(models.GAME_STATE_FEATURES))
    assert (controls.SPEC_BY_KEY["ref_tier5"]["features"]
            == list(models.OPPONENT_FEATURES))
    assert (controls.SPEC_BY_KEY["ref_tier2"]["features"]
            == list(models.GENERIC_FEATURES))


def test_linear_controls_use_a_model_that_cannot_isolate_games():
    """
    A logistic regression has no splits, so it is structurally immune to the
    mechanism under test. That is why C8 and C9 exist.
    """
    for key in ("c8_linear_opponent", "c9_linear_strength_diff"):
        spec = controls.SPEC_BY_KEY[key]
        assert spec["factory"] is models.make_generic_model
        assert spec["transform"] is models.add_generic_interaction


# ---------------------------------------------------------------------------
# The verdict rule
# ---------------------------------------------------------------------------

def test_verdict_confirms_the_artefact_when_random_reproduces_the_damage():
    text = controls.verdict(random_gap=0.045, opponent_gap=0.049)
    assert "ARTEFACT CONFIRMED" in text


def test_verdict_rejects_the_artefact_when_random_does_nothing():
    text = controls.verdict(random_gap=0.001, opponent_gap=0.049)
    assert "ARTEFACT NOT SUPPORTED" in text


def test_verdict_handles_an_opponent_effect_that_did_not_degrade():
    text = controls.verdict(random_gap=0.0, opponent_gap=-0.01)
    assert "no artefact to explain" in text


def test_verdict_threshold_is_exactly_half():
    """Half the damage from a meaningless column is already disqualifying."""
    assert "CONFIRMED" in controls.verdict(0.050, 0.100)
    assert "NOT SUPPORTED" in controls.verdict(0.049, 0.100)


# ---------------------------------------------------------------------------
# End-to-end smoke: every spec must actually fit and predict
# ---------------------------------------------------------------------------

def synthetic_model_frame(n_games=24, events_per_game=40, seed=3):
    """
    A frame carrying every column any control spec asks for.

    Catches a misspelled feature name or a transform shape mismatch here, in a
    two-second test, rather than ten minutes into the real run.
    """
    from src import features as feats

    rng = np.random.default_rng(seed)
    columns = set(feats.FEATURE_COLUMNS)
    for spec in controls.CONTROL_SPECS:
        columns |= set(spec["features"])

    rows = []
    for g in range(n_games):
        game_id = f"00216{g:05d}"
        won = int(rng.integers(0, 2))
        for e in range(events_per_game):
            row = {c: float(rng.normal()) for c in columns}
            row["game_id"] = game_id
            row["event_index"] = e
            row["celtics_won"] = won
            rows.append(row)
    return pd.DataFrame(rows)


def test_every_control_spec_fits_and_predicts():
    frame = synthetic_model_frame()
    target = frame["celtics_won"].astype(int)
    train = frame.iloc[:600]
    test = frame.iloc[600:]

    for spec in controls.CONTROL_SPECS:
        probabilities, model = models.fit_predict(
            spec, train, target.iloc[:600], test)
        assert len(probabilities) == len(test)
        assert np.all((probabilities >= 0) & (probabilities <= 1))

        # The in-sample path the runner uses for the memorisation gap.
        x_train = train[spec["features"]].astype(float).to_numpy()
        if spec["transform"] is not None:
            x_train = spec["transform"](x_train)
        in_sample = model.predict_proba(x_train)[:, 1]
        assert len(in_sample) == len(train)


def test_report_builder_survives_a_full_pass():
    """The report does real formatting; a f-string bug should not surface live."""
    from src import evaluate, run_controls

    frame = synthetic_model_frame()
    for column in ["period", "seconds_remaining_game", "is_clutch",
                   "is_overtime", "seconds_elapsed_game"]:
        if column not in frame:
            frame[column] = 0.0
    frame["period"] = 1
    frame["seconds_elapsed_game"] = 100.0
    frame["seconds_remaining_game"] = 2780.0
    frame["is_clutch"] = False          # a mask in evaluate.PHASES, so bool
    frame["is_overtime"] = False

    target = frame["celtics_won"].astype(int)
    rng = np.random.default_rng(5)
    predictions = {s["key"]: pd.Series(rng.uniform(0.2, 0.8, len(frame)))
                   for s in controls.CONTROL_SPECS}
    comparison = evaluate.compare_tiers(
        {k: v.to_numpy() for k, v in predictions.items()}, target.to_numpy())
    phase_tables = {k: evaluate.phase_table(frame, target, v.to_numpy())
                    for k, v in predictions.items()}
    bootstraps = {
        f"{a} vs {b}": evaluate.bootstrap_brier_difference(
            frame["game_id"].to_numpy(), target.to_numpy(),
            predictions[a].to_numpy(), predictions[b].to_numpy(),
            n_boot=50, seed=1)
        for a, b in controls.CONTROL_COMPARISONS}
    train_scores = {s["key"]: 0.2 for s in controls.CONTROL_SPECS}

    report = run_controls.build_report(frame, predictions, train_scores,
                                       comparison, bootstraps, phase_tables)
    assert "THE DECISIVE CONTROL" in report
    assert "TRAINING VERSUS OUT-OF-FOLD BRIER" in report
