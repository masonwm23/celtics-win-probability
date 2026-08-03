"""
The what-if endpoint must not build feature vectors that describe no game.

Two of the thirteen model features are functions of the others. If a caller
overrides `celtics_margin` and those two keep values computed from the real
margin, the model is handed a row claiming Boston are 32 down while the pressure
feature still says 12 down. That state cannot occur, the model was never fitted
on anything like it, and the number it returns is meaningless: before this was
fixed, one real second-quarter event moved from -12 to -32 and the win
probability went UP, 31.9% to 32.5%.
"""
import numpy as np
import pandas as pd
import pytest

from src.features import (
    CLUTCH_MARGIN,
    CLUTCH_SECONDS,
    DERIVED_FEATURES,
    DERIVED_INPUTS,
    recompute_derived,
)


def row(**kw):
    base = dict(
        celtics_margin=-12.0, seconds_remaining_period=600.0,
        seconds_remaining_game=1441.5, seconds_elapsed_game=1438.5,
        period=2, is_overtime=False, celtics_is_home=True,
        celtics_has_possession=True, momentum_120s=-4.0, momentum_300s=-6.0,
        is_clutch=False, margin_per_minute_remaining=-0.4995,
        possession_number=100,
    )
    base.update(kw)
    return pd.DataFrame([base])


def test_pressure_follows_the_margin():
    out = recompute_derived(row(celtics_margin=-32.0), skip={"celtics_margin"})
    minutes = 1441.5 / 60.0
    assert out["margin_per_minute_remaining"].iloc[0] == pytest.approx(-32.0 / minutes)


def test_pressure_is_not_left_holding_the_old_margin():
    """The exact shape of the reported bug."""
    stale = row(celtics_margin=-32.0)          # what the old endpoint produced
    fixed = recompute_derived(stale, skip={"celtics_margin"})
    assert stale["margin_per_minute_remaining"].iloc[0] == pytest.approx(-0.4995)
    assert fixed["margin_per_minute_remaining"].iloc[0] < -1.3


def test_clutch_turns_off_when_the_game_stops_being_close():
    close = row(period=4, seconds_remaining_period=60.0, celtics_margin=-2.0)
    assert recompute_derived(close, skip={"celtics_margin"})["is_clutch"].iloc[0]

    blowout = recompute_derived(close.assign(celtics_margin=-25.0),
                                skip={"celtics_margin"})
    assert not blowout["is_clutch"].iloc[0]


def test_clutch_turns_on_when_a_blowout_becomes_close():
    blowout = row(period=4, seconds_remaining_period=60.0, celtics_margin=-25.0)
    close = recompute_derived(blowout.assign(celtics_margin=-2.0),
                              skip={"celtics_margin"})
    assert close["is_clutch"].iloc[0]


def test_clutch_needs_the_fourth_period_not_just_a_close_score():
    early = row(period=2, seconds_remaining_period=10.0, celtics_margin=1.0)
    assert not recompute_derived(early)["is_clutch"].iloc[0]


def test_clutch_boundaries_are_inclusive_as_the_nba_defines_them():
    on = row(period=4, seconds_remaining_period=float(CLUTCH_SECONDS),
             celtics_margin=float(CLUTCH_MARGIN))
    assert recompute_derived(on)["is_clutch"].iloc[0]

    off = row(period=4, seconds_remaining_period=float(CLUTCH_SECONDS),
              celtics_margin=float(CLUTCH_MARGIN) + 1)
    assert not recompute_derived(off)["is_clutch"].iloc[0]


def test_an_explicit_override_of_a_derived_feature_is_respected():
    """Silently overwriting what the caller asked for would be its own bug."""
    out = recompute_derived(
        row(celtics_margin=-32.0, margin_per_minute_remaining=-99.0),
        skip={"celtics_margin", "margin_per_minute_remaining"})
    assert out["margin_per_minute_remaining"].iloc[0] == -99.0
    # The one that was NOT explicitly set is still rebuilt.
    assert out["is_clutch"].iloc[0] == False  # noqa: E712


def test_recomputing_an_untouched_row_changes_nothing():
    """Otherwise this would silently rewrite the training frame."""
    original = row()
    out = recompute_derived(original)
    assert out["is_clutch"].iloc[0] == original["is_clutch"].iloc[0]
    assert out["margin_per_minute_remaining"].iloc[0] == pytest.approx(
        original["margin_per_minute_remaining"].iloc[0], abs=1e-3)


def test_time_remaining_is_floored_so_the_final_event_cannot_divide_by_zero():
    out = recompute_derived(row(seconds_remaining_game=0.0, celtics_margin=-7.0))
    value = out["margin_per_minute_remaining"].iloc[0]
    assert np.isfinite(value)
    assert value == pytest.approx(-7.0 * 60.0)


def test_the_derived_registry_matches_what_is_actually_rebuilt():
    assert set(DERIVED_FEATURES) == set(DERIVED_INPUTS)
    for name, inputs in DERIVED_INPUTS.items():
        assert "celtics_margin" in inputs, f"{name} should depend on the margin"
