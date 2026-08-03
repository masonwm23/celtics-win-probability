"""
Tests for Phase 8: the forward-chaining comparison and the figures.

Two things need proving. First, that the forward-chaining split really does
exclude the future and that the comparison is scored on a shared test set rather
than on two different ones. Second, that the figures are derived from saved
results rather than from numbers typed in, and that they degrade gracefully when
an input is missing instead of failing halfway through.
"""

import numpy as np
import pandas as pd
import pytest

from src import config, figures, forward_chaining, splits


def season_frame(seasons=None, games_per_season=4, events=6):
    seasons = seasons or list(config.SEASONS)
    rows = []
    counter = 0
    for season in seasons:
        for g in range(games_per_season):
            counter += 1
            game_id = f"00216{counter:05d}"
            won = counter % 2
            for e in range(events):
                rows.append({
                    "game_id": game_id,
                    "event_index": e,
                    "season": season,
                    "celtics_won": won,
                    "celtics_margin": float(e - 3),
                    "seconds_remaining_game": float(2880 - e * 100),
                    "seconds_elapsed_game": float(e * 100),
                    "opponent_point_diff_prior": float(counter % 7) - 3.0,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The split really excludes the future
# ---------------------------------------------------------------------------

def test_forward_chaining_never_trains_on_a_later_season():
    """The entire point of the split. If this fails, nothing else matters."""
    frame = season_frame()
    order = {s: i for i, s in enumerate(config.SEASONS)}
    for season, train_index, _test in splits.forward_chaining(frame):
        train_seasons = set(frame.loc[train_index, "season"])
        assert train_seasons, "empty training set"
        assert max(order[s] for s in train_seasons) < order[season]


def test_forward_chaining_expands_the_window():
    frame = season_frame()
    sizes = [len(set(frame.loc[train, "season"]))
             for _s, train, _t in splits.forward_chaining(frame)]
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]


def test_forward_chaining_respects_the_minimum_training_seasons():
    frame = season_frame()
    for _season, train_index, _test in splits.forward_chaining(frame):
        assert len(set(frame.loc[train_index, "season"])) >= 3


def test_leave_one_season_out_does_train_on_the_future():
    """
    The contrast the paper draws. Stated as a test so the claim is verified
    rather than asserted in prose.
    """
    frame = season_frame()
    order = {s: i for i, s in enumerate(config.SEASONS)}
    saw_future = False
    for season, train_index, _test in splits.leave_one_season_out(frame):
        train_seasons = set(frame.loc[train_index, "season"])
        if max(order[s] for s in train_seasons) > order[season]:
            saw_future = True
    assert saw_future


# ---------------------------------------------------------------------------
# The comparison must be like for like
# ---------------------------------------------------------------------------

def test_common_test_seasons_is_the_intersection():
    loso = [{"held_out_season": s} for s in config.SEASONS]
    fc = [{"held_out_season": s} for s in config.SEASONS[3:]]
    shared = forward_chaining.common_test_seasons(loso, fc)
    assert shared == sorted(config.SEASONS[3:])
    assert len(shared) == 5


def test_common_test_seasons_excludes_seasons_only_one_split_covers():
    loso = [{"held_out_season": "2016-17"}, {"held_out_season": "2019-20"}]
    fc = [{"held_out_season": "2019-20"}]
    assert forward_chaining.common_test_seasons(loso, fc) == ["2019-20"]


def test_scoring_is_restricted_to_the_shared_rows():
    """
    A model scored on eight seasons cannot be compared to one scored on five.
    The mask is what makes the comparison honest.
    """
    frame = season_frame()
    target = frame["celtics_won"].astype(int)
    predictions = pd.Series(0.6, index=frame.index)
    mask = frame["season"].isin(config.SEASONS[3:]).to_numpy()
    result = forward_chaining.score_on(frame, target, predictions, mask)
    assert result["n"] == int(mask.sum())
    assert result["n"] < len(frame)


def test_scoring_ignores_rows_a_split_never_predicted():
    frame = season_frame()
    target = frame["celtics_won"].astype(int)
    predictions = pd.Series(np.nan, index=frame.index)
    predictions.loc[frame["season"].eq(config.SEASONS[4])] = 0.7
    mask = frame["season"].isin(config.SEASONS[3:]).to_numpy()
    result = forward_chaining.score_on(frame, target, predictions, mask)
    assert result["n"] == int(frame["season"].eq(config.SEASONS[4]).sum())


def test_scoring_returns_none_when_nothing_is_usable():
    frame = season_frame()
    target = frame["celtics_won"].astype(int)
    predictions = pd.Series(np.nan, index=frame.index)
    mask = np.ones(len(frame), dtype=bool)
    assert forward_chaining.score_on(frame, target, predictions, mask) is None


# ---------------------------------------------------------------------------
# The verdict must be able to say all three things
# ---------------------------------------------------------------------------

def boot(difference, low, high):
    return {"observed_difference": difference, "ci_low": low, "ci_high": high,
            "excludes_zero": bool(low > 0 or high < 0)}


def test_verdict_reports_no_penalty_when_intervals_span_zero():
    text = forward_chaining.verdict({"a": boot(-0.001, -0.004, 0.002)})
    assert "NO MEASURABLE PENALTY" in text


def test_verdict_reports_a_penalty_and_names_the_confound():
    text = forward_chaining.verdict({"a": boot(-0.012, -0.020, -0.005)})
    assert "MEASURABLY WORSE" in text
    assert "confound" in text


def test_verdict_flags_an_unexpected_improvement():
    text = forward_chaining.verdict({"a": boot(0.012, 0.005, 0.020)})
    assert "MEASURABLY BETTER" in text


def test_verdict_handles_an_empty_comparison():
    assert "No comparison" in forward_chaining.verdict({})


# ---------------------------------------------------------------------------
# Figures: derived from saved results, and failing softly
# ---------------------------------------------------------------------------

def test_comeback_game_is_chosen_by_rule_not_by_eye():
    """
    The selected game must be one Boston WON, and must have the lowest win
    probability of any such game. A cherry-picked figure is not evidence.
    """
    oof = pd.DataFrame({
        "game_id": ["a"] * 3 + ["b"] * 3 + ["c"] * 3,
        "celtics_won": [1] * 3 + [1] * 3 + [0] * 3,
        "tier3_celtics": [0.9, 0.8, 0.7,      # a: min 0.70
                          0.9, 0.1, 0.95,     # b: min 0.10, the comeback
                          0.5, 0.02, 0.1],    # c: lower, but a LOSS
    })
    game_id, low = figures.choose_comeback_game(pd.DataFrame(), oof)
    assert game_id == "b"
    assert low == pytest.approx(0.1)


def test_comeback_choice_never_returns_a_loss():
    oof = pd.DataFrame({
        "game_id": ["win", "loss"],
        "celtics_won": [1, 0],
        "tier3_celtics": [0.4, 0.001],
    })
    game_id, _low = figures.choose_comeback_game(pd.DataFrame(), oof)
    assert game_id == "win"


def test_figure_builder_skips_missing_inputs_instead_of_raising(tmp_path,
                                                               monkeypatch):
    """
    A missing prediction file should produce a clear instruction, not a
    traceback halfway through writing figures.
    """
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(config, "ALL_DIRS", [tmp_path / "processed",
                                             tmp_path / "reports",
                                             tmp_path / "figures"])
    monkeypatch.setattr(config, "MODEL_FRAME_PARQUET",
                        tmp_path / "model_frame.parquet")
    season_frame().to_parquet(tmp_path / "model_frame.parquet", index=False)

    written, skipped = figures.build_all()
    assert written == []
    assert len(skipped) == 2
    assert any("11_train_model" in hint for _n, hint in skipped)
    assert any("16_run_clean_tests" in hint for _n, hint in skipped)


def test_dose_response_figure_is_built_from_saved_scores(tmp_path, monkeypatch):
    """
    The signature figure must come from a file, so that a changed result changes
    the figure. This builds it from a synthetic score table end to end.
    """
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path)
    monkeypatch.setattr(config, "ALL_DIRS", [tmp_path])
    scores = pd.DataFrame({
        "tier": ["p7_tier3", "p7_opp_bins5", "p7_opp_bins20",
                 "p7_opp_bins100", "p7_opp_raw", "p7_rand_bins5",
                 "p7_rand_bins20", "p7_rand_bins100", "p7_rand_raw"],
        "brier": [0.1630, 0.1622, 0.1695, 0.1879, 0.1998,
                  0.1664, 0.1745, 0.1967, 0.2010],
        "cardinality": [np.nan, 5, 20, 88, 607, 5, 20, 100, 636],
    })
    path = figures.figure_dose_response(scores)
    assert path.exists()
    assert path.stat().st_size > 5000


def test_memorisation_figure_needs_both_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path)
    monkeypatch.setattr(config, "ALL_DIRS", [tmp_path])
    scores = pd.DataFrame({
        "tier": ["p7_tier3", "p7_opp_raw", "p7_rand_raw"],
        "name": ["tier 3", "opponent raw", "random raw"],
        "brier": [0.1630, 0.1998, 0.2010],
        "train_brier": [0.1487, 0.0677, 0.0692],
    })
    path = figures.figure_memorisation(scores)
    assert path.exists()


def test_figures_write_at_publication_resolution():
    assert figures.DPI >= 200
