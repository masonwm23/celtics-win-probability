"""
Tests for Phase 5: as-of-date opponent strength.

The leakage tests here are the point of the file. A full-season record includes
games played after the one being predicted, and that is the single most likely
way this project could produce an inflated opponent effect. These tests construct
a season where the answer is known by hand and assert that no future information
reaches the feature.
"""

import numpy as np
import pandas as pd
import pytest

from src import config, opponent_strength


def league_log(rows):
    """
    Build a league log from (date, game_id, team, points) tuples.

    Two rows per game, as the real feed provides.
    """
    frame = pd.DataFrame(rows, columns=["GAME_DATE", "GAME_ID",
                                        "TEAM_ABBREVIATION", "PTS"])
    frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"])
    frame["SEASON"] = "2016-17"
    frame["GAME_ID"] = frame["GAME_ID"].astype(str)
    totals = frame.groupby("GAME_ID")["PTS"].transform("sum")
    frame["WON"] = (frame["PTS"] > totals - frame["PTS"]).astype(int)
    return frame


def simple_season():
    """
    ATL plays four games. They win the first three by 10 and lose the fourth
    by 30, so 'record so far' is very different from 'record all season'.
    """
    return league_log([
        ("2016-10-25", "0021600001", "ATL", 110), ("2016-10-25", "0021600001", "BKN", 100),
        ("2016-10-27", "0021600002", "ATL", 110), ("2016-10-27", "0021600002", "CHI", 100),
        ("2016-10-29", "0021600003", "ATL", 110), ("2016-10-29", "0021600003", "MIA", 100),
        ("2016-11-01", "0021600004", "ATL",  90), ("2016-11-01", "0021600004", "BOS", 120),
    ])


# ---------------------------------------------------------------------------
# Margins come from the paired rows, not from PLUS_MINUS
# ---------------------------------------------------------------------------

def test_margin_is_computed_from_the_paired_team_row():
    logs = opponent_strength.add_true_margins(simple_season())
    atl = logs.loc[logs["TEAM_ABBREVIATION"].eq("ATL")].sort_values("GAME_DATE")
    assert list(atl["margin"]) == [10, 10, 10, -30]
    bos = logs.loc[logs["TEAM_ABBREVIATION"].eq("BOS")]
    assert list(bos["margin"]) == [30]


def test_margin_refuses_a_game_without_two_rows():
    """A one-sided game makes the margin uncomputable; it must raise."""
    logs = simple_season()
    logs = logs.drop(logs.index[1])          # remove BKN's row
    with pytest.raises(ValueError, match="exactly two rows"):
        opponent_strength.add_true_margins(logs)


# ---------------------------------------------------------------------------
# THE LEAKAGE TESTS
# ---------------------------------------------------------------------------

def prior_frame():
    logs = opponent_strength.add_true_margins(simple_season())
    return opponent_strength._prior_stats(logs)


def test_a_game_never_contributes_to_its_own_feature():
    """
    The most direct leak. ATL's fourth game is a 30-point loss. Its own prior
    point differential must reflect only the three wins that came before it.
    """
    stats = prior_frame()
    atl = stats.loc[stats["TEAM_ABBREVIATION"].eq("ATL")].sort_values("GAME_DATE")
    fourth = atl.iloc[3]
    assert fourth["games_played_prior"] == 3
    assert fourth["point_diff_prior_raw"] == pytest.approx(10.0)
    assert fourth["win_pct_prior_raw"] == pytest.approx(1.0)


def test_no_future_game_reaches_an_earlier_feature():
    """
    ATL's second game must see only the first. If the loss in game four leaked
    backwards, the prior differential would drop below 10.
    """
    stats = prior_frame()
    atl = stats.loc[stats["TEAM_ABBREVIATION"].eq("ATL")].sort_values("GAME_DATE")
    assert atl.iloc[1]["games_played_prior"] == 1
    assert atl.iloc[1]["point_diff_prior_raw"] == pytest.approx(10.0)
    assert atl.iloc[2]["games_played_prior"] == 2
    assert atl.iloc[2]["point_diff_prior_raw"] == pytest.approx(10.0)


def test_the_first_game_of_a_season_has_no_prior_information():
    stats = prior_frame()
    atl = stats.loc[stats["TEAM_ABBREVIATION"].eq("ATL")].sort_values("GAME_DATE")
    first = atl.iloc[0]
    assert first["games_played_prior"] == 0
    assert np.isnan(first["point_diff_prior_raw"])
    assert np.isnan(first["win_pct_prior_raw"])


def test_full_season_average_would_have_been_different():
    """
    Proves the test above is actually testing something. ATL's full-season
    differential is 0.0; the correct as-of-date value before game four is 10.0.
    A leaky implementation would report the former.
    """
    logs = opponent_strength.add_true_margins(simple_season())
    atl_full = logs.loc[logs["TEAM_ABBREVIATION"].eq("ATL"), "margin"].mean()
    assert atl_full == pytest.approx(0.0)

    stats = prior_frame()
    atl = stats.loc[stats["TEAM_ABBREVIATION"].eq("ATL")].sort_values("GAME_DATE")
    assert atl.iloc[3]["point_diff_prior_raw"] == pytest.approx(10.0)
    assert atl.iloc[3]["point_diff_prior_raw"] != pytest.approx(atl_full)


def test_seasons_do_not_bleed_into_each_other():
    """A new season starts with no prior games, not last season's record."""
    rows = []
    for i, (season, date) in enumerate([("2016-17", "2016-11-01"),
                                        ("2016-17", "2016-11-03"),
                                        ("2017-18", "2017-11-01")]):
        gid = f"002160000{i + 1}"
        rows.append((date, gid, "ATL", 120))
        rows.append((date, gid, "BKN", 100))
    frame = pd.DataFrame(rows, columns=["GAME_DATE", "GAME_ID",
                                        "TEAM_ABBREVIATION", "PTS"])
    frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"])
    frame["SEASON"] = ["2016-17"] * 4 + ["2017-18"] * 2
    totals = frame.groupby("GAME_ID")["PTS"].transform("sum")
    frame["WON"] = (frame["PTS"] > totals - frame["PTS"]).astype(int)

    stats = opponent_strength._prior_stats(
        opponent_strength.add_true_margins(frame))
    new_season = stats.loc[stats["SEASON"].eq("2017-18")
                           & stats["TEAM_ABBREVIATION"].eq("ATL")].iloc[0]
    assert new_season["games_played_prior"] == 0
    assert np.isnan(new_season["point_diff_prior_raw"])


# ---------------------------------------------------------------------------
# Shrinkage
# ---------------------------------------------------------------------------

def test_shrinkage_pulls_a_tiny_sample_toward_the_league_mean():
    """A team that is 1-0 is not the best team in the league."""
    one_game = opponent_strength._shrink([1.0], [1], 0.5)[0]
    forty_games = opponent_strength._shrink([1.0], [40], 0.5)[0]
    assert 0.5 < one_game < 0.6          # barely moved from the mean
    assert forty_games > 0.85            # now taken close to face value
    assert one_game < forty_games


def test_shrinkage_after_ten_games_is_half_weight():
    """K is 10 games, so at n = 10 the raw value counts for half."""
    value = opponent_strength._shrink([20.0], [10], 0.0)[0]
    assert value == pytest.approx(10.0)


def test_shrinkage_handles_a_team_with_no_prior_games():
    value = opponent_strength._shrink([np.nan], [0], 0.0)[0]
    assert value == pytest.approx(0.0)


def test_shrinkage_is_symmetric_for_negative_values():
    good = opponent_strength._shrink([15.0], [20], 0.0)[0]
    bad = opponent_strength._shrink([-15.0], [20], 0.0)[0]
    assert good == pytest.approx(-bad)


# ---------------------------------------------------------------------------
# Feature declaration
# ---------------------------------------------------------------------------

def test_every_declared_feature_is_pregame():
    """
    No in-game quantity may appear among the opponent features. This is a guard
    against someone later adding a column that knows the score.
    """
    banned = {"celtics_margin", "seconds_remaining_game", "momentum_120s",
              "celtics_score", "opponent_score", "is_clutch",
              "celtics_won", "possession_number"}
    assert not set(opponent_strength.OPPONENT_FEATURE_COLUMNS) & banned


def test_feature_names_say_prior_or_form():
    """Naming discipline: every measure advertises that it is as-of-date."""
    for name in opponent_strength.OPPONENT_FEATURE_COLUMNS:
        assert "prior" in name or "form" in name
