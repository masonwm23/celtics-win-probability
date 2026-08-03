"""
Tests for Phase 3: features, splits, and fold-safe lineup strength.

No network, synthetic fixtures. The split and lineup-strength tests are the ones
that matter most: they are the difference between a defensible result and an
impressive worthless one.
"""

import numpy as np
import pandas as pd
import pytest

from src import config, features, lineup_strength, splits


# ---------------------------------------------------------------------------
# Possession state machine
# ---------------------------------------------------------------------------

def ev(action_type, tricode, **kw):
    row = {"action_type": action_type, "team_tricode": tricode}
    row.update(kw)
    return row


def possession_of(rows):
    frame = pd.DataFrame(rows)
    return list(features.assign_possession(frame))


def test_shot_attempt_shows_who_has_the_ball():
    assert possession_of([ev("Missed Shot", "BOS")]) == ["BOS"]
    assert possession_of([ev("Made Shot", "ATL")]) == ["ATL"]


def test_rebound_transfers_possession():
    got = possession_of([ev("Missed Shot", "BOS"), ev("Rebound", "ATL")])
    assert got == ["BOS", "ATL"]


def test_offensive_rebound_keeps_possession():
    got = possession_of([ev("Missed Shot", "BOS"), ev("Rebound", "BOS")])
    assert got == ["BOS", "BOS"]


def test_turnover_shows_the_committing_team_had_the_ball():
    got = possession_of([ev("Turnover", "BOS"), ev("Made Shot", "ATL")])
    assert got == ["BOS", "ATL"]


def test_foul_implies_the_other_team_has_the_ball():
    """A foul is committed by the defence, so possession is with the opponent."""
    got = possession_of([ev("Made Shot", "BOS"), ev("Foul", "ATL")])
    assert got == ["BOS", "BOS"]
    got = possession_of([ev("Made Shot", "ATL"), ev("Foul", "BOS")])
    assert got == ["ATL", "ATL"]


def test_substitutions_and_timeouts_do_not_change_possession():
    got = possession_of([
        ev("Missed Shot", "BOS"),
        ev("Substitution", "ATL"),
        ev("Timeout", "ATL"),
        ev("", "ATL"),                 # a steal annotation
        ev("Rebound", "BOS"),
    ])
    assert got == ["BOS", "BOS", "BOS", "BOS", "BOS"]


def test_free_throws_show_possession():
    got = possession_of([ev("Free Throw", "ATL")])
    assert got == ["ATL"]


def test_possession_is_empty_until_first_evidence():
    got = possession_of([ev("period", ""), ev("Substitution", "BOS"),
                         ev("Missed Shot", "ATL")])
    assert got == ["", "", "ATL"]


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

def test_momentum_counts_only_the_window():
    seconds = np.array([0.0, 10.0, 200.0, 400.0])
    celtics = np.array([2.0, 3.0, 0.0, 2.0])
    opponent = np.array([0.0, 0.0, 2.0, 0.0])
    # 120 second window at t=400 sees only the event at 400.
    got = features.rolling_momentum(seconds, celtics, opponent, 120.0)
    assert got[0] == 2
    assert got[1] == 5          # both early events inside the window
    assert got[-1] == 2         # only the last event


def test_momentum_is_celtics_perspective():
    seconds = np.array([0.0, 5.0])
    got = features.rolling_momentum(seconds, np.array([0.0, 0.0]),
                                    np.array([3.0, 3.0]), 120.0)
    assert got[-1] == -6


# ---------------------------------------------------------------------------
# Clutch and time
# ---------------------------------------------------------------------------

def game_frame(rows):
    base = {
        "game_id": "0021600006", "season": "2016-17",
        "game_date": pd.Timestamp("2016-10-26"), "opponent_tricode": "BKN",
        "celtics_is_home": True, "celtics_won": 1, "event_index": 0,
        "period": 4, "is_overtime": False, "seconds_remaining_period": 100.0,
        "seconds_elapsed_game": 2780.0, "action_type": "Made Shot",
        "team_tricode": "BOS", "celtics_score": 100, "opponent_score": 98,
        "celtics_margin": 2,
    }
    out = []
    for i, row in enumerate(rows):
        merged = dict(base)
        merged.update(row)
        merged["event_index"] = i
        out.append(merged)
    return pd.DataFrame(out)


def test_clutch_requires_late_period_and_close_score():
    frame = game_frame([
        {"period": 4, "seconds_remaining_period": 100, "celtics_margin": 2},
        {"period": 4, "seconds_remaining_period": 100, "celtics_margin": 20},
        {"period": 4, "seconds_remaining_period": 400, "celtics_margin": 2},
        {"period": 2, "seconds_remaining_period": 100, "celtics_margin": 2},
        {"period": 5, "seconds_remaining_period": 100, "celtics_margin": -5},
    ])
    got = list(features.build_features(frame)["is_clutch"])
    assert got == [True, False, False, False, True]


def test_seconds_remaining_never_negative_in_overtime():
    frame = game_frame([
        {"period": 5, "seconds_remaining_period": 120.0,
         "seconds_elapsed_game": 3060.0},
    ])
    out = features.build_features(frame)
    assert out["seconds_remaining_game"].iloc[0] == 120.0
    assert (out["seconds_remaining_game"] >= 0).all()


def test_score_change_sums_to_final_margin():
    frame = game_frame([
        {"celtics_score": 0, "opponent_score": 0, "celtics_margin": 0},
        {"celtics_score": 2, "opponent_score": 0, "celtics_margin": 2},
        {"celtics_score": 2, "opponent_score": 3, "celtics_margin": -1},
        {"celtics_score": 5, "opponent_score": 3, "celtics_margin": 2},
    ])
    out = features.build_features(frame)
    assert out["score_change"].sum() == out["celtics_margin"].iloc[-1]


# ---------------------------------------------------------------------------
# Splits. The leakage guards.
# ---------------------------------------------------------------------------

def split_frame(games_per_season=4, events_per_game=10):
    rows = []
    rng = np.random.default_rng(0)
    for season in config.SEASONS:
        for g in range(games_per_season):
            game_id = f"{season}-{g}"
            won = int(rng.integers(0, 2))
            for i in range(events_per_game):
                rows.append({"game_id": game_id, "season": season,
                             "celtics_won": won, "event_index": i})
    return pd.DataFrame(rows)


def test_leave_one_season_out_covers_every_season():
    frame = split_frame()
    seasons = [s for s, _, _ in splits.leave_one_season_out(frame)]
    assert seasons == config.SEASONS


def test_no_game_appears_in_both_train_and_test():
    frame = split_frame()
    for _season, train, test in splits.leave_one_season_out(frame):
        assert set(frame.loc[train, "game_id"]).isdisjoint(
            set(frame.loc[test, "game_id"]))


def test_no_season_appears_in_both_train_and_test():
    frame = split_frame()
    for season, train, test in splits.leave_one_season_out(frame):
        assert season not in set(frame.loc[train, "season"])


def test_straddling_split_raises():
    """The guard must fire, otherwise it is decoration."""
    frame = split_frame()
    with pytest.raises(splits.LeakageError):
        splits.assert_no_game_straddles(frame, frame.index[:15], frame.index[5:25])


def test_forward_chaining_never_trains_on_the_future():
    frame = split_frame()
    order = {s: i for i, s in enumerate(config.SEASONS)}
    for season, train, _test in splits.forward_chaining(frame):
        train_seasons = set(frame.loc[train, "season"])
        assert all(order[s] < order[season] for s in train_seasons)


def test_fold_seasons_excludes_the_held_out_season():
    frame = split_frame()
    for season, train, _test in splits.leave_one_season_out(frame):
        assert season not in splits.fold_seasons(frame, train)
        assert len(splits.fold_seasons(frame, train)) == len(config.SEASONS) - 1


# ---------------------------------------------------------------------------
# Shuffled target
# ---------------------------------------------------------------------------

def test_shuffle_keeps_every_event_of_a_game_together():
    frame = split_frame()
    shuffled = splits.shuffle_target_by_game(frame, seed=1)
    per_game = pd.Series(shuffled.to_numpy(), index=frame["game_id"])
    assert (per_game.groupby(level=0).nunique() == 1).all()


def test_shuffle_preserves_the_overall_win_rate():
    frame = split_frame()
    shuffled = splits.shuffle_target_by_game(frame, seed=1)
    original = frame.drop_duplicates("game_id")["celtics_won"]
    new = (pd.Series(shuffled.to_numpy(), index=frame["game_id"])
           .groupby(level=0).first())
    assert original.mean() == pytest.approx(new.mean())


def test_shuffle_actually_changes_some_labels():
    frame = split_frame()
    shuffled = splits.shuffle_target_by_game(frame, seed=1)
    assert not (shuffled.to_numpy() == frame["celtics_won"].to_numpy()).all()


def test_shuffle_is_reproducible_with_a_seed():
    frame = split_frame()
    a = splits.shuffle_target_by_game(frame, seed=7)
    b = splits.shuffle_target_by_game(frame, seed=7)
    assert (a.to_numpy() == b.to_numpy()).all()


# ---------------------------------------------------------------------------
# Fold-safe lineup strength
# ---------------------------------------------------------------------------

def roster_frame():
    rows = []
    for season in config.SEASONS:
        for pid in (1, 2, 3):
            rows.append({"season": season, "game_id": f"{season}-0",
                         "person_id": pid, "plusMinusPoints": 100.0,
                         "minutes_played": 500.0})
    # Player 99 exists only in the final season.
    rows.append({"season": "2023-24", "game_id": "2023-24-0", "person_id": 99,
                 "plusMinusPoints": 500.0, "minutes_played": 500.0})
    return pd.DataFrame(rows)


def test_player_values_require_an_explicit_season_list():
    """
    Refusing the default is the whole point. A caller who forgets to pass seasons
    would otherwise silently compute on the test season too.
    """
    with pytest.raises(ValueError, match="seasons is required"):
        lineup_strength.compute_player_values(roster_frame(), None)
    with pytest.raises(ValueError, match="empty"):
        lineup_strength.compute_player_values(roster_frame(), [])


def test_player_values_only_use_the_given_seasons():
    rosters = roster_frame()
    values = lineup_strength.compute_player_values(rosters, ["2016-17"])
    assert 99 not in values.index          # played only in 2023-24
    assert set(values.index) == {1, 2, 3}


def test_a_player_from_the_held_out_season_never_leaks_in():
    rosters = roster_frame()
    training = [s for s in config.SEASONS if s != "2023-24"]
    values = lineup_strength.compute_player_values(rosters, training)
    assert 99 not in values.index


def test_unknown_player_gets_the_default_not_an_error():
    values = pd.Series({1: 0.10, 2: 0.20})
    assert lineup_strength.lineup_value([1, 2, 999], values) == pytest.approx(0.30)


def test_shrinkage_pulls_low_minute_players_toward_zero():
    """
    Two players with an IDENTICAL raw rate of +10 per minute. The one with ten
    minutes is pulled far closer to zero than the one with two thousand, because
    ten strong minutes is not evidence of a strong player.
    """
    rosters = pd.DataFrame([
        {"season": "2016-17", "game_id": "g", "person_id": 1,
         "plusMinusPoints": 100.0, "minutes_played": 10.0},
        {"season": "2016-17", "game_id": "g", "person_id": 2,
         "plusMinusPoints": 20000.0, "minutes_played": 2000.0},
    ])
    values = lineup_strength.compute_player_values(rosters, ["2016-17"])
    raw_rate = 10.0
    assert values[1] == pytest.approx(100 / 510)          # shrunk to ~2% of raw
    assert values[2] == pytest.approx(20000 / 2500)       # shrunk to 80% of raw
    assert values[1] / raw_rate < 0.05
    assert values[2] / raw_rate > 0.75
    assert values[1] < values[2]


def test_shrinkage_equals_raw_rate_times_the_shrinkage_factor():
    """value = raw_rate * minutes/(minutes + K), standard shrinkage toward zero."""
    minutes, plus_minus = 300.0, 150.0
    k = lineup_strength.DEFAULT_SHRINKAGE_MINUTES
    rosters = pd.DataFrame([{"season": "2016-17", "game_id": "g", "person_id": 1,
                             "plusMinusPoints": plus_minus,
                             "minutes_played": minutes}])
    value = lineup_strength.compute_player_values(rosters, ["2016-17"])[1]
    expected = (plus_minus / minutes) * (minutes / (minutes + k))
    assert value == pytest.approx(expected)


def test_exclude_games_supports_the_sensitivity_check():
    """
    The 10 games whose plus/minus does not reconcile can be dropped, so the
    lineup result can be reported with and without them.
    """
    rosters = pd.DataFrame([
        {"season": "2016-17", "game_id": "good", "person_id": 1,
         "plusMinusPoints": 50.0, "minutes_played": 400.0},
        {"season": "2016-17", "game_id": "suspect", "person_id": 1,
         "plusMinusPoints": 500.0, "minutes_played": 30.0},
    ])
    full = lineup_strength.compute_player_values(rosters, ["2016-17"])
    reduced = lineup_strength.compute_player_values(
        rosters, ["2016-17"], exclude_games=["suspect"])
    assert full[1] == pytest.approx(550 / 930)
    assert reduced[1] == pytest.approx(50 / 900)
    assert reduced[1] < full[1]


def test_parse_lineup_handles_string_and_sequence():
    assert lineup_strength.parse_lineup("1,2,3") == [1, 2, 3]
    assert lineup_strength.parse_lineup([1, 2]) == [1, 2]
    assert lineup_strength.parse_lineup("") == []
    assert lineup_strength.parse_lineup(None) == []


# ---------------------------------------------------------------------------
# Leak-test statistics. These decide whether the centrepiece check is honest.
# ---------------------------------------------------------------------------

def test_null_standard_error_shrinks_as_games_increase():
    """
    The effective sample is games, not events. Two games gives an enormous null
    band; 636 games gives a usable one. An early version of this audit judged
    folds of 2 to 7 games and reported noise as a leak.
    """
    from src.validate_phase3 import auc_null_standard_error
    se_tiny = auc_null_standard_error(1, 1)
    se_fold = auc_null_standard_error(52, 28)
    se_pooled = auc_null_standard_error(413, 223)
    assert se_tiny > se_fold > se_pooled
    assert se_pooled < 0.05
    assert se_tiny > 0.4


def test_null_standard_error_undefined_without_both_classes():
    from src.validate_phase3 import auc_null_standard_error
    assert np.isnan(auc_null_standard_error(0, 10))
    assert np.isnan(auc_null_standard_error(10, 0))


def test_pooled_threshold_is_wider_than_a_single_fold_threshold():
    """The derived tolerance must reflect the sample it is applied to."""
    from src.validate_phase3 import auc_null_standard_error
    assert 3 * auc_null_standard_error(52, 28) > 3 * auc_null_standard_error(413, 223)
