"""
Phase 3: leakage-safe cross-validation design.

This module exists because the single easiest way to produce an impressive and
worthless sports model is to let information cross between training and test. The
research plan calls for season-based validation, and this is where that is
implemented and, more importantly, where it is TESTED.

Three distinct leak risks, each handled explicitly.

1. EVENT-LEVEL SPLITTING. A game contributes hundreds of events that share one
   outcome. Splitting events at random puts events from the same game on both
   sides, so the model can memorise "this game was a win" from one event and be
   graded on another. Every split here is by season, and games belong to exactly
   one season, so no game can straddle. `assert_no_game_straddles` verifies it
   rather than trusting it.

2. AGGREGATE FEATURES COMPUTED ON EVERYTHING. Lineup strength and opponent
   strength are averages over games. Computing them once across the full dataset
   means a test-season game influenced a feature used to predict it. Those
   features are therefore built per fold from training seasons only, in
   src/lineup_strength.py, and `fold_seasons` exposes exactly which seasons a
   fold may look at.

3. SILENT LEAKS OF UNKNOWN ORIGIN. The two above are the ones we know about. The
   shuffled-target test is the check for the ones we do not: permute the outcome
   across games and refit. If performance does not collapse to chance, something
   is leaking and every downstream number is worthless.

On the shuffle being done BY GAME
---------------------------------
Permuting the target independently per event would destroy the game-level
structure and trivially wreck any model, proving nothing. The honest test keeps
all events of a game together and permutes which outcome each game carries, so
the features, the grouping and the class balance are all preserved and the ONLY
thing broken is the link between a game's state and its result.
"""

import logging

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


class LeakageError(AssertionError):
    """Raised when a split would leak information between train and test."""


def assert_no_game_straddles(frame: pd.DataFrame, train_index, test_index,
                             group_column="game_id"):
    """
    Verify that no game appears on both sides of a split.

    Raises LeakageError with the offending game IDs. This is an assertion, not a
    warning, because a straddling game invalidates every metric computed from the
    split and there is no safe way to continue.
    """
    train_games = set(frame.loc[train_index, group_column])
    test_games = set(frame.loc[test_index, group_column])
    overlap = train_games & test_games
    if overlap:
        raise LeakageError(
            f"{len(overlap)} game(s) appear in both train and test: "
            f"{sorted(overlap)[:10]}"
        )
    return True


def assert_no_season_straddles(frame: pd.DataFrame, train_index, test_index,
                               season_column="season"):
    """Verify the split is clean at the season level too."""
    train_seasons = set(frame.loc[train_index, season_column])
    test_seasons = set(frame.loc[test_index, season_column])
    overlap = train_seasons & test_seasons
    if overlap:
        raise LeakageError(
            f"season(s) appear in both train and test: {sorted(overlap)}"
        )
    return True


def leave_one_season_out(frame: pd.DataFrame, season_column="season"):
    """
    Yield (season, train_index, test_index) holding out one season at a time.

    This is the design named in the research plan. Each fold trains on seven
    seasons and is graded on the eighth, so a fold never sees any part of the
    season it is judged on.
    """
    seasons = [s for s in config.SEASONS if s in set(frame[season_column])]
    for season in seasons:
        test_mask = frame[season_column].eq(season)
        train_index = frame.index[~test_mask]
        test_index = frame.index[test_mask]
        if len(train_index) == 0 or len(test_index) == 0:
            continue
        assert_no_game_straddles(frame, train_index, test_index)
        assert_no_season_straddles(frame, train_index, test_index)
        yield season, train_index, test_index


def forward_chaining(frame: pd.DataFrame, min_train_seasons=3,
                     season_column="season"):
    """
    Yield (season, train_index, test_index) training only on EARLIER seasons.

    Leave-one-season-out lets a fold train on the future to predict the past,
    which is fine for measuring team-specific calibration but is not how the model
    would be deployed. This expanding-window split is the honest deployment
    analogue, and reporting both is more informative than picking one.
    """
    seasons = [s for s in config.SEASONS if s in set(frame[season_column])]
    for position in range(min_train_seasons, len(seasons)):
        test_season = seasons[position]
        train_seasons = seasons[:position]
        train_index = frame.index[frame[season_column].isin(train_seasons)]
        test_index = frame.index[frame[season_column].eq(test_season)]
        if len(train_index) == 0 or len(test_index) == 0:
            continue
        assert_no_game_straddles(frame, train_index, test_index)
        assert_no_season_straddles(frame, train_index, test_index)
        yield test_season, train_index, test_index


def fold_seasons(frame: pd.DataFrame, train_index, season_column="season"):
    """
    The seasons a fold is allowed to look at when building aggregate features.

    Any feature averaged over games must be computed from these seasons only.
    Passing this explicitly is what stops lineup strength from quietly being
    fitted on the test season.
    """
    return sorted(set(frame.loc[train_index, season_column]))


def shuffle_target_by_game(frame: pd.DataFrame, seed=None,
                           target_column="celtics_won",
                           group_column="game_id") -> pd.Series:
    """
    Permute the outcome ACROSS GAMES, keeping every event of a game together.

    Returns a new target Series aligned to `frame`. The features, the grouping and
    the overall win rate are all unchanged; the only thing destroyed is the
    correspondence between a game's state and its result.

    A model trained on this must score at chance. If it does not, a feature is
    carrying the answer.
    """
    seed = config.RANDOM_SEED if seed is None else seed
    rng = np.random.default_rng(seed)

    per_game = (frame[[group_column, target_column]]
                .drop_duplicates(group_column)
                .set_index(group_column)[target_column])
    shuffled_values = rng.permutation(per_game.to_numpy())
    shuffled = pd.Series(shuffled_values, index=per_game.index)
    return frame[group_column].map(shuffled).astype(int)


def describe_split(frame: pd.DataFrame, season, train_index, test_index,
                   group_column="game_id", target_column="celtics_won"):
    """One-line summary of a fold, for the validation report."""
    train, test = frame.loc[train_index], frame.loc[test_index]
    train_games = train[[group_column, target_column]].drop_duplicates(group_column)
    test_games = test[[group_column, target_column]].drop_duplicates(group_column)
    return {
        "held_out_season": season,
        "train_events": len(train),
        "test_events": len(test),
        "train_games": len(train_games),
        "test_games": len(test_games),
        "train_seasons": len(set(train["season"])),
        "train_win_rate": round(float(train_games[target_column].mean()), 4),
        "test_win_rate": round(float(test_games[target_column].mean()), 4),
    }
