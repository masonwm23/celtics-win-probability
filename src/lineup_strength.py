"""
Phase 3: lineup strength, computed inside a fold.

Why this module is separate from src/features.py
------------------------------------------------
Everything in features.py is computed within a single game from information
available at that moment, so it cannot leak. Lineup strength is different: it is
an average over many games, and averaging over the WHOLE dataset means a test
season's games help build a feature used to predict them. That is leakage, it
inflates results, and it is invisible unless you look for it.

So this module never computes anything without being told which seasons it may
use. `compute_player_values` requires an explicit season list, which callers get
from `splits.fold_seasons(frame, train_index)`.

The measure
-----------
Per-minute plus/minus, shrunk toward zero by playing time:

    value(p) = total_plus_minus(p) / (total_minutes(p) + K)

K is a shrinkage constant expressed in minutes. A player with few minutes is
pulled toward zero, which is the right behaviour: ten strong minutes is not
evidence of a strong player. With K = 500 (roughly seven full games), a player
needs a real body of work before their raw rate is taken at face value.

This is a deliberately simple measure. It is NOT a plus/minus model like RAPM:
those need regularised regression across all lineups and are a project of their
own. The honest description for the paper is "shrunk per-minute plus/minus",
and its limitations belong in the limitations section rather than being dressed up.

A caveat that has to be stated. Player plus/minus does not reconcile in 10 of 636
games (see data/lineup_risk_games.csv). Those games are included by default since
the error is small relative to a season of minutes, but `exclude_games` lets the
sensitivity check be run, and it should be run before the result is reported.

Lineup strength for a five-man unit is the sum of its five player values, so it is
in the same units and a lineup with an unknown player is not silently penalised
more than the shrinkage already implies.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Shrinkage constant, in minutes. Roughly seven full games.
DEFAULT_SHRINKAGE_MINUTES = 500.0


def compute_player_values(rosters: pd.DataFrame, seasons,
                          shrinkage_minutes=DEFAULT_SHRINKAGE_MINUTES,
                          exclude_games=None) -> pd.Series:
    """
    Player value from the given seasons ONLY.

    Parameters
    ----------
    rosters : DataFrame
        The Phase 2 roster table, one row per player per game per team.
    seasons : list
        The seasons this fold is permitted to use. Required, not optional, so a
        caller cannot accidentally compute on everything.
    shrinkage_minutes : float
        Minutes of shrinkage toward zero.
    exclude_games : iterable or None
        Game IDs to leave out, for the sensitivity check against the games whose
        plus/minus does not reconcile.

    Returns
    -------
    Series indexed by person_id, giving plus/minus per minute, shrunk.
    """
    if seasons is None:
        raise ValueError(
            "seasons is required. Computing player values across all seasons "
            "leaks test information into training."
        )
    seasons = list(seasons)
    if not seasons:
        raise ValueError("seasons is empty, so no player values can be computed")

    subset = rosters[rosters["season"].isin(seasons)]
    if exclude_games:
        subset = subset[~subset["game_id"].isin(set(exclude_games))]
    if subset.empty:
        raise ValueError(f"no roster rows for seasons {seasons}")

    grouped = subset.groupby("person_id").agg(
        total_plus_minus=("plusMinusPoints", "sum"),
        total_minutes=("minutes_played", "sum"),
    )
    values = (grouped["total_plus_minus"]
              / (grouped["total_minutes"] + shrinkage_minutes))
    values.name = "player_value"
    logger.info("player values from %d season(s): %d players, "
                "mean %.4f, sd %.4f", len(seasons), len(values),
                float(values.mean()), float(values.std()))
    return values


def lineup_value(lineup_ids, player_values: pd.Series,
                 default=0.0) -> float:
    """
    Sum of player values for a five-man unit.

    A player absent from `player_values` contributes `default`. That happens for
    a rookie whose only season is the held-out one, which is exactly the situation
    a fold should face: the model does not get to know about them in advance.
    """
    if not lineup_ids:
        return default * 5
    return float(sum(player_values.get(int(pid), default) for pid in lineup_ids))


def parse_lineup(value):
    """Turn the stored 'id,id,id,id,id' lineup string into a list of ints."""
    if isinstance(value, (list, tuple, set)):
        return [int(v) for v in value]
    text = str(value or "").strip()
    if not text:
        return []
    return [int(part) for part in text.split(",") if part]


def attach_lineup_strength(events: pd.DataFrame, lineups: pd.DataFrame,
                           player_values: pd.Series,
                           celtics_is_home_column="celtics_is_home"
                           ) -> pd.DataFrame:
    """
    Add Celtics and opponent lineup strength to an event table.

    `lineups` carries home and away five-man units per event; which one is Boston
    depends on the game, so it is resolved per row rather than assumed.
    """
    merged = events.merge(
        lineups[["game_id", "event_index", "home_lineup", "away_lineup"]],
        on=["game_id", "event_index"], how="left", validate="one_to_one")

    home_values = merged["home_lineup"].map(
        lambda v: lineup_value(parse_lineup(v), player_values))
    away_values = merged["away_lineup"].map(
        lambda v: lineup_value(parse_lineup(v), player_values))

    is_home = merged[celtics_is_home_column].astype(bool)
    merged["celtics_lineup_strength"] = np.where(is_home, home_values, away_values)
    merged["opponent_lineup_strength"] = np.where(is_home, away_values, home_values)
    merged["lineup_strength_diff"] = (merged["celtics_lineup_strength"]
                                      - merged["opponent_lineup_strength"])
    return merged


LINEUP_FEATURE_COLUMNS = [
    "celtics_lineup_strength",
    "opponent_lineup_strength",
    "lineup_strength_diff",
]
