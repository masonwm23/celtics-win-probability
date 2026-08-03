"""
Phase 5: opponent strength, as of the game date.

THE LEAKAGE TRAP THIS MODULE EXISTS TO AVOID
--------------------------------------------
A team's full-season win percentage or point differential includes games played
AFTER the game being predicted. Joining an end-of-season record onto a game in
November means the feature knows how the rest of the season went. It is a leak,
and it is a subtle one because the feature feels like pregame information: a
season record is the sort of thing you would look up before tip-off.

Every measure here is computed from games strictly BEFORE the game in question.
`_prior_stats` shifts by one row per team so that a game never contributes to its
own feature, and the tests assert that a game's own result and every later game
are excluded.

Is using earlier games from the TEST season itself a leak?
----------------------------------------------------------
No, and the distinction matters. Leave-one-season-out means the MODEL never
trains on the held-out season. But a feature that says "this opponent is 12-4 so
far this season" uses only information that genuinely existed before tip-off, and
would be available to a live system. Refusing it would make the evaluation
unrealistically pessimistic rather than more honest.

What would be a leak is fitting the feature's parameters on test data. The
shrinkage constant here is fixed in advance, not tuned.

Early-season sample sizes
-------------------------
A team that is 1-0 is not the best team in the league. Every rate is shrunk
toward the league mean by games played:

    value = raw * n / (n + K)

with K = 10 games. After 10 games a team's record counts for half; after 40, for
80 percent. `opponent_games_played_prior` is exposed as a feature too, so the
model can learn to discount early-season measures on its own.

Margins come from the two team rows per game, not from PLUS_MINUS
----------------------------------------------------------------
Phase 1 established that the game log's PLUS_MINUS column is unreliable: ten of
636 Celtics games had player plus/minus that did not reconcile, and five carried
impossible fractional team values. Each game appears twice in the league log, once
per team, so the true margin is computable by joining a game to itself. That is
what is done here.

Output
------
data/interim/opponent_strength.csv, one row per Celtics game.
"""

import logging

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

# Shrinkage toward the league mean, in games.
SHRINKAGE_GAMES = 10.0

# Window for the recent-form measure.
RECENT_WINDOW = 10

# MODEL-FACING. Every one of these is SHRUNK toward a neutral centre by
# SHRINKAGE_GAMES, which is right for a feature: five games is not enough
# evidence to believe a raw differential. It also means none of them is a
# statement about what a team actually did.
OPPONENT_FEATURE_COLUMNS = [
    "opponent_win_pct_prior",
    "opponent_point_diff_prior",
    "opponent_recent_form",
    "opponent_games_played_prior",
    "celtics_point_diff_prior",
    "strength_diff_prior",
]

# DISPLAY-FACING, and deliberately a separate list. These are the unshrunk
# season-to-date figures: what the team's record and scoring margin genuinely
# were before this game.
#
# They exist because the dashboard was reading the shrunk columns and narrating
# them as fact. On 30 Oct 2021 the opponent panel said Washington "were 1.5
# points per game on the season to that date". Washington had actually outscored
# opponents by 4.4 a game and were 4-1; 1.5 is that 4.4 pulled two thirds of the
# way to zero because only five games had been played. The estimate was correct
# and the sentence was not.
#
# NEVER add these to OPPONENT_FEATURE_COLUMNS. Unshrunk rates over a handful of
# games are exactly the noisy quantity the shrinkage exists to control, and a
# model fitted on them would be learning small-sample noise.
OPPONENT_DISPLAY_COLUMNS = [
    "opponent_win_pct_prior_raw",
    "opponent_point_diff_prior_raw",
    "celtics_point_diff_prior_raw",
    "celtics_games_played_prior",
]


def load_league_logs() -> pd.DataFrame:
    path = config.RAW_DIR / "league_game_logs.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/13_pull_league_games.py first.")
    df = pd.read_csv(path, parse_dates=["GAME_DATE"], dtype={"GAME_ID": str})
    df["GAME_ID"] = df["GAME_ID"].str.zfill(10)
    return df


def add_true_margins(logs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute each team-row's true margin by joining the game to its other row.

    Deliberately avoids PLUS_MINUS. See the module docstring.
    """
    logs = logs.copy()
    totals = logs.groupby("GAME_ID")["PTS"].transform("sum")
    counts = logs.groupby("GAME_ID")["PTS"].transform("size")
    if not (counts == 2).all():
        bad = logs.loc[counts != 2, "GAME_ID"].unique()[:10]
        raise ValueError(
            f"{int((counts != 2).sum())} team-rows belong to games without "
            f"exactly two rows, so margins cannot be computed. Examples: {bad}")
    logs["opponent_pts"] = totals - logs["PTS"]
    logs["margin"] = logs["PTS"] - logs["opponent_pts"]
    return logs


def _prior_stats(logs: pd.DataFrame) -> pd.DataFrame:
    """
    Per team-row, summarise that team's PREVIOUS games in the same season.

    The shift(1) is the whole point: without it a game contributes to its own
    feature, which is leakage of the most direct kind.
    """
    logs = logs.sort_values(["SEASON", "TEAM_ABBREVIATION", "GAME_DATE",
                             "GAME_ID"]).copy()
    grouped = logs.groupby(["SEASON", "TEAM_ABBREVIATION"], sort=False)

    # Expanding means over games BEFORE this one.
    logs["games_played_prior"] = grouped.cumcount()
    logs["win_pct_prior_raw"] = (
        grouped["WON"].apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=[0, 1], drop=True))
    logs["point_diff_prior_raw"] = (
        grouped["margin"].apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=[0, 1], drop=True))
    logs["recent_form_raw"] = (
        grouped["margin"].apply(
            lambda s: s.shift(1).rolling(RECENT_WINDOW, min_periods=1).mean())
        .reset_index(level=[0, 1], drop=True))
    return logs


def _shrink(value, games, centre, k=SHRINKAGE_GAMES):
    """Shrink a rate toward `centre` by games played."""
    value = pd.Series(value).astype(float)
    games = pd.Series(games).astype(float)
    weight = games / (games + k)
    return (centre + weight.to_numpy() * (value.fillna(centre).to_numpy()
                                          - centre))


def build_opponent_strength(game_index: pd.DataFrame = None) -> pd.DataFrame:
    """
    One row per Celtics game with pregame, as-of-date opponent measures.
    """
    logs = add_true_margins(load_league_logs())
    logs = _prior_stats(logs)

    logs["win_pct_prior"] = _shrink(logs["win_pct_prior_raw"],
                                    logs["games_played_prior"], 0.5)
    logs["point_diff_prior"] = _shrink(logs["point_diff_prior_raw"],
                                       logs["games_played_prior"], 0.0)
    logs["recent_form"] = _shrink(
        logs["recent_form_raw"],
        logs["games_played_prior"].clip(upper=RECENT_WINDOW), 0.0)

    if game_index is None:
        game_index = pd.read_csv(config.GAME_INDEX_CSV, dtype={"GAME_ID": str},
                                 parse_dates=["GAME_DATE"])
        game_index["GAME_ID"] = game_index["GAME_ID"].str.zfill(10)

    keep = ["GAME_ID", "TEAM_ABBREVIATION", "games_played_prior",
            "win_pct_prior", "point_diff_prior", "recent_form",
            # The unshrunk figures travel alongside, for display only.
            "win_pct_prior_raw", "point_diff_prior_raw"]
    slim = logs[keep]

    opponent = slim.rename(columns={
        "TEAM_ABBREVIATION": "OPPONENT_ABBREV",
        "games_played_prior": "opponent_games_played_prior",
        "win_pct_prior": "opponent_win_pct_prior",
        "point_diff_prior": "opponent_point_diff_prior",
        "recent_form": "opponent_recent_form",
        "win_pct_prior_raw": "opponent_win_pct_prior_raw",
        "point_diff_prior_raw": "opponent_point_diff_prior_raw",
    })
    celtics = slim.loc[slim["TEAM_ABBREVIATION"].eq(config.CELTICS_ABBREV)].rename(
        columns={
            "games_played_prior": "celtics_games_played_prior",
            "win_pct_prior": "celtics_win_pct_prior",
            "point_diff_prior": "celtics_point_diff_prior",
            "recent_form": "celtics_recent_form",
            "win_pct_prior_raw": "celtics_win_pct_prior_raw",
            "point_diff_prior_raw": "celtics_point_diff_prior_raw",
        }).drop(columns=["TEAM_ABBREVIATION"])

    merged = (game_index
              .merge(opponent, on=["GAME_ID", "OPPONENT_ABBREV"], how="left",
                     validate="one_to_one")
              .merge(celtics, on="GAME_ID", how="left", validate="one_to_one"))

    merged["strength_diff_prior"] = (merged["celtics_point_diff_prior"]
                                     - merged["opponent_point_diff_prior"])

    unmatched = merged["opponent_point_diff_prior"].isna().sum()
    if unmatched:
        logger.warning("%d Celtics game(s) could not be matched to a league "
                       "log row for the opponent", unmatched)

    columns = (["GAME_ID", "GAME_DATE", "SEASON", "OPPONENT_ABBREV"]
               + OPPONENT_FEATURE_COLUMNS
               + OPPONENT_DISPLAY_COLUMNS
               + ["celtics_win_pct_prior", "celtics_recent_form"])
    return merged[columns].sort_values(["GAME_DATE", "GAME_ID"]).reset_index(
        drop=True)


def attach_opponent_strength(events: pd.DataFrame,
                             strength: pd.DataFrame) -> pd.DataFrame:
    """Join the per-game opponent measures onto the event table."""
    columns = ["GAME_ID"] + OPPONENT_FEATURE_COLUMNS
    slim = strength[columns].rename(columns={"GAME_ID": "game_id"})
    merged = events.merge(slim, on="game_id", how="left", validate="many_to_one")
    return merged
