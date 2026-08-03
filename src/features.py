"""
Phase 3: live game-state features from the validated event table.

Everything here is computed WITHIN a single game from information available at
that moment. Nothing in this module looks at the final result, at later events,
or at any cross-game aggregate. Features that require aggregation across games,
notably lineup strength and opponent strength, are deliberately NOT here: they
must be computed inside cross-validation folds, and mixing them in would make it
easy to leak by accident. See src/lineup_strength.py.

Features produced
-----------------
Required by the research plan:
  celtics_margin              already on the event table, carried through
  seconds_remaining_period    time left in the current period
  seconds_remaining_game      time left, treating overtime as possibly final
  period, is_overtime         where in the game we are
  celtics_is_home             home court
  celtics_has_possession      who has the ball, from an explicit state machine
  momentum_120s / 300s        recent scoring run, in Celtics-perspective points
  is_clutch                   NBA clutch definition, stated explicitly

Supporting:
  possession_number           possessions elapsed in the game
  score_change                points scored on this event, Celtics perspective
  margin_per_minute_remaining pressure: how much margin per remaining minute

Possession
----------
The feed has no possession field, so it is derived. Rules, applied in delivered
event order:

  A team demonstrably has the ball when it attempts a shot, attempts a free
  throw, or commits a turnover. A team gains the ball when it secures a rebound.
  A foul is committed by the DEFENDING team, so it implies the other team has the
  ball, which is useful because fouls often occur where nothing else marks
  possession.

  Substitutions, timeouts, period markers, replays and the blank-actionType
  block/steal annotations never change possession.

Jump balls are left unchanged rather than parsed. The tip winner appears only in
free text ("Tip to Irving") and there are 29 jump balls in an 8,000 event sample,
so guessing there would add more risk than it removes. Period-opening possession
is therefore inherited until the first possession-bearing event, which is
typically within a few seconds.

The sanity checks on this are in src/validate_phase3.py: possession share should
sit near 50 percent, and possessions per team per game should land in the normal
NBA range of roughly 90 to 110.
"""

import logging

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

# Action types that reveal which team has the ball.
OFFENSIVE_ACTIONS = {"Made Shot", "Missed Shot", "Turnover", "Free Throw"}
GAIN_ACTIONS = {"Rebound"}
DEFENSIVE_ACTIONS = {"Foul"}

# NBA's clutch definition: last five minutes of the fourth period or any
# overtime, with the score within five points.
CLUTCH_SECONDS = 300
CLUTCH_MARGIN = 5

MOMENTUM_WINDOWS = (120, 300)

# Features that are FUNCTIONS of other features rather than independent facts
# about the game. Named here so anything editing a feature vector can ask which
# columns it is not allowed to leave alone.
DERIVED_FEATURES = ("is_clutch", "margin_per_minute_remaining")

# What each derived feature is computed FROM, so a recomputation can be
# explained to a caller and skipped when nothing it depends on moved.
DERIVED_INPUTS = {
    "is_clutch": ("period", "seconds_remaining_period", "celtics_margin"),
    "margin_per_minute_remaining": ("celtics_margin", "seconds_remaining_game"),
}


def recompute_derived(frame: pd.DataFrame, skip=()) -> pd.DataFrame:
    """
    Rebuild the features that are defined in terms of other features.

    THE ONLY definition of these two columns. `build_features` calls it while
    constructing the training frame and the what-if endpoint calls it after
    replacing a value, so a hand-edited feature vector is put back onto the
    manifold the model was fitted on rather than being handed a state that
    cannot physically occur.

    This exists because it did not. The what-if endpoint replaced
    `celtics_margin` and left `margin_per_minute_remaining` and `is_clutch`
    holding values computed from the ORIGINAL margin. The model then saw a row
    claiming Boston were 32 points down while the pressure feature still said
    12 down, which is not a game state, and it answered accordingly: on one real
    second-quarter event, moving the margin from -12 to -32 RAISED the win
    probability from 31.9% to 32.5%, and asking for +20 returned 21.8%.
    Recomputing turns the same sweep into 80.2% at +20 and 3.6% at -40.

    `skip` names columns the caller overrode ON PURPOSE. Silently overwriting an
    explicit instruction would be its own bug, so an explicit override wins and
    the caller is told which columns were rebuilt and which were left.
    """
    out = frame.copy()

    if "margin_per_minute_remaining" not in skip:
        minutes_left = (out["seconds_remaining_game"] / 60.0).clip(lower=1 / 60)
        out["margin_per_minute_remaining"] = out["celtics_margin"] / minutes_left

    if "is_clutch" not in skip:
        out["is_clutch"] = (
            (out["period"] >= 4)
            & (out["seconds_remaining_period"] <= CLUTCH_SECONDS)
            & (out["celtics_margin"].abs() <= CLUTCH_MARGIN)
        )

    return out


def assign_possession(game: pd.DataFrame, celtics_tricode=None) -> pd.Series:
    """
    Return the tricode of the team holding the ball at each event.

    Operates in delivered event order. Returns an object Series aligned to the
    input index, with empty string before the first possession-bearing event.
    """
    celtics_tricode = celtics_tricode or config.CELTICS_ABBREV
    tricodes = [t for t in game["team_tricode"].unique() if t]
    opponent = next((t for t in tricodes if t != celtics_tricode), "")

    possession = []
    current = ""
    for action_type, tricode in zip(game["action_type"], game["team_tricode"]):
        if tricode:
            if action_type in OFFENSIVE_ACTIONS or action_type in GAIN_ACTIONS:
                current = tricode
            elif action_type in DEFENSIVE_ACTIONS:
                # A foul is committed by the defence, so the other team has it.
                other = opponent if tricode == celtics_tricode else celtics_tricode
                if other:
                    current = other
        possession.append(current)
    return pd.Series(possession, index=game.index, dtype=object)


def rolling_momentum(seconds: np.ndarray, celtics_points: np.ndarray,
                     opponent_points: np.ndarray, window: float) -> np.ndarray:
    """
    Celtics points minus opponent points scored within the last `window` seconds.

    Uses cumulative sums and a binary search for the window start, so it is O(n
    log n) rather than a nested loop over 300,000 events.
    """
    celtics_cumulative = np.concatenate([[0.0], np.cumsum(celtics_points)])
    opponent_cumulative = np.concatenate([[0.0], np.cumsum(opponent_points)])
    # First index whose timestamp is >= (now - window).
    start = np.searchsorted(seconds, seconds - window, side="left")
    index = np.arange(len(seconds)) + 1
    celtics_window = celtics_cumulative[index] - celtics_cumulative[start]
    opponent_window = opponent_cumulative[index] - opponent_cumulative[start]
    return celtics_window - opponent_window


def seconds_remaining_game(period: pd.Series,
                           seconds_remaining_period: pd.Series,
                           seconds_elapsed: pd.Series) -> pd.Series:
    """
    Time left in the game.

    In regulation this is time to the end of the fourth period. In overtime the
    game may end when the period does, so the remaining time in the current
    overtime is the honest answer. A model should not be told an overtime game
    has negative time left.
    """
    regulation_remaining = (config.REGULATION_SECONDS_TOTAL - seconds_elapsed
                            if hasattr(config, "REGULATION_SECONDS_TOTAL")
                            else 2880 - seconds_elapsed)
    return pd.Series(
        np.where(period <= 4, regulation_remaining.clip(lower=0),
                 seconds_remaining_period),
        index=period.index, dtype=float)


def build_features(events: pd.DataFrame) -> pd.DataFrame:
    """Add live game-state features to the parsed event table."""
    frames = []

    for game_id, game in events.groupby("game_id", sort=False):
        game = game.sort_values("event_index").copy()

        # --- possession ---
        possession = assign_possession(game)
        game["possession_team"] = possession
        game["celtics_has_possession"] = (
            possession.eq(config.CELTICS_ABBREV))
        # A possession change is any transition between two non-empty holders.
        changed = (possession.ne(possession.shift())
                   & possession.ne("")
                   & possession.shift().ne(""))
        game["possession_change"] = changed
        game["possession_number"] = changed.cumsum()

        # --- scoring, for momentum ---
        celtics_delta = game["celtics_score"].diff().fillna(
            game["celtics_score"].iloc[0]).clip(lower=0)
        opponent_delta = game["opponent_score"].diff().fillna(
            game["opponent_score"].iloc[0]).clip(lower=0)
        game["score_change"] = celtics_delta - opponent_delta

        seconds = game["seconds_elapsed_game"].to_numpy(dtype=float)
        for window in MOMENTUM_WINDOWS:
            game[f"momentum_{window}s"] = rolling_momentum(
                seconds, celtics_delta.to_numpy(dtype=float),
                opponent_delta.to_numpy(dtype=float), float(window))

        # --- time ---
        game["seconds_remaining_game"] = seconds_remaining_game(
            game["period"], game["seconds_remaining_period"],
            game["seconds_elapsed_game"])

        # --- clutch and pressure ---
        # Both are functions of columns already present, and both are defined in
        # recompute_derived so that the training frame and any later edit to a
        # feature vector cannot drift apart.
        game = recompute_derived(game)

        frames.append(game)

    out = pd.concat(frames, ignore_index=True)
    return out


FEATURE_COLUMNS = [
    "celtics_margin",
    "seconds_remaining_period",
    "seconds_remaining_game",
    "seconds_elapsed_game",
    "period",
    "is_overtime",
    "celtics_is_home",
    "celtics_has_possession",
    "momentum_120s",
    "momentum_300s",
    "is_clutch",
    "margin_per_minute_remaining",
    "possession_number",
]

TARGET_COLUMN = "celtics_won"
GROUP_COLUMN = "game_id"
SEASON_COLUMN = "season"
