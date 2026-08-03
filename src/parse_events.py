"""
Parse raw PlayByPlayV3 payloads into one canonical event table.

Hazards this handles, all found by inspecting the real payloads
---------------------------------------------------------------
`scoreHome` and `scoreAway` are STRINGS, and how a non-scoring event encodes
"no score here" CHANGES BETWEEN SEASONS. Measured across all 636 games:

    season     empty      "0"
    2016-17    2,910   27,462     <- zero-encoded
    2017-18   29,630      167
    2018-19   30,106      152
    2019-20   27,112      121
    2020-21   26,152      131
    2021-22   29,547      165
    2022-23   28,866      145
    2023-24   28,336      146

In 2016-17 the score field on a rebound, foul or substitution is the string "0",
not "". A naive forward fill therefore accepts 0 as a real score and resets the
game to 0-0 over and over. In game 0021600006 that corrupted the margin on 74%
of events. Score margin is the most important feature in this model, so an entire
season of it would have been wrong while every total still looked plausible.

Separately, a few TRAILING events report a stale score. In game 0022301202,
Hield hits a three at 0.3 seconds to make it 122-112, and then an "Instant
Replay" marker and the period-end row both report the pre-shot 119-112.

Both problems have the same shape: a reported score that is BEHIND the true one.
So the score is reconstructed as MONOTONE NON-DECREASING, taking the componentwise
maximum of what has been reported so far. A basketball score cannot go down.

The guard on that rule is check 4 in src/validate_phase2.py: the final
reconstructed score must equal the boxscore point totals for all 636 games. If a
basket were ever legitimately voided on review, that check would catch it.

Blocks and steals are separate events with a BLANK `actionType`. There were 411
of them in a 16 game sample. Each attaches to a shot or turnover that is already
its own event, so they are marked `is_secondary_event` and must not be treated as
possession changes.

Clocks are ISO 8601 durations with hundredths, "PT06M47.00S".

Rebound `subType` is `Unknown` on 1,680 of 1,766 rebounds, so it cannot be used
to tell offensive from defensive rebounds. That distinction has to come from
comparing the rebounding team against the team that shot, which is a Phase 3
concern, so no such column is invented here.

Margin is computed from BOSTON's perspective throughout, not raw home minus away,
because the model predicts whether Boston wins.

Output
------
data/interim/events.parquet
"""

import json
import logging
import re

import pandas as pd

from src import config

logger = logging.getLogger(__name__)

CLOCK_PATTERN = re.compile(r"^PT(?P<m>\d+)M(?P<s>[\d.]+)S$")

REGULATION_PERIODS = 4
REGULATION_PERIOD_SECONDS = 12 * 60      # 720
OVERTIME_PERIOD_SECONDS = 5 * 60         # 300
REGULATION_SECONDS = REGULATION_PERIODS * REGULATION_PERIOD_SECONDS  # 2880

# Events whose actionType is blank. Verified to be blocks and steals, which are
# annotations on another event rather than events that change possession.
SECONDARY_ACTION_TYPE = ""


def parse_clock(value) -> float:
    """
    ISO 8601 duration to seconds remaining in the period. "PT06M47.00S" -> 407.0

    Raises on anything unrecognised. A silent default here would corrupt the time
    remaining feature, which is the second most important input to the model
    after score margin.
    """
    text = str(value or "").strip()
    if not text:
        raise ValueError("empty clock value")
    match = CLOCK_PATTERN.match(text)
    if not match:
        raise ValueError(f"unrecognised clock format {value!r}")
    return int(match.group("m")) * 60 + float(match.group("s"))


def period_length(period: int) -> int:
    """Seconds in a given period. Regulation is 12 minutes, overtime is 5."""
    return (REGULATION_PERIOD_SECONDS if period <= REGULATION_PERIODS
            else OVERTIME_PERIOD_SECONDS)


def seconds_elapsed(period: int, clock_remaining: float) -> float:
    """Total seconds of game clock played at this moment."""
    completed = sum(period_length(p) for p in range(1, period))
    return completed + (period_length(period) - clock_remaining)


def _to_int_score(value):
    """Return an int score, or None when the field is empty."""
    text = str(value if value is not None else "").strip()
    if not text:
        return None
    return int(text)


def load_actions(game_id: str) -> list:
    path = config.RAW_PBP_DIR / f"{game_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"no cached play-by-play for {game_id}: {path}")
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    actions = (payload.get("game") or {}).get("actions")
    if not actions:
        raise ValueError(f"{game_id} has no game.actions in its payload")
    return actions


def parse_game(game_id: str, index_row) -> pd.DataFrame:
    """
    Parse one game's events into a tidy frame.

    ORDERING. The payload's own order is authoritative and is preserved exactly.
    An earlier version of this function sorted by (period, actionNumber), which
    was a real bug with real consequences: `actionNumber` is NOT unique within a
    game. In game 0021600006, 521 events contain duplicate action numbers, for
    example a Turnover and its associated STEAL both numbered 370. Sorting on a
    non-unique key silently reordered events, which made the game clock
    non-monotonic and corrupted every downstream lineup and duration.

    Measured on the same game: the delivered order has ZERO clock inversions
    within a period, while the sorted order had four. So the fix is to trust the
    feed's sequence and record it as `event_index`.
    """
    actions = load_actions(game_id)

    is_home = bool(index_row["IS_HOME"])
    rows = []
    last_home, last_away = 0, 0

    for event_index, action in enumerate(actions):
        period = int(action.get("period") or 0)
        clock_remaining = parse_clock(action.get("clock"))

        home = _to_int_score(action.get("scoreHome"))
        away = _to_int_score(action.get("scoreAway"))
        score_reported = home is not None and away is not None
        score_stale = False
        if score_reported:
            # Monotone non-decreasing. A report below the running score is either
            # 2016-17 zero-encoding or a stale trailing snapshot; either way it is
            # not the current score. See the module docstring.
            if home < last_home or away < last_away:
                score_stale = True
            last_home = max(last_home, home)
            last_away = max(last_away, away)

        celtics_score = last_home if is_home else last_away
        opponent_score = last_away if is_home else last_home

        elapsed = seconds_elapsed(period, clock_remaining)
        action_type = action.get("actionType") or ""

        rows.append({
            "game_id": game_id,
            "season": index_row["SEASON"],
            "game_date": index_row["GAME_DATE"],
            "opponent_tricode": index_row["OPPONENT_ABBREV"],
            "celtics_is_home": is_home,
            "celtics_won": int(index_row["CELTICS_WON"]),

            "event_index": event_index,
            "action_number": int(action.get("actionNumber") or 0),
            "period": period,
            "is_overtime": period > REGULATION_PERIODS,
            "clock_raw": action.get("clock"),
            "seconds_remaining_period": clock_remaining,
            "seconds_elapsed_game": elapsed,
            "seconds_remaining_regulation": max(0.0, REGULATION_SECONDS - elapsed),

            "action_type": action_type,
            "sub_type": action.get("subType") or "",
            "description": action.get("description") or "",
            "is_secondary_event": action_type == SECONDARY_ACTION_TYPE,

            "team_id": int(action.get("teamId") or 0),
            "team_tricode": action.get("teamTricode") or "",
            "person_id": int(action.get("personId") or 0),
            "player_name": action.get("playerName") or "",
            "player_name_initial": action.get("playerNameI") or "",

            "score_home": last_home,
            "score_away": last_away,
            "score_reported": score_reported,
            "score_report_stale": score_stale,
            "celtics_score": celtics_score,
            "opponent_score": opponent_score,
            "celtics_margin": celtics_score - opponent_score,

            "shot_result": action.get("shotResult") or "",
            "shot_value": action.get("shotValue") or 0,
            "shot_distance": action.get("shotDistance") or 0,
            "is_field_goal": int(action.get("isFieldGoal") or 0),
            "points_total": action.get("pointsTotal") or 0,
            "loc_x": action.get("xLegacy"),
            "loc_y": action.get("yLegacy"),
        })

    frame = pd.DataFrame(rows)

    # Guard the assumption this function now depends on. If the feed order is
    # ever non-monotonic in elapsed game time, downstream durations and lineups
    # are meaningless, so it must surface loudly rather than be absorbed.
    elapsed = frame["seconds_elapsed_game"].to_numpy()
    inversions = int((elapsed[1:] < elapsed[:-1] - 1e-6).sum())
    frame.attrs["clock_inversions"] = inversions
    frame.attrs["stale_score_reports"] = int(frame["score_report_stale"].sum())
    if inversions:
        logger.warning("%s has %d clock inversions in delivered order",
                       game_id, inversions)
    return frame


def build_events(game_ids=None) -> pd.DataFrame:
    """Parse every cached game, or a given subset."""
    index = pd.read_csv(config.GAME_INDEX_CSV, dtype={"GAME_ID": str},
                        parse_dates=["GAME_DATE"])
    index["GAME_ID"] = index["GAME_ID"].str.zfill(10)
    index["IS_HOME"] = index["IS_HOME"].astype(bool)
    index = index.set_index("GAME_ID")

    if game_ids is None:
        game_ids = sorted(p.stem for p in config.RAW_PBP_DIR.glob("*.json"))

    frames = []
    for n, game_id in enumerate(game_ids, start=1):
        if game_id not in index.index:
            raise KeyError(f"{game_id} has play-by-play but is not in the index")
        frames.append(parse_game(game_id, index.loc[game_id]))
        if n % 100 == 0:
            logger.info("parsed %d/%d games", n, len(game_ids))

    events = pd.concat(frames, ignore_index=True)
    # Sort on event_index, never action_number: action_number is not unique.
    return events.sort_values(["game_date", "game_id", "event_index"]
                              ).reset_index(drop=True)
