"""
Roster and starter extraction from cached boxscores.

Produces one row per player per game per team, for BOTH teams. The opponent rows
are kept deliberately: the project's matchup context needs opponent rosters,
positions and photos, and discarding them here would mean re-downloading 636
games later.

Starter detection
-----------------
Uses ROW ORDER, not the `position` field. The reason is measured, not assumed:

  From 2017-18 onward exactly five players per team carry a `position` value, so
  that field does identify starters. In 2016-17 it is populated for NINE OR TEN
  players per team and is useless for the purpose.

  Row order was cross-checked against it. Across every team-game in the
  development sample where `position` is populated with exactly five players, the
  first five roster rows are exactly the position-flagged set, with no exceptions.

So row order is the season-independent mechanism, and `starter_flag_agrees` is
carried on every row so the full 636 game run can re-verify the agreement rather
than take it on trust.

Position values here are only the coarse buckets G, F and C. Granular
PG/SG/SF/PF/C positions are not available from this endpoint and need a separate
source, which is a later phase.
"""

import json
import logging

import pandas as pd

from src import config

logger = logging.getLogger(__name__)

N_STARTERS = 5

# Statistics fields lifted onto the roster row. Everything else stays in the raw
# payload, which is never deleted.
STAT_FIELDS = [
    "minutes", "points", "plusMinusPoints",
    "fieldGoalsMade", "fieldGoalsAttempted",
    "threePointersMade", "threePointersAttempted",
    "freeThrowsMade", "freeThrowsAttempted",
    "reboundsOffensive", "reboundsDefensive", "reboundsTotal",
    "assists", "steals", "blocks", "turnovers", "foulsPersonal",
]


def parse_minutes(value) -> float:
    """
    Convert a boxscore minutes string to a float number of minutes.

    Observed format is "29:11". An empty string means the player did not play,
    which is returned as 0.0 rather than NaN so minutes can be summed safely.
    Anything unrecognised raises, because silently zeroing a real value would
    understate a player's time on court.
    """
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    # Some feeds use the ISO duration form. Handle it rather than fail.
    if text.startswith("PT"):
        import re
        match = re.fullmatch(r"PT(\d+)M([\d.]+)S", text)
        if not match:
            raise ValueError(f"unrecognised minutes value {value!r}")
        return int(match.group(1)) + float(match.group(2)) / 60.0
    if ":" in text:
        mins, secs = text.split(":", 1)
        return int(mins) + float(secs) / 60.0
    return float(text)


def load_boxscore(game_id: str) -> dict:
    path = config.RAW_BOX_DIR / f"{game_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"no cached boxscore for {game_id}: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)["boxScoreTraditional"]


def roster_rows_for_game(game_id: str, index_row) -> list:
    """Build roster rows for one game, both teams."""
    box = load_boxscore(game_id)
    rows = []

    for side in ("homeTeam", "awayTeam"):
        team = box[side]
        players = team["players"]
        tricode = team.get("teamTricode")
        team_id = int(team.get("teamId") or 0)
        is_home = side == "homeTeam"

        # Row order defines starters.
        starter_ids = {int(p["personId"]) for p in players[:N_STARTERS]}
        # The position field, for cross-checking only.
        position_ids = {int(p["personId"]) for p in players
                        if str(p.get("position") or "").strip()}
        position_usable = len(position_ids) == N_STARTERS
        agrees = (starter_ids == position_ids) if position_usable else None

        for order, player in enumerate(players):
            stats = player.get("statistics") or {}
            person_id = int(player["personId"])
            row = {
                "game_id": game_id,
                "season": index_row["SEASON"],
                "game_date": index_row["GAME_DATE"],
                "team_id": team_id,
                "team_tricode": tricode,
                "is_home_team": is_home,
                "is_celtics_team": tricode == config.CELTICS_ABBREV,
                "opponent_tricode": index_row["OPPONENT_ABBREV"],
                "roster_order": order,
                "person_id": person_id,
                "first_name": player.get("firstName") or "",
                "family_name": player.get("familyName") or "",
                "name_initial": player.get("nameI") or "",
                "player_slug": player.get("playerSlug") or "",
                "coarse_position": (player.get("position") or "").strip(),
                "jersey_number": (player.get("jerseyNum") or "").strip(),
                "comment": (player.get("comment") or "").strip(),
                "is_starter": person_id in starter_ids,
                "position_field_usable": position_usable,
                "starter_flag_agrees": agrees,
                "minutes_played": parse_minutes(stats.get("minutes")),
            }
            for field in STAT_FIELDS:
                if field == "minutes":
                    continue
                row[field] = stats.get(field)
            rows.append(row)

    return rows


def build_rosters(game_ids=None) -> pd.DataFrame:
    """Build the roster table across games. Defaults to every indexed game."""
    index = pd.read_csv(config.GAME_INDEX_CSV, dtype={"GAME_ID": str})
    index["GAME_ID"] = index["GAME_ID"].str.zfill(10)
    index = index.set_index("GAME_ID")

    if game_ids is None:
        game_ids = sorted(
            p.stem for p in config.RAW_BOX_DIR.glob("*.json")
        )

    all_rows = []
    for game_id in game_ids:
        if game_id not in index.index:
            raise KeyError(f"{game_id} has a boxscore but is not in the game index")
        all_rows.extend(roster_rows_for_game(game_id, index.loc[game_id]))

    df = pd.DataFrame(all_rows)
    return df.sort_values(["game_date", "game_id", "team_tricode",
                           "roster_order"]).reset_index(drop=True)


def starters_by_team(df: pd.DataFrame, game_id: str) -> dict:
    """{team_tricode: set of starter personIds} for one game."""
    game = df.loc[df["game_id"].eq(game_id) & df["is_starter"]]
    return {tri: set(grp["person_id"])
            for tri, grp in game.groupby("team_tricode")}
