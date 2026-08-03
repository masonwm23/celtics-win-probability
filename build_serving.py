"""
Phase 9b: build the JSON the dashboard reads.

WHAT THIS IS
------------
A precompute step. It joins the five tables the API needs and writes one compact
JSON file per game, plus an index. The API then serves files rather than
recomputing joins on every request, which keeps the backend simple enough to
audit and fast enough to scrub through a game in real time.

THE PROBABILITY SHOWN IS OUT OF FOLD
------------------------------------
This is the most important decision in the file. The saved deployment model in
models/ was fitted on ALL eight seasons, so asking it to predict any game in the
dataset is in-sample. It would look better and mean less.

Instead the dashboard replays `tier3_celtics` from `oof_predictions.parquet`.
Every probability on screen comes from a model that never saw that game's
season. `tier2_generic` is carried alongside so the interface can show the
generic baseline next to it, which is the paper's headline comparison made
visible.

THE JOIN KEY TRAP
-----------------
`game_index.csv` and `opponent_strength.csv` store GAME_ID as an integer
(21600006). Every parquet stores it as a zero-padded string ("0021600006").
Joining those directly returns zero rows and raises nothing. Everything is
normalised through `normalise_game_id` before any merge, and the row counts are
asserted afterwards.

PLAYER INFORMATION, AND WHAT IS MISSING
---------------------------------------
Cards are assembled from three sources, in this order of preference:

  1. `player_bios.csv`  granular position (G-F, F-C) and height
  2. `rosters.parquet`  jersey number, coarse position, minutes, box score
  3. `person_id` alone  the headshot URL, which needs nothing else

187 player-seasons have no bio row, so those cards lose height and the granular
position. They keep a photo, a name, a jersey number and a coarse position.
Nothing is filled in with a guess, and `bio_coverage_by_minutes` measures how
many MINUTES those players actually played, because "5% of players" and "5% of
minutes" are very different problems for a dashboard.

Output
------
data/serving/index.json          one row per game, for the game picker
data/serving/games/{id}.json     per-game timeline, players and lineups
data/serving/coverage.json       what is missing, measured rather than assumed
"""

import json
import logging
import re
from collections import OrderedDict

import numpy as np
import pandas as pd

from src import config
from src.opponent_strength import SHRINKAGE_GAMES

logger = logging.getLogger(__name__)

# Which out-of-fold column drives the dashboard, and which sits beside it.
PRIMARY_TIER = "tier3_celtics"
BASELINE_TIER = "tier2_generic"

# The CURRENT photo: whatever jersey the player wears today. Used as a
# fallback, never as the first choice, because on a 2017-18 game it puts Kyrie
# Irving in a Mavericks shirt.
HEADSHOT_TEMPLATE = ("https://cdn.nba.com/headshots/nba/latest/1040x760/"
                     "{person_id}.png")
TEAM_LOGO_TEMPLATE = ("https://cdn.nba.com/logos/nba/{team_id}/primary/L/"
                      "logo.svg")

# The SEASON photo comes from `data/processed/season_headshots.csv`, which
# scripts/39_build_season_headshots.py builds by actually fetching every
# player-season and recording what came back. Nothing here constructs a URL:
# if a row was not fetched and confirmed to be a real image, the player gets
# the current photo instead. 3,680 of 4,009 player-seasons were confirmed.
SEASON_HEADSHOT_CSV = "season_headshots.csv"


def load_season_headshots() -> pd.DataFrame:
    """The verified season-photo map, or an empty frame if it was never built."""
    path = config.PROCESSED_DIR / SEASON_HEADSHOT_CSV
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def team_names(bios) -> dict:
    """
    Tricode to full team name, from the bios file.

    The dashboard needs "Boston Celtics defeat Denver Nuggets" at the end of a
    game and the payload previously carried only tricodes. Read from the data
    rather than typed out here: a hand-written list of thirty team names is a
    list that goes stale and that nothing checks.
    """
    if bios is None or bios.empty:
        return {}
    pairs = bios[["team_abbrev", "team_name"]].drop_duplicates()
    return {str(r.team_abbrev): str(r.team_name) for r in pairs.itertuples()}


def build_headshot_lookup(frame) -> dict:
    """
    Two indexes over the verified map, in priority order.

    `by_team` is keyed by player, season AND team, which is the only key that
    is right for a player traded mid-season: he wore two shirts that year and
    the game's own roster says which one applies here.

    `by_season` drops the team, for the case where the boxscore's tricode and
    the bios file's abbreviation disagree. It deliberately EXCLUDES any player
    who has more than one confirmed row in a season, because for exactly those
    players a team-blind lookup would be a coin flip.
    """
    by_team, by_season, counts = {}, {}, {}
    if frame is None or len(frame) == 0:
        return {"by_team": by_team, "by_season": by_season}

    usable = frame.loc[frame["usable"].astype(bool)]
    for row in usable.itertuples():
        person_id = int(row.person_id)
        season = str(row.season)
        by_team[(person_id, season, str(row.team_abbrev))] = str(row.url)
        key = (person_id, season)
        counts[key] = counts.get(key, 0) + 1
        by_season[key] = str(row.url)

    for key, seen in counts.items():
        if seen > 1:
            by_season.pop(key, None)
    return {"by_team": by_team, "by_season": by_season}


def season_headshot(lookup, person_id: int, season: str, team_abbrev: str):
    """The confirmed season photo for this player in this game, or None."""
    if not lookup:
        return None
    exact = lookup["by_team"].get((int(person_id), str(season),
                                   str(team_abbrev)))
    if exact:
        return exact
    return lookup["by_season"].get((int(person_id), str(season)))


def normalise_jersey(raw) -> str:
    """
    A jersey number as the text it actually is.

    Two things go wrong without this.

    The float artefact: a column containing blanks is typed float64 by pandas,
    so jersey 7 becomes 7.0 and prints as "7.0". Stripping a trailing ".0"
    fixes that.

    The one that matters more: "00" and "0" are different jerseys, worn by
    different players, and once a value has been through float64 they are
    indistinguishable. This function cannot recover that, which is why the read
    itself is done as text. What it does guarantee is that a value that is
    ALREADY text is never mangled: "00" stays "00".
    """
    if raw is None:
        return ""
    if not isinstance(raw, str):
        try:
            if pd.isna(raw):
                return ""
        except (TypeError, ValueError):
            pass
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "<na>", "none"}:
        return ""
    match = re.fullmatch(r"(\d+)\.0+", text)
    return match.group(1) if match else text


def normalise_game_id(values) -> pd.Series:
    """
    Every game id to the same zero-padded 10-character string.

    The single most likely silent failure in this file. See the module
    docstring.
    """
    return (pd.Series(values).astype("string").str.strip()
            .str.replace(r"\.0$", "", regex=True).str.zfill(10))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_tables():
    frame = pd.read_parquet(config.MODEL_FRAME_PARQUET)
    lineups = pd.read_parquet(config.LINEUPS_PARQUET)
    rosters = pd.read_parquet(config.ROSTERS_PARQUET)
    oof = pd.read_parquet(config.PROCESSED_DIR / "oof_predictions.parquet")

    for table in (frame, lineups, rosters, oof):
        table["game_id"] = normalise_game_id(table["game_id"])

    index = pd.read_csv(config.GAME_INDEX_CSV)
    index["game_id"] = normalise_game_id(index["GAME_ID"])

    bios = pd.DataFrame()
    bios_path = config.RAW_DIR / "player_bios.csv"
    if bios_path.exists():
        # `jersey` is read as TEXT deliberately. 14 of the 4,009 rows are
        # blank, so pandas types the column float64 and jersey 7 arrives as
        # 7.0, which reached the dashboard as "7.0" on every player card.
        #
        # Worse than cosmetic: "00" is a real jersey and a DIFFERENT one from
        # "0". Through float64 both become 0.0 and the distinction is gone for
        # good. Reading as text preserves it.
        bios = pd.read_csv(bios_path, dtype={"jersey": "string"})

    values = pd.DataFrame()
    values_path = config.MODELS_DIR / "player_values.csv"
    if values_path.exists():
        values = pd.read_csv(values_path)

    strength = pd.DataFrame()
    strength_path = config.INTERIM_DIR / "opponent_strength.csv"
    if strength_path.exists():
        strength = pd.read_csv(strength_path)
        strength["game_id"] = normalise_game_id(strength["GAME_ID"])

    return frame, lineups, rosters, oof, index, bios, values, strength


def attach_probabilities(frame: pd.DataFrame, oof: pd.DataFrame):
    """
    Join out-of-fold probabilities onto the event table.

    Asserts the row count is unchanged. A merge that silently drops rows would
    show up as a timeline with gaps, which is easy to miss on screen.
    """
    before = len(frame)
    columns = ["game_id", "event_index", PRIMARY_TIER, BASELINE_TIER]
    merged = frame.merge(oof[columns], on=["game_id", "event_index"],
                         how="left", validate="one_to_one")
    if len(merged) != before:
        raise ValueError(
            f"probability join changed the row count: {before} -> {len(merged)}")
    missing = int(merged[PRIMARY_TIER].isna().sum())
    if missing:
        raise ValueError(
            f"{missing} events have no out-of-fold probability. The dashboard "
            "will not display a made-up number in their place.")
    return merged


# ---------------------------------------------------------------------------
# Coverage, measured rather than assumed
# ---------------------------------------------------------------------------

def bio_coverage_by_minutes(rosters: pd.DataFrame,
                            bios: pd.DataFrame) -> pd.DataFrame:
    """
    How much PLAYING TIME belongs to players with no bio row.

    A count of players overstates the problem: a 4-minute call-up and a starter
    count the same. Minutes is the measure that matches what a viewer sees.
    """
    if bios.empty:
        return pd.DataFrame()

    have = set(zip(bios["season"], bios["person_id"]))
    rosters = rosters.copy()
    rosters["has_bio"] = [
        (s, p) in have
        for s, p in zip(rosters["season"], rosters["person_id"])]

    rows = []
    for season, group in rosters.groupby("season"):
        total_minutes = float(group["minutes_played"].sum())
        missing = group.loc[~group["has_bio"]]
        missing_players = missing["person_id"].nunique()
        missing_minutes = float(missing["minutes_played"].sum())
        rows.append({
            "season": season,
            "players": int(group["person_id"].nunique()),
            "players_without_bio": int(missing_players),
            "minutes": round(total_minutes, 1),
            "minutes_without_bio": round(missing_minutes, 1),
            "share_of_minutes": (round(missing_minutes / total_minutes, 5)
                                 if total_minutes else 0.0),
            "median_minutes_of_missing": (
                round(float(missing.groupby("person_id")["minutes_played"]
                            .sum().median()), 1)
                if len(missing) else 0.0),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------

def build_player_table(game_rosters: pd.DataFrame, bios: pd.DataFrame,
                       values: pd.DataFrame, headshots=None) -> dict:
    """
    One card per player in this game, from all available sources.

    Missing fields are emitted as null rather than as a plausible default, so
    the interface can render an honest blank instead of a wrong number.
    """
    season = game_rosters["season"].iloc[0]
    bio_rows = {}
    if not bios.empty:
        subset = bios.loc[bios["season"].eq(season)]
        bio_rows = {int(r.person_id): r for r in subset.itertuples()}

    value_by_id = {}
    if not values.empty:
        value_by_id = dict(zip(values["person_id"].astype(int),
                               values["player_value"].astype(float)))

    players = OrderedDict()
    for row in game_rosters.itertuples():
        person_id = int(row.person_id)
        bio = bio_rows.get(person_id)
        first = (row.first_name or "").strip()
        family = (row.family_name or "").strip()

        # The boxscore's jersey field is blank for whole seasons, so the roster
        # pull's NUM is the fallback. The court shows jersey numbers, and an
        # empty circle would be a worse answer than a number that exists in a
        # second source.
        jersey = normalise_jersey(row.jersey_number)
        if not jersey and bio is not None:
            jersey = normalise_jersey(bio.jersey)

        season_photo = season_headshot(headshots, person_id, season,
                                       row.team_tricode)

        players[str(person_id)] = {
            "person_id": person_id,
            "name": (f"{first} {family}".strip()
                     or (bio.full_name if bio is not None else "")),
            "team": row.team_tricode,
            "is_celtics": bool(row.is_celtics_team),
            "jersey": jersey or None,
            "position": (bio.position if bio is not None else None),
            "coarse_position": (row.coarse_position or None),
            "height": (bio.height if bio is not None else None),
            "height_inches": (float(bio.height_inches)
                              if bio is not None
                              and not pd.isna(bio.height_inches) else None),
            "is_starter": bool(row.is_starter),
            "minutes": round(float(row.minutes_played), 2),
            "points": int(row.points),
            "rebounds": int(row.reboundsTotal),
            "assists": int(row.assists),
            "plus_minus": (None if pd.isna(row.plusMinusPoints)
                           else float(row.plusMinusPoints)),
            "player_value": value_by_id.get(person_id),
            # Season photo when one was confirmed for this player on this team
            # in this season, otherwise the current one. `headshot_current` is
            # always emitted so the interface has something to fall back to if
            # the season image fails to load, and `headshot_is_season` lets it
            # say which of the two it is showing.
            "headshot": (season_photo
                         or HEADSHOT_TEMPLATE.format(person_id=person_id)),
            "headshot_current": HEADSHOT_TEMPLATE.format(person_id=person_id),
            "headshot_is_season": bool(season_photo),
            "has_bio": bio is not None,
        }
    return players


# ---------------------------------------------------------------------------
# Per-game payload
# ---------------------------------------------------------------------------

def split_lineup(text) -> list:
    if not isinstance(text, str) or not text:
        return []
    return [p for p in text.split(",") if p]


def build_game_payload(game_id: str, events: pd.DataFrame,
                       lineups: pd.DataFrame, rosters: pd.DataFrame,
                       bios: pd.DataFrame, values: pd.DataFrame,
                       index_row: pd.Series, strength_row,
                       headshots=None, names=None) -> dict:
    """
    One game, as compact columnar JSON.

    Columnar rather than an array of objects: the same data is roughly a third
    the size and parses faster in the browser, which matters when scrubbing a
    500-event timeline.

    Lineups are stored once in a lookup table with per-event indices into it,
    because a lineup changes perhaps 40 times in a game and repeating five ids
    on every one of 486 events is pure waste.
    """
    events = events.sort_values("event_index").reset_index(drop=True)
    merged = events.merge(
        lineups[["game_id", "event_index", "home_lineup", "away_lineup"]],
        on=["game_id", "event_index"], how="left", validate="one_to_one")
    if len(merged) != len(events):
        raise ValueError(f"lineup join changed row count for {game_id}")
    # A left join does NOT lose rows when the right side is short: it fills
    # them with nulls. So the row count alone is not a sufficient check, and
    # the real symptom of a missing lineup is a null rather than a short frame.
    # Phase 2 established that all 308,975 events carry five players a side, so
    # any null here is a join failure and not a property of the data.
    absent = int(merged[["home_lineup", "away_lineup"]].isna().any(axis=1).sum())
    if absent:
        raise ValueError(
            f"lineup join left {absent} event(s) without a lineup in "
            f"{game_id}. Phase 2 verified every event has five players a "
            "side, so this is a join failure.")

    celtics_home = bool(events["celtics_is_home"].iloc[0])
    celtics_column = "home_lineup" if celtics_home else "away_lineup"
    opponent_column = "away_lineup" if celtics_home else "home_lineup"

    lineup_table, lineup_ids = [], {}

    def lineup_index(text):
        key = text if isinstance(text, str) else ""
        if key not in lineup_ids:
            lineup_ids[key] = len(lineup_table)
            lineup_table.append(split_lineup(key))
        return lineup_ids[key]

    celtics_lineup = [lineup_index(v) for v in merged[celtics_column]]
    opponent_lineup = [lineup_index(v) for v in merged[opponent_column]]

    players = build_player_table(rosters, bios, values, headshots)

    def column(name, cast=None):
        series = merged[name]
        if cast is not None:
            series = series.astype(cast)
        return series.where(series.notna(), None).tolist()

    opponent_context = None
    if strength_row is not None:
        # TWO KINDS OF NUMBER, and the panel that reads this must not blur them.
        #
        # The *_prior values are SHRUNK toward a neutral centre, which is what
        # makes them usable as features and what makes them wrong to narrate:
        # they are estimates, not what a team did. The *_raw values are the
        # actual season-to-date figures and are the ones a reader should be
        # shown. A raw value is None before a team has played, because a record
        # over nothing is not zero, it is undefined.
        def _raw(name):
            value = getattr(strength_row, name, None)
            return None if value is None or pd.isna(value) else float(value)

        opponent_context = {
            # shrunk, model-facing
            "opponent_win_pct_prior": float(strength_row.opponent_win_pct_prior),
            "opponent_point_diff_prior":
                float(strength_row.opponent_point_diff_prior),
            "opponent_recent_form": float(strength_row.opponent_recent_form),
            "opponent_games_played_prior":
                int(strength_row.opponent_games_played_prior),
            "celtics_point_diff_prior":
                float(strength_row.celtics_point_diff_prior),
            "strength_diff_prior": float(strength_row.strength_diff_prior),
            # unshrunk, display only
            "opponent_win_pct_prior_raw": _raw("opponent_win_pct_prior_raw"),
            "opponent_point_diff_prior_raw":
                _raw("opponent_point_diff_prior_raw"),
            "celtics_point_diff_prior_raw":
                _raw("celtics_point_diff_prior_raw"),
            "celtics_games_played_prior":
                _raw("celtics_games_played_prior"),
            "shrinkage_games": SHRINKAGE_GAMES,
        }

    final = merged.iloc[-1]
    opponent_team_id = int(
        rosters.loc[~rosters["is_celtics_team"], "team_id"].iloc[0])

    return {
        "meta": {
            "game_id": game_id,
            "season": str(events["season"].iloc[0]),
            "date": str(pd.Timestamp(events["game_date"].iloc[0]).date()),
            "matchup": str(index_row["MATCHUP"]),
            "opponent": str(events["opponent_tricode"].iloc[0]),
            # Full names, so a finished game can read "Boston Celtics defeat
            # Denver Nuggets" rather than "BOS defeat DEN". Falls back to the
            # tricode when the bios file has no row for that team.
            "celtics_name": (names or {}).get("BOS", "BOS"),
            "opponent_name": (names or {}).get(
                str(events["opponent_tricode"].iloc[0]),
                str(events["opponent_tricode"].iloc[0])),
            "celtics_is_home": celtics_home,
            "celtics_final": int(final["celtics_score"]),
            "opponent_final": int(final["opponent_score"]),
            "celtics_won": bool(int(events["celtics_won"].iloc[0])),
            "periods": int(merged["period"].max()),
            "events": int(len(merged)),
            "celtics_logo": TEAM_LOGO_TEMPLATE.format(
                team_id=config.CELTICS_TEAM_ID),
            "opponent_logo": TEAM_LOGO_TEMPLATE.format(
                team_id=opponent_team_id),
            "probability_source": (
                "out-of-fold tier3_celtics: predicted by a model that never "
                "saw this season"),
        },
        "opponent_context": opponent_context,
        "players": players,
        "lineup_table": lineup_table,
        "events": {
            "event_index": column("event_index", int),
            "period": column("period", int),
            "clock": column("clock_raw"),
            "elapsed": [round(float(v), 1) for v in merged["seconds_elapsed_game"]],
            "celtics_score": column("celtics_score", int),
            "opponent_score": column("opponent_score", int),
            "margin": column("celtics_margin", int),
            "wp": [round(float(v), 5) for v in merged[PRIMARY_TIER]],
            "wp_generic": [round(float(v), 5) for v in merged[BASELINE_TIER]],
            "description": column("description"),
            "action_type": column("action_type"),
            "team": column("team_tricode"),
            "person_id": column("person_id", int),
            "shot_result": column("shot_result"),
            "shot_value": column("shot_value", int),
            "loc_x": column("loc_x", int),
            "loc_y": column("loc_y", int),
            "is_clutch": [bool(v) for v in merged["is_clutch"]],
            "celtics_lineup": celtics_lineup,
            "opponent_lineup": opponent_lineup,

            # Model inputs needed for browser-based what-if predictions.
            "seconds_remaining_period": [
                round(float(v), 2) for v in merged["seconds_remaining_period"]],
            "seconds_remaining_game": [
                round(float(v), 2) for v in merged["seconds_remaining_game"]],
            "celtics_has_possession": [
                int(bool(v)) for v in merged["celtics_has_possession"]],
            "momentum_120s": [int(v) for v in merged["momentum_120s"]],
            "momentum_300s": [int(v) for v in merged["momentum_300s"]],
            "possession_number": column("possession_number", int),
        },
    }


# ---------------------------------------------------------------------------

def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"), ensure_ascii=False,
                  default=_jsonable)


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    raise TypeError(f"not JSON serialisable: {type(value)}")


def comeback_fields(events: dict) -> dict:
    """
    The largest deficit in one game, and the moment it was reached.

    `margin` is celtics_score - opponent_score at every event, both taken from
    the play-by-play, so this is arithmetic on the scoreboard rather than a
    model output. A game the Celtics led wire to wire returns 0 and no moment,
    which is a real answer and not a missing one.

    The dashboard ranks these for WINS only. Trailing by 30 and losing is not
    a comeback, and that ranking lives in web/lib/comebacks.js, not here.
    """
    margin = list(events["margin"])
    if not margin:
        raise ValueError("empty margin series")
    worst = min(margin)
    deficit = max(0, -int(worst))
    if deficit == 0:
        return {"largest_deficit": 0, "deficit_period": None,
                "deficit_clock": None, "deficit_event": None}
    at = margin.index(worst)
    return {
        "largest_deficit": deficit,
        "deficit_period": int(events["period"][at]),
        "deficit_clock": str(events["clock"][at]),
        "deficit_event": int(at),
    }


def build_all():
    config.ensure_dirs()
    frame, lineups, rosters, oof, index, bios, values, strength = load_tables()
    frame = attach_probabilities(frame, oof)
    headshots = build_headshot_lookup(load_season_headshots())
    names = team_names(bios)

    index = index.set_index("game_id")
    strength_by_game = (strength.set_index("game_id")
                        if not strength.empty else None)

    games_dir = config.SERVING_DIR / "games"
    games_dir.mkdir(parents=True, exist_ok=True)

    catalogue, written = [], 0
    for game_id, events in frame.groupby("game_id", sort=True):
        game_rosters = rosters.loc[rosters["game_id"].eq(game_id)]
        if game_rosters.empty:
            raise ValueError(f"no roster rows for {game_id}")
        strength_row = None
        if strength_by_game is not None and game_id in strength_by_game.index:
            strength_row = strength_by_game.loc[game_id]

        payload = build_game_payload(
            game_id, events, lineups, game_rosters, bios, values,
            index.loc[game_id], strength_row, headshots, names)
        write_json(games_dir / f"{game_id}.json", payload)
        written += 1

        meta = payload["meta"]
        catalogue.append({
            "game_id": game_id,
            "season": meta["season"],
            "date": meta["date"],
            "matchup": meta["matchup"],
            "opponent": meta["opponent"],
            "celtics_is_home": meta["celtics_is_home"],
            "celtics_final": meta["celtics_final"],
            "opponent_final": meta["opponent_final"],
            "celtics_won": meta["celtics_won"],
            "periods": meta["periods"],
            "lowest_wp": round(float(min(payload["events"]["wp"])), 5),
            "highest_wp": round(float(max(payload["events"]["wp"])), 5),
            **comeback_fields(payload["events"]),
            "opponent_logo": meta["opponent_logo"],
        })
        if written % 100 == 0:
            logger.info("  %d games written", written)

    write_json(config.SERVING_DIR / "index.json", {
        "games": catalogue,
        "seasons": sorted({g["season"] for g in catalogue}),
        "count": len(catalogue),
        "probability_source": (
            "out-of-fold tier3_celtics: every probability comes from a model "
            "that never saw that game's season"),
    })

    coverage = bio_coverage_by_minutes(rosters, bios)
    if len(coverage):
        write_json(config.SERVING_DIR / "coverage.json",
                   {"by_season": coverage.to_dict(orient="records")})
    return catalogue, coverage


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    catalogue, coverage = build_all()

    total_bytes = sum(p.stat().st_size
                      for p in (config.SERVING_DIR / "games").glob("*.json"))
    print()
    print("=" * 70)
    print("SERVING DATA BUILT")
    print("=" * 70)
    print(f"Games written: {len(catalogue)}  ->  {config.SERVING_DIR}")
    print(f"Total size: {total_bytes / 1e6:.1f} MB "
          f"({total_bytes / max(len(catalogue), 1) / 1e3:.0f} KB per game)")
    print()
    print("Every probability is OUT OF FOLD. The saved deployment model was")
    print("fitted on all eight seasons, so using it here would be in-sample")
    print("for every game you could replay. It would look better and mean less.")

    if len(coverage):
        print()
        print("BIO COVERAGE, MEASURED IN MINUTES NOT PLAYERS")
        print(f"  {'season':<10}{'players':>9}{'no bio':>8}"
              f"{'minutes':>12}{'no-bio min':>12}{'share':>8}{'median':>9}")
        for _, row in coverage.iterrows():
            print(f"  {row['season']:<10}{row['players']:>9,}"
                  f"{row['players_without_bio']:>8,}{row['minutes']:>12,.0f}"
                  f"{row['minutes_without_bio']:>12,.0f}"
                  f"{row['share_of_minutes']:>8.2%}"
                  f"{row['median_minutes_of_missing']:>9.1f}")
        share = (coverage["minutes_without_bio"].sum()
                 / coverage["minutes"].sum())
        print()
        print(f"  Overall, {share:.2%} of minutes belong to players with no "
              "bio row.")
        print("  Those cards keep a photo, a name, a jersey and a coarse")
        print("  position; they lose height and the granular position.")
        print("  'median' is the median total minutes of a missing player,")
        print("  which is the test of whether these are call-ups or rotation")
        print("  players.")

    # Filter to wins FIRST, then sort. Sorting all games and filtering the top
    # five afterwards printed nothing, because the five lowest probabilities in
    # the dataset all belong to games Boston lost.
    wins = [g for g in catalogue if g["celtics_won"]]
    wins.sort(key=lambda g: g["lowest_wp"])
    print()
    print("Biggest comebacks in the dataset, by lowest out-of-fold probability")
    print("in a game Boston won:")
    for game in wins[:5]:
        print(f"  {game['date']}  {game['matchup']:<14}"
              f"low {game['lowest_wp']:.4f}   final "
              f"{game['celtics_final']}-{game['opponent_final']}")

    print()
    print("Next: run scripts/21_serve_api.py")
    return catalogue


if __name__ == "__main__":
    main()