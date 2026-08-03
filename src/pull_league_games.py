"""
Phase 5, step 1: pull league-wide game logs.

Why this pull is necessary
--------------------------
Opponent strength has to answer "how good was this opponent, as of this date".
Answering it needs the opponent's FULL schedule, not just their games against
Boston. Boston plays most teams two to four times a season, so a record computed
from those games alone would rest on a handful of results and would be biased by
construction: it would literally be "how well did they do against Boston".

So this pulls every team's every regular season game for the eight study seasons.
Eight API calls, one per season. Around 2,460 games per season with one row per
team per game.

Output
------
data/raw/league_game_logs.csv

Nothing here is Celtics-specific. It is the league context that makes an honest
opponent measure possible.
"""

import logging

import pandas as pd

from src import config
from src.nba_client import call_endpoint

logger = logging.getLogger(__name__)

KEEP_COLUMNS = [
    "SEASON_ID", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID",
    "GAME_DATE", "MATCHUP", "WL", "PTS", "PLUS_MINUS",
]

# A full NBA season is 1,230 games, so 2,460 team-rows. The shortened seasons
# are smaller. These are floors used to catch a truncated response, not exact
# expectations.
MIN_TEAM_ROWS = {
    "2016-17": 2400, "2017-18": 2400, "2018-19": 2400,
    "2019-20": 2000, "2020-21": 2000,
    "2021-22": 2400, "2022-23": 2400, "2023-24": 2400,
}


def fetch_season(season: str) -> pd.DataFrame:
    """Every team's regular season games for one season."""
    from nba_api.stats.endpoints import leaguegamefinder

    logger.info("Fetching league game log for %s", season)
    endpoint = call_endpoint(
        leaguegamefinder.LeagueGameFinder,
        season_nullable=season,
        season_type_nullable=config.SEASON_TYPE,
        league_id_nullable="00",          # NBA only, excludes G League and WNBA
    )
    frames = endpoint.get_data_frames()
    if not frames or frames[0].empty:
        raise RuntimeError(f"LeagueGameFinder returned no rows for {season}")

    df = frames[0].copy()
    missing = [c for c in KEEP_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Expected columns absent: {missing}")

    df = df[KEEP_COLUMNS].copy()
    df["SEASON"] = season

    # Regular season game IDs begin 002. Filtering by ID rather than trusting
    # the season_type parameter alone is cheap insurance.
    df = df.loc[df["GAME_ID"].astype(str).str.startswith("002")]
    return df


def build_league_logs() -> pd.DataFrame:
    config.ensure_dirs()

    frames = []
    for season in config.SEASONS:
        frame = fetch_season(season)
        floor = MIN_TEAM_ROWS.get(season, 2000)
        if len(frame) < floor:
            raise RuntimeError(
                f"{season} returned only {len(frame)} team-rows, below the "
                f"floor of {floor}. This looks truncated; do not proceed.")
        frames.append(frame)
        logger.info("  %s: %d team-rows, %d games",
                    season, len(frame), frame["GAME_ID"].nunique())

    df = pd.concat(frames, ignore_index=True)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="raise")
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
    df["WON"] = df["WL"].map({"W": 1, "L": 0})

    if df["WON"].isna().any():
        raise RuntimeError("Unparseable WL values in the league game log")

    df = df.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ABBREVIATION"])
    return df.reset_index(drop=True)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    df = build_league_logs()
    out = config.RAW_DIR / "league_game_logs.csv"
    df.to_csv(out, index=False)

    print()
    print("=" * 68)
    print("LEAGUE GAME LOGS PULLED")
    print("=" * 68)
    print(f"Saved to: {out}")
    print(f"Team-rows: {len(df):,}   distinct games: {df.GAME_ID.nunique():,}")
    print()
    print(f"  {'season':<10}{'team-rows':>11}{'games':>8}{'teams':>8}")
    for season, group in df.groupby("SEASON"):
        print(f"  {season:<10}{len(group):>11,}{group.GAME_ID.nunique():>8,}"
              f"{group.TEAM_ABBREVIATION.nunique():>8}")
    print()
    print("Each game appears twice, once per team, which is what makes an")
    print("as-of-date record computable for every team.")

    # Every game should have exactly two team-rows.
    counts = df.groupby("GAME_ID").size()
    bad = counts[counts != 2]
    print()
    if bad.empty:
        print("Every game has exactly 2 team-rows.")
        print("Next: run scripts/14_build_opponent_strength.py")
    else:
        print(f"WARNING: {len(bad)} game(s) do not have exactly 2 team-rows.")
        print("Stop and report this before continuing.")
    return df


if __name__ == "__main__":
    main()
