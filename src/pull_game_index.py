"""
Phase 1, step 2: build the game index.

This is the spine of the whole project. Every later step iterates over this
file. It contains one row per Boston Celtics regular season game across the
eight study seasons, with the game ID, date, season, opponent, home or away
flag, the result, and the final score.

Nothing here is derived or guessed. Opponent and home/away are parsed from the
API's own MATCHUP string, which is either "BOS vs. XXX" (home) or
"BOS @ XXX" (away).

Output
------
data/raw/game_index.csv

HOW TO RUN IN SPYDER
  Open scripts/01_pull_game_index.py and press F5.
"""

import logging

import pandas as pd

from src import config
from src.nba_client import call_endpoint

logger = logging.getLogger(__name__)

# Columns we keep from LeagueGameFinder. Anything else is dropped so the index
# stays readable, and because the wide game-log columns duplicate information
# we will recompute properly from play-by-play later.
KEEP_COLUMNS = [
    "SEASON_ID", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID",
    "GAME_DATE", "MATCHUP", "WL", "PTS", "PLUS_MINUS",
]


def parse_matchup(matchup: str):
    """
    Split an NBA MATCHUP string into (opponent_abbrev, is_home).

    "BOS vs. ATL" -> ("ATL", True)
    "BOS @ ATL"   -> ("ATL", False)

    Raises ValueError on anything unexpected rather than guessing, so a format
    change surfaces loudly instead of corrupting the index.
    """
    if not isinstance(matchup, str):
        raise ValueError(f"MATCHUP is not a string: {matchup!r}")

    text = matchup.strip()
    if " vs. " in text:
        left, right = text.split(" vs. ", 1)
        is_home = True
    elif " @ " in text:
        left, right = text.split(" @ ", 1)
        is_home = False
    else:
        raise ValueError(f"Unrecognised MATCHUP format: {matchup!r}")

    left, right = left.strip(), right.strip()
    if left != config.CELTICS_ABBREV:
        raise ValueError(
            f"MATCHUP does not start with {config.CELTICS_ABBREV}: {matchup!r}"
        )
    if len(right) != 3:
        raise ValueError(f"Opponent abbreviation looks wrong: {matchup!r}")

    return right, is_home


def fetch_season(season: str) -> pd.DataFrame:
    """Fetch Boston's regular season game log for one season."""
    from nba_api.stats.endpoints import leaguegamefinder

    logger.info("Fetching game index for %s", season)
    endpoint = call_endpoint(
        leaguegamefinder.LeagueGameFinder,
        team_id_nullable=config.CELTICS_TEAM_ID,
        season_nullable=season,
        season_type_nullable=config.SEASON_TYPE,
    )
    frames = endpoint.get_data_frames()
    if not frames or frames[0].empty:
        raise RuntimeError(f"LeagueGameFinder returned no rows for {season}")

    df = frames[0].copy()
    missing = [c for c in KEEP_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Expected columns absent from LeagueGameFinder response: {missing}"
        )

    df = df[KEEP_COLUMNS].copy()
    df["SEASON"] = season
    return df


def build_game_index() -> pd.DataFrame:
    """Fetch every season, combine, parse, validate structurally, and sort."""
    config.ensure_dirs()

    frames = []
    for season in config.SEASONS:
        frames.append(fetch_season(season))

    df = pd.concat(frames, ignore_index=True)

    # Parse opponent and home/away from MATCHUP. Any failure raises.
    parsed = df["MATCHUP"].apply(parse_matchup)
    df["OPPONENT_ABBREV"] = [p[0] for p in parsed]
    df["IS_HOME"] = [p[1] for p in parsed]

    # Types and ordering.
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="raise")
    df["GAME_ID"] = df["GAME_ID"].astype(str)
    df["CELTICS_WON"] = df["WL"].map({"W": 1, "L": 0})

    if df["CELTICS_WON"].isna().any():
        bad = df.loc[df["CELTICS_WON"].isna(), ["GAME_ID", "WL"]]
        raise RuntimeError(f"Unparseable WL values:\n{bad}")

    # A team game log should contain each game exactly once.
    dupes = df["GAME_ID"].duplicated().sum()
    if dupes:
        raise RuntimeError(f"{dupes} duplicate GAME_ID rows in the game index")

    df = df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    return df


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    df = build_game_index()
    df.to_csv(config.GAME_INDEX_CSV, index=False)

    print()
    print("=" * 62)
    print("GAME INDEX BUILT")
    print("=" * 62)
    print(f"Saved to: {config.GAME_INDEX_CSV}")
    print(f"Total games: {len(df)}")
    print()
    print("Games per season (actual vs expected):")
    counts = df.groupby("SEASON").size()
    all_match = True
    for season in config.SEASONS:
        actual = int(counts.get(season, 0))
        expected = config.EXPECTED_GAME_COUNTS[season]
        flag = "ok" if actual == expected else "MISMATCH"
        if actual != expected:
            all_match = False
        print(f"  {season}  actual {actual:>3}   expected {expected:>3}   {flag}")
    print()
    print(f"  TOTAL     actual {len(df):>3}   "
          f"expected {config.EXPECTED_TOTAL_GAMES:>3}   "
          f"{'ok' if len(df) == config.EXPECTED_TOTAL_GAMES else 'MISMATCH'}")
    print()
    print(f"Home games: {int(df['IS_HOME'].sum())}   "
          f"Away games: {int((~df['IS_HOME']).sum())}")
    print(f"Wins: {int(df['CELTICS_WON'].sum())}   "
          f"Losses: {int((1 - df['CELTICS_WON']).sum())}   "
          f"Win rate: {df['CELTICS_WON'].mean():.3f}")
    print(f"Distinct opponents: {df['OPPONENT_ABBREV'].nunique()}")
    print(f"Date range: {df['GAME_DATE'].min().date()} to "
          f"{df['GAME_DATE'].max().date()}")
    print()
    if all_match and len(df) == config.EXPECTED_TOTAL_GAMES:
        print("All season counts match expectations.")
        print("Next: run scripts/02_validate_game_index.py")
    else:
        print("COUNTS DO NOT MATCH. Stop here and report this output.")
        print("Do not start the raw pull until the difference is explained.")


if __name__ == "__main__":
    main()
