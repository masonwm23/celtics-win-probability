"""
Phase 9a: player biographical data, for the dashboard.

WHY THIS PULL EXISTS
--------------------
The dashboard's player cards need a name, a position, a height and a headshot.
The pipeline already has names. It does not have the other three, and Phase 1
recorded exactly why:

  - `position` in the boxscore is the coarse G / F / C bucket, and before
    2017-18 it is populated for nine or ten players per team rather than five,
    so it cannot even identify starters. Granular guard/forward/centre splits
    were flagged then as needing a separate source. This is that source.
  - Height appears nowhere in the play-by-play or boxscore endpoints.
  - Headshots are served by CDN from a player's `personId`, which the pipeline
    does have, but the ID has to be tied to a bio row to be useful.

WHY CommonTeamRoster AND NOT PlayerIndex
----------------------------------------
PlayerIndex was tried first: one call per season instead of thirty, which is a
much cheaper pull. It returned **106 players for 2016-17**, against a floor of
400, and the floor stopped the run rather than letting a thin file through.

The reason is that PlayerIndex's `Historical` parameter defaults to excluding
retired players, so asking it for 2016-17 returns the players from that season
who are STILL on a roster today. That is a survivorship-filtered set, and it is
exactly the wrong thing for a historical dashboard: every player who has since
retired would show a blank card. Setting `Historical=1` would widen it, but the
semantics of that flag combined with a season filter are not documented clearly
enough to rely on, and a wrong guess here is invisible rather than loud.

CommonTeamRoster asks a question with one unambiguous answer: who was on this
team in this season. Thirty teams times eight seasons is 240 calls, roughly
four minutes. Slower, and correct.

It also matches what the dashboard's roster panel actually displays, which is a
team's roster for a given season, so no reshaping is needed.

RESUMABLE
---------
Each team-season is cached to its own CSV. A failed or interrupted run picks up
where it stopped instead of re-pulling everything, which is the same discipline
`pull_raw.py` uses for the 636-game pull.

NOTHING HERE FEEDS THE MODEL
----------------------------
This is display data. It is pulled after the research is finished, it enters no
feature, and it cannot affect any published number. Keeping that boundary
explicit matters: a bio field that leaked into a feature would be a
season-long-summary leak of exactly the kind Phase 5 was built to avoid.

Output
------
data/raw/team_rosters/{season}_{abbrev}.csv   one per team-season, cached
data/raw/player_bios.csv                      the combined table
"""

import logging

import pandas as pd

from src import config
from src.nba_client import call_endpoint

logger = logging.getLogger(__name__)

# Columns CommonTeamRoster returns, mapped to the names the dashboard uses.
# Requested BY NAME so an endpoint schema change is a loud failure rather than
# a silently missing column.
WANTED = {
    "PLAYER_ID": "person_id",
    "PLAYER": "full_name",
    "PLAYER_SLUG": "slug",
    "NUM": "jersey",
    "POSITION": "position",
    "HEIGHT": "height",
    "WEIGHT": "weight",
    "BIRTH_DATE": "birth_date",
    "AGE": "age",
    "EXP": "experience",
    "SCHOOL": "school",
}

# An NBA roster carries 14 to 21 players over a season including ten-day
# contracts and two-ways. These are bounds to catch a truncated or wrong-season
# response, not exact expectations.
MIN_ROSTER = 10
MAX_ROSTER = 40

HEADSHOT_TEMPLATE = ("https://cdn.nba.com/headshots/nba/latest/1040x760/"
                     "{person_id}.png")
TEAM_LOGO_TEMPLATE = ("https://cdn.nba.com/logos/nba/{team_id}/primary/L/"
                      "logo.svg")


def roster_dir():
    path = config.RAW_DIR / "team_rosters"
    path.mkdir(parents=True, exist_ok=True)
    return path


def all_teams():
    """The 30 current franchises, from nba_api's static table. No network."""
    from nba_api.stats.static import teams
    return sorted(teams.get_teams(), key=lambda t: t["abbreviation"])


def fetch_team_season(team_id: int, season: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import commonteamroster

    endpoint = call_endpoint(commonteamroster.CommonTeamRoster,
                             team_id=team_id,
                             season=season,
                             league_id_nullable="00")
    frames = endpoint.get_data_frames()
    if not frames or frames[0].empty:
        raise RuntimeError(
            f"CommonTeamRoster returned no rows for team {team_id} in {season}")

    df = frames[0].copy()
    missing = [c for c in WANTED if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"CommonTeamRoster is missing expected columns: {missing}. "
            "The endpoint's schema has changed; do not guess at replacements.")
    return df[list(WANTED)].rename(columns=WANTED)


def load_or_fetch(team: dict, season: str, resume=True) -> pd.DataFrame:
    """One team-season, from cache when available."""
    path = roster_dir() / f"{season}_{team['abbreviation']}.csv"
    if resume and path.exists():
        return pd.read_csv(path)

    frame = fetch_team_season(team["id"], season)
    if not MIN_ROSTER <= len(frame) <= MAX_ROSTER:
        raise RuntimeError(
            f"{team['abbreviation']} {season} returned {len(frame)} players, "
            f"outside the plausible range {MIN_ROSTER} to {MAX_ROSTER}. "
            "Do not proceed on a roster this shape.")

    frame["season"] = season
    frame["team_id"] = team["id"]
    frame["team_abbrev"] = team["abbreviation"]
    frame["team_name"] = team["full_name"]
    frame.to_csv(path, index=False)
    return frame


def parse_height_inches(height) -> float:
    """
    Convert a feet-dash-inches string to inches.

    The feed gives '6-8'. Returned as a number so the dashboard can sort and
    compare without re-parsing, while the original string is kept for display.
    Anything unparseable becomes NaN rather than a guess.
    """
    if not isinstance(height, str) or "-" not in height:
        return float("nan")
    feet, _, inches = height.partition("-")
    try:
        return float(feet) * 12 + float(inches)
    except ValueError:
        return float("nan")


def build_player_bios(resume=True) -> pd.DataFrame:
    config.ensure_dirs()
    teams = all_teams()
    total = len(teams) * len(config.SEASONS)

    frames, done = [], 0
    for season in config.SEASONS:
        for team in teams:
            frames.append(load_or_fetch(team, season, resume=resume))
            done += 1
            if done % 30 == 0:
                logger.info("  %d/%d team-seasons", done, total)

    df = pd.concat(frames, ignore_index=True)
    df["person_id"] = df["person_id"].astype("int64")
    df["height_inches"] = df["height"].map(parse_height_inches)
    df["headshot_url"] = df["person_id"].map(
        lambda pid: HEADSHOT_TEMPLATE.format(person_id=pid))
    df["team_logo_url"] = df["team_id"].map(
        lambda tid: TEAM_LOGO_TEMPLATE.format(team_id=tid))

    return df.sort_values(["season", "team_abbrev", "full_name"]).reset_index(
        drop=True)


def coverage_against_rosters(bios: pd.DataFrame) -> pd.DataFrame:
    """
    How many players the pipeline actually saw are covered by the bio pull.

    Reported rather than assumed. A player who appeared in a Celtics game but
    has no bio row will show a name and no headshot in the dashboard, and it is
    better to know the number in advance than to discover it on screen.
    """
    if not config.ROSTERS_PARQUET.exists():
        return pd.DataFrame()
    rosters = pd.read_parquet(config.ROSTERS_PARQUET)
    id_column = next((c for c in ("person_id", "player_id", "personId")
                      if c in rosters.columns), None)
    if id_column is None:
        return pd.DataFrame()

    if "season" in rosters.columns:
        groups = list(rosters.groupby("season"))
    else:
        groups = [("all seasons", rosters)]

    rows = []
    for season, group in groups:
        seen = set(group[id_column].dropna().astype("int64"))
        if season == "all seasons":
            have = set(bios["person_id"])
        else:
            have = set(bios.loc[bios["season"].eq(season), "person_id"])
        rows.append({
            "season": season,
            "players_seen": len(seen),
            "with_bio": len(seen & have),
            "missing": len(seen - have),
        })
    return pd.DataFrame(rows)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    cached = len(list(roster_dir().glob("*.csv")))
    total = 30 * len(config.SEASONS)
    print(f"Team-seasons cached: {cached}/{total}. "
          f"{total - cached} to fetch, about {(total - cached) * 1.1 / 60:.1f} "
          f"minutes.")

    df = build_player_bios()
    out = config.RAW_DIR / "player_bios.csv"
    df.to_csv(out, index=False)

    print()
    print("=" * 70)
    print("PLAYER BIOS PULLED")
    print("=" * 70)
    print(f"Saved to: {out}")
    print(f"Rows: {len(df):,} player-seasons   "
          f"distinct players: {df.person_id.nunique():,}")
    print()
    print(f"  {'season':<10}{'players':>9}{'teams':>7}{'with height':>13}"
          f"{'with position':>15}")
    for season, group in df.groupby("season"):
        has_position = group["position"].fillna("").astype(str).str.strip().ne("")
        print(f"  {season:<10}{len(group):>9,}{group.team_abbrev.nunique():>7}"
              f"{group.height_inches.notna().sum():>13,}"
              f"{int(has_position.sum()):>15,}")

    print()
    print("Positions available (granular, unlike the boxscore G/F/C field):")
    counts = df["position"].fillna("(blank)").replace("", "(blank)") \
        .value_counts()
    for value, count in counts.items():
        print(f"    {str(value):<16}{count:>7,}")

    print()
    print(f"Height range: {df.height_inches.min():.0f} to "
          f"{df.height_inches.max():.0f} inches "
          f"({df.height_inches.isna().sum()} unparseable)")

    coverage = coverage_against_rosters(df)
    if len(coverage):
        print()
        print("Coverage against players the pipeline actually saw:")
        print(f"  {'season':<12}{'seen':>8}{'with bio':>11}{'missing':>10}")
        for _, row in coverage.iterrows():
            print(f"  {str(row['season']):<12}{row['players_seen']:>8,}"
                  f"{row['with_bio']:>11,}{row['missing']:>10,}")
        total_missing = int(coverage["missing"].sum())
        print()
        if total_missing:
            print(f"  {total_missing} player-seasons have no bio row. Those "
                  "cards will show a name and no")
            print("  headshot, which is the honest fallback. They are not "
                  "filled in with guesses.")
        else:
            print("  Every player the pipeline saw has a bio row.")

    print()
    print("NOTE: this is DISPLAY data. It feeds no model feature and cannot")
    print("affect any published number.")
    print()
    print("Next: run scripts/19b_probe_schema.py")
    return df


if __name__ == "__main__":
    main()
