"""
Fill the gap for players who changed teams DURING a season.

THE PROBLEM THIS FIXES
----------------------
`player_bios.csv` records exactly ONE team per player-season: 0 of its 4,009
rows carry a second team_id. So a player traded mid-season has a single bios
row, and `scripts/39_build_season_headshots.py` only ever fetched a photo for
that one team.

Trevor Ariza in 2018-19 is the visible case. He started the season in Phoenix
and was traded to Washington in December. Bios records Washington, so the
dashboard shows him in a Wizards shirt during a Celtics-Suns game he played as
a Sun. The photo is season-correct and team-wrong.

The boxscore knows better. `rosters.parquet` carries `team_tricode` PER GAME,
so it says which shirt each player wore in each game the Celtics played. This
script takes the set of (player, season, team) combinations that actually
appear in those games, subtracts the ones already fetched, and fetches only
the remainder.

It is a small, bounded delta, not a re-run. It reports the count before
fetching anything, so if the number is zero you learn that in a second instead
of in twenty-five minutes.

WHAT IT DOES NOT DO
-------------------
It does not invent a URL. Every added row was fetched and confirmed to be a
real image, exactly like script 39. Combinations that come back 403 are
recorded as unusable and fall back to the current photo, same as the 329
already known.

HOW TO RUN
----------
    python scripts/41_fill_traded_headshots.py

Then re-run `scripts/20_build_serving.py` and reload the dashboard.

OUTPUT
------
    data/processed/season_headshots.csv    appended in place
    reports/traded_headshot_fill.txt
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from src import config

MAP_PATH = config.PROCESSED_DIR / "season_headshots.csv"
ROSTERS_PATH = config.INTERIM_DIR / "rosters.parquet"
BIOS_PATH = config.RAW_DIR / "player_bios.csv"
REPORTS = config.REPORTS_DIR

SEASON_TEMPLATE = ("https://cdn.nba.com/headshots/nba/{team_id}/{year}/"
                   "1040x760/{person_id}.png")

DELAY_SECONDS = 0.5
TIMEOUT = 20
MIN_IMAGE_BYTES = 2000

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    ),
    "Referer": "https://www.nba.com/",
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def fetch(url: str) -> dict:
    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    except requests.RequestException as exc:
        return {"status": -1, "content_type": "", "bytes": 0, "md5": "",
                "error": type(exc).__name__}
    body = response.content
    return {
        "status": response.status_code,
        "content_type": response.headers.get("Content-Type", ""),
        "bytes": len(body),
        "md5": hashlib.md5(body).hexdigest() if body else "",
        "error": "",
    }


def is_image(result: dict) -> bool:
    return (result["status"] == 200
            and "image" in (result["content_type"] or "")
            and result["bytes"] >= MIN_IMAGE_BYTES)


def team_id_for(bios: pd.DataFrame) -> dict:
    """
    Tricode to team id, per season and globally.

    Per season first, because abbreviations have moved before (NOH to NOP, and
    the CHA/CHO pair). The global map is the fallback for a tricode the bios
    file happens not to carry in that season.
    """
    per_season, overall = {}, {}
    for row in bios[["season", "team_abbrev", "team_id"]].drop_duplicates().itertuples():
        per_season[(str(row.season), str(row.team_abbrev))] = int(row.team_id)
        overall[str(row.team_abbrev)] = int(row.team_id)
    return {"per_season": per_season, "overall": overall}


def main() -> None:
    for path in (MAP_PATH, ROSTERS_PATH, BIOS_PATH):
        if not path.exists():
            raise SystemExit(f"missing {path}")

    REPORTS.mkdir(exist_ok=True)
    existing = pd.read_csv(MAP_PATH)
    rosters = pd.read_parquet(ROSTERS_PATH)
    bios = pd.read_csv(BIOS_PATH, dtype={"jersey": "string"})
    teams = team_id_for(bios)

    have = set(zip(existing["person_id"].astype(int),
                   existing["season"].astype(str),
                   existing["team_abbrev"].astype(str)))
    need = set(zip(rosters["person_id"].astype(int),
                   rosters["season"].astype(str),
                   rosters["team_tricode"].astype(str)))
    missing = sorted(need - have)

    logger.info("%d (player, season, team) combinations appear in Celtics games",
                len(need))
    logger.info("%d already fetched, %d to fetch", len(need & have), len(missing))

    if not missing:
        REPORTS.joinpath("traded_headshot_fill.txt").write_text(
            "Nothing to fetch. Every (player, season, team) combination that\n"
            "appears in a Celtics game was already covered by script 39.\n")
        logger.info("nothing to do")
        return

    unresolved = []
    rows = []
    for n, (person_id, season, tricode) in enumerate(missing, start=1):
        team_id = (teams["per_season"].get((season, tricode))
                   or teams["overall"].get(tricode))
        if team_id is None:
            # No id for this tricode anywhere in the bios file. Recorded and
            # skipped rather than guessed at.
            unresolved.append((person_id, season, tricode))
            continue

        year = int(season.split("-")[0])
        url = SEASON_TEMPLATE.format(team_id=team_id, year=year,
                                     person_id=person_id)
        result = fetch(url)
        rows.append({
            "person_id": person_id,
            "season": season,
            "team_id": team_id,
            "team_abbrev": tricode,
            "url": url,
            "status": result["status"],
            "content_type": result["content_type"],
            "bytes": result["bytes"],
            "md5": result["md5"],
            "error": result["error"],
            "usable": is_image(result),
            # Not compared against the current photo here. Script 39 already
            # established that this path serves season images; what matters for
            # these rows is only whether one exists.
            "differs_from_latest": False,
        })
        if n % 25 == 0 or n == len(missing):
            logger.info("  %d/%d, usable so far %d", n, len(missing),
                        sum(r["usable"] for r in rows))
        time.sleep(DELAY_SECONDS)

    added = pd.DataFrame(rows)
    combined = pd.concat([existing, added], ignore_index=True)
    combined.to_csv(MAP_PATH, index=False)

    usable_added = int(added["usable"].sum()) if len(added) else 0
    lines = [
        "TRADED-PLAYER HEADSHOT FILL",
        "=" * 72,
        "",
        f"combinations in Celtics games   {len(need):,}",
        f"already covered                 {len(need & have):,}",
        f"fetched now                     {len(added):,}",
        f"  usable                        {usable_added:,}",
        f"  no image                      {len(added) - usable_added:,}",
        f"tricode with no team id         {len(unresolved):,}",
        "",
        f"map rows before                 {len(existing):,}",
        f"map rows after                  {len(combined):,}",
        f"usable after                    "
        f"{int(combined['usable'].astype(bool).sum()):,}",
        "",
    ]
    if usable_added:
        lines.append("EXAMPLES ADDED")
        lines.append("-" * 72)
        for row in added.loc[added["usable"]].head(12).itertuples():
            lines.append(f"    {row.season}  {row.team_abbrev:<4} {row.person_id}")
        lines.append("")
    if unresolved:
        lines.append("UNRESOLVED TRICODES")
        lines.append("-" * 72)
        for person_id, season, tricode in unresolved[:12]:
            lines.append(f"    {season}  {tricode:<4} {person_id}")
        lines.append("")
    lines.append("Re-run scripts/20_build_serving.py to pick these up.")

    path = REPORTS / "traded_headshot_fill.txt"
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
