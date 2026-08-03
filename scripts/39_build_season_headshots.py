"""
Build a VERIFIED map of season-correct headshot URLs.

WHAT THE PROBE ESTABLISHED
--------------------------
`scripts/38_probe_season_headshots.py` found that this path returns a real,
season-specific image:

    https://cdn.nba.com/headshots/nba/{team_id}/{season_start_year}/1040x760/{person_id}.png

24 of 24 sampled player-seasons returned an image, spread across all eight
seasons. The END-year form returned only 13 of 24, so the START year is the
convention and the end-year hits were coincidence.

One number in that report needs reading carefully rather than at face value.
The 260x190 rows scored 24 of 24 "distinct from latest" while the 1040x760
rows scored 18 of 24. That is NOT evidence that the small size is better. The
control image was 1040x760, so ANY 260x190 body differs from it on bytes
alone, whatever it depicts. The 260 comparison was measuring resolution, not
season. The clean comparison is the 1040x760 one, and its 6 identical cases
have an obvious candidate explanation: a player still on the team he played
for that season has the same photo now as he did then. This script TESTS that
explanation instead of assuming it.

WHAT THIS SCRIPT DOES
---------------------
Fetches the season URL for every (person_id, team_id, season) in the bios file
and records what actually came back: status, content type, byte length, md5.
A row is written as usable only when the response is a real image. Nothing is
constructed for a player-season that was not fetched and confirmed, so the
dashboard never points at a URL that was assumed to exist.

It also fetches each player's current `latest` image once, so the report can
state how many season photos genuinely differ from the one on screen today.
That is the number that says whether this work was worth doing.

HOW TO RUN
----------
    python scripts/39_build_season_headshots.py

Roughly 5,000 requests over four threads, so five to ten minutes. It writes no
images to disk and modifies no existing data file.

OUTPUT
------
    data/processed/season_headshots.csv     the verified map, consumed later
                                            by src/build_serving.py
    reports/season_headshot_coverage.txt    coverage and what is missing
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
BIOS = ROOT / "data" / "raw" / "player_bios.csv"
OUT_DIR = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

SEASON_TEMPLATE = ("https://cdn.nba.com/headshots/nba/{team_id}/{year}/"
                   "1040x760/{person_id}.png")
LATEST_TEMPLATE = ("https://cdn.nba.com/headshots/nba/latest/1040x760/"
                   "{person_id}.png")

WORKERS = 4
TIMEOUT = 15
# A real image is tens of kilobytes. Anything under this is an error page or a
# placeholder, and calling it a headshot would put a blank circle on the court.
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

_print_lock = threading.Lock()
_counter = {"done": 0}


def season_start_year(season: str) -> int:
    """'2018-19' -> 2018."""
    return int(str(season).split("-")[0])


def fetch(url: str, attempts: int = 3) -> dict:
    """
    GET, with backoff on rate limiting and server errors.

    429 and 5xx are retried rather than recorded as "no image". Filing a
    rate-limited request as an absent one is how a previous probe in this
    project poisoned its own results: the miss got written down as a fact and
    a re-run skipped it forever.
    """
    delay = 2.0
    last = {"status": -1, "content_type": "", "bytes": 0, "md5": "",
            "error": "not attempted"}
    for attempt in range(attempts):
        try:
            response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        except requests.RequestException as exc:
            last = {"status": -1, "content_type": "", "bytes": 0, "md5": "",
                    "error": type(exc).__name__}
            time.sleep(delay)
            delay *= 3
            continue

        if response.status_code == 429 or response.status_code >= 500:
            last = {"status": response.status_code, "content_type": "",
                    "bytes": 0, "md5": "", "error": "retryable"}
            time.sleep(delay)
            delay *= 3
            continue

        body = response.content
        return {
            "status": response.status_code,
            "content_type": response.headers.get("Content-Type", ""),
            "bytes": len(body),
            "md5": hashlib.md5(body).hexdigest() if body else "",
            "error": "",
        }
    return last


def is_image(result: dict) -> bool:
    return (result["status"] == 200
            and "image" in (result["content_type"] or "")
            and result["bytes"] >= MIN_IMAGE_BYTES)


def tick(total: int, label: str) -> None:
    with _print_lock:
        _counter["done"] += 1
        done = _counter["done"]
        if done % 100 == 0 or done == total:
            logger.info("  %s %d/%d", label, done, total)


def main() -> None:
    if not BIOS.exists():
        raise SystemExit(f"missing {BIOS}. Run the raw pull first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    bios = pd.read_csv(BIOS, dtype={"jersey": "string"})
    triples = (bios[["person_id", "team_id", "team_abbrev", "season"]]
               .drop_duplicates()
               .reset_index(drop=True))
    players = sorted(triples["person_id"].unique())

    logger.info("%d player-seasons, %d distinct players",
                len(triples), len(players))

    # ---- the current photo, once per player -------------------------------
    logger.info("fetching current photos")
    _counter["done"] = 0
    latest: dict[int, dict] = {}

    def do_latest(pid: int) -> None:
        latest[int(pid)] = fetch(LATEST_TEMPLATE.format(person_id=int(pid)))
        tick(len(players), "current")

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(do_latest, players))

    # ---- the season photo, once per player-season -------------------------
    logger.info("fetching season photos")
    _counter["done"] = 0
    rows: list[dict] = []
    rows_lock = threading.Lock()

    def do_season(item) -> None:
        _, row = item
        pid = int(row.person_id)
        team_id = int(row.team_id)
        year = season_start_year(row.season)
        url = SEASON_TEMPLATE.format(team_id=team_id, year=year, person_id=pid)
        result = fetch(url)
        current = latest.get(pid, {})
        record = {
            "person_id": pid,
            "season": row.season,
            "team_id": team_id,
            "team_abbrev": row.team_abbrev,
            "url": url,
            "status": result["status"],
            "content_type": result["content_type"],
            "bytes": result["bytes"],
            "md5": result["md5"],
            "error": result["error"],
            "usable": is_image(result),
            # Whether the season photo is actually a different picture from the
            # one the dashboard shows today. False is not a failure by itself:
            # a player still on that team has the same photo now as he did
            # then, which is checked in the report below.
            "differs_from_latest": bool(
                result["md5"] and current.get("md5")
                and result["md5"] != current["md5"]
            ),
        }
        with rows_lock:
            rows.append(record)
        tick(len(triples), "season")

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(do_season, triples.iterrows()))

    frame = pd.DataFrame(rows).sort_values(["season", "person_id"])
    out_path = OUT_DIR / "season_headshots.csv"
    frame.to_csv(out_path, index=False)

    # ---- report ------------------------------------------------------------
    usable = frame[frame["usable"]]
    lines: list[str] = []
    lines.append("SEASON HEADSHOT COVERAGE")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"pattern   {SEASON_TEMPLATE}")
    lines.append("")
    lines.append(f"player-seasons in bios      {len(frame):,}")
    lines.append(f"returned a real image       {len(usable):,} "
                 f"({100 * len(usable) / max(1, len(frame)):.1f}%)")
    lines.append(f"different from current      {int(frame['differs_from_latest'].sum()):,}")
    lines.append("")

    current_ok = sum(1 for r in latest.values() if is_image(r))
    lines.append(f"current photos that loaded  {current_ok:,} of {len(players):,}")
    if current_ok == 0:
        lines.append("")
        lines.append("THE CONTROL FAILED. Current photos did not load either, so")
        lines.append("everything below is about the network, not about the NBA's")
        lines.append("archive. Do not act on these numbers.")
    lines.append("")

    lines.append("BY SEASON")
    lines.append("-" * 72)
    lines.append(f"{'season':<10}{'rows':>8}{'usable':>9}{'%':>8}{'differs':>10}")
    for season, group in frame.groupby("season"):
        ok = int(group["usable"].sum())
        lines.append(f"{season:<10}{len(group):>8,}{ok:>9,}"
                     f"{100 * ok / max(1, len(group)):>7.1f}%"
                     f"{int(group['differs_from_latest'].sum()):>10,}")
    lines.append("")

    # ---- the identical-photo explanation, tested rather than assumed -------
    # A season photo matching the current one is expected when the player is
    # STILL on that team. The bios file knows each player's most recent team,
    # so this is checkable.
    last_team = (bios.sort_values("season")
                 .groupby("person_id")["team_id"].last().to_dict())
    same = usable[~usable["differs_from_latest"]].copy()
    if len(same):
        same["still_there"] = [
            last_team.get(pid) == tid
            for pid, tid in zip(same["person_id"], same["team_id"])
        ]
        explained = int(same["still_there"].sum())
        lines.append("PHOTOS IDENTICAL TO THE CURRENT ONE")
        lines.append("-" * 72)
        lines.append(f"identical                   {len(same):,}")
        lines.append(f"  player still on that team {explained:,} "
                     f"({100 * explained / max(1, len(same)):.1f}%)")
        lines.append(f"  unexplained               {len(same) - explained:,}")
        lines.append("")
        if explained / max(1, len(same)) >= 0.9:
            lines.append("Almost all of them are players who never left, so the")
            lines.append("identical photo is correct rather than a stale one.")
        else:
            lines.append("A large share are NOT explained by the player staying")
            lines.append("put, which means the season path may be falling back")
            lines.append("to the current image for some team-years. Worth")
            lines.append("looking at before wiring this in.")
        lines.append("")

    misses = frame[~frame["usable"]]
    lines.append("MISSES")
    lines.append("-" * 72)
    lines.append(f"no usable image             {len(misses):,}")
    if len(misses):
        by_status = misses["status"].value_counts().to_dict()
        lines.append(f"by status                   {by_status}")
        lines.append("")
        lines.append("These fall back to the current photo in the dashboard,")
        lines.append("and to the jersey number if that fails too. First 15:")
        for row in misses.head(15).itertuples():
            lines.append(f"    {row.season}  {row.team_abbrev:<4} "
                         f"{row.person_id}  status={row.status}")
    lines.append("")
    lines.append(f"map written to {out_path}")

    report_path = REPORTS / "season_headshot_coverage.txt"
    report_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {report_path}")


if __name__ == "__main__":
    main()
