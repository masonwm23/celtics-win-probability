"""
Probe: does the NBA publish a SEASON-SPECIFIC headshot, or only the current one?

WHY THIS EXISTS
---------------
Every headshot URL in `data/raw/player_bios.csv` points at the `latest` path:

    https://cdn.nba.com/headshots/nba/latest/1040x760/{person_id}.png

`latest` means the player's CURRENT team. On the dashboard that shows Kyrie
Irving in a Mavericks jersey during a 2017-18 Celtics game, Marcus Smart in a
Lakers jersey, Terry Rozier in a Heat jersey. The photo is of the right person
and the wrong era.

This script does not fix anything. It answers one question with evidence:
is there a public URL that returns the jersey the player actually wore that
season? If there is, the dashboard can use it. If there is not, we say so and
stop, rather than shipping a URL pattern that happens to return 200.

WHAT IT CHECKS, AND WHY IT IS NOT JUST A STATUS CODE
----------------------------------------------------
A 200 proves nothing here. A CDN can answer any path under a prefix with the
same default image, and this project has already been fooled once by exactly
that shape of result. So for every candidate URL this records:

    status, content-type, byte length, and the md5 of the body

and then compares that md5 against the `latest` image for the SAME player.

    different md5  ->  the server has a genuinely different image for that
                       season. Promising.
    same md5       ->  the season path is an alias for `latest`. USELESS, and
                       it would have looked like success.
    not an image   ->  a 200 carrying an HTML error page. Also useless.

The sample is drawn from players who ACTUALLY CHANGED TEAMS inside the
dataset, read out of the bios file itself. That matters: for a player who
never moved, a correct season-specific image and the `latest` image would
legitimately be identical, and the test could not tell a working pattern from
a broken one.

HOW TO RUN
----------
From the project root, in Spyder or a terminal:

    python scripts/38_probe_season_headshots.py

It makes roughly 300 requests with a delay between each and takes two or three
minutes. It writes nothing except the two report files, touches no data, and
downloads no images to disk.

OUTPUT
------
    reports/headshot_probe_rows.csv      every request, one row each
    reports/headshot_probe_summary.txt   the plain-English answer
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
BIOS = ROOT / "data" / "raw" / "player_bios.csv"
REPORTS = ROOT / "reports"

# Candidate URL shapes. NONE of these is known to work. They are the patterns
# worth ruling in or out, and the script exists because guessing is not an
# answer. `{yr}` is filled twice per pattern, once with the season's starting
# year and once with its ending year, because both conventions appear in NBA
# asset paths and we do not know which, if either, is right.
CANDIDATES = [
    ("cdn_team_year_1040", "https://cdn.nba.com/headshots/nba/{team_id}/{yr}/1040x760/{pid}.png"),
    ("cdn_team_year_260", "https://cdn.nba.com/headshots/nba/{team_id}/{yr}/260x190/{pid}.png"),
    ("cdn_year_1040", "https://cdn.nba.com/headshots/nba/{yr}/1040x760/{pid}.png"),
    ("akstatic_team_year_260",
     "https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/{team_id}/{yr}/260x190/{pid}.png"),
    ("akstatic_team_year_1040",
     "https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/{team_id}/{yr}/1040x760/{pid}.png"),
]

CONTROL = ("latest_control", "https://cdn.nba.com/headshots/nba/latest/1040x760/{pid}.png")

SAMPLE_PLAYERS = 28
DELAY_SECONDS = 0.35
TIMEOUT = 15

HEADERS = {
    # A plain browser user agent. The CDN rejects the default requests one.
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    ),
    "Referer": "https://www.nba.com/",
}


def season_years(season: str) -> tuple[int, int]:
    """'2018-19' -> (2018, 2019)."""
    start = int(str(season).split("-")[0])
    return start, start + 1


def pick_movers(bios: pd.DataFrame) -> pd.DataFrame:
    """
    Players who appear on MORE THAN ONE team id across the dataset.

    These are the only useful test subjects. For a player who never moved, a
    correct season image and the current one would look the same, so a broken
    pattern would pass.
    """
    teams_per_player = bios.groupby("person_id")["team_id"].nunique()
    movers = teams_per_player[teams_per_player > 1].index
    rows = bios[bios["person_id"].isin(movers)].copy()

    # Spread across seasons rather than taking the head of a season-ordered
    # frame, which would draw every test case from 2016-17. This project has
    # made that exact mistake twice, so the loop is explicit rather than a
    # groupby whose behaviour depends on the pandas version.
    rows = rows.sort_values(["season", "person_id"])
    seasons = sorted(rows["season"].unique())
    per_season = max(1, SAMPLE_PLAYERS // max(1, len(seasons)))

    picked_index = []
    seen_players = set()
    for season in seasons:
        taken = 0
        for row in rows[rows["season"] == season].itertuples():
            if taken >= per_season:
                break
            if row.person_id in seen_players:
                continue
            seen_players.add(row.person_id)
            picked_index.append(row.Index)
            taken += 1

    return rows.loc[picked_index].head(SAMPLE_PLAYERS).reset_index(drop=True)


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


def main() -> None:
    if not BIOS.exists():
        raise SystemExit(f"missing {BIOS}. Run the raw pull first.")

    REPORTS.mkdir(exist_ok=True)
    bios = pd.read_csv(BIOS, dtype={"jersey": "string"})
    sample = pick_movers(bios)

    if sample.empty:
        raise SystemExit("no players changed teams in the bios file, so this "
                         "probe cannot distinguish a working pattern from a "
                         "broken one. Stopping rather than reporting nothing.")

    print(f"probing {len(sample)} players who changed teams, "
          f"across {sample['season'].nunique()} seasons")

    rows = []
    latest_md5: dict[int, str] = {}

    for n, player in enumerate(sample.itertuples(), start=1):
        pid = int(player.person_id)
        team_id = int(player.team_id)
        start, end = season_years(player.season)

        # The control first, so every candidate has something to compare to.
        control = fetch(CONTROL[1].format(pid=pid))
        latest_md5[pid] = control["md5"]
        rows.append({
            "person_id": pid, "name": player.full_name, "season": player.season,
            "team_id": team_id, "pattern": CONTROL[0], "year_used": "",
            "url": CONTROL[1].format(pid=pid), **control,
            "same_as_latest": True,
        })
        time.sleep(DELAY_SECONDS)

        for name, template in CANDIDATES:
            for label, yr in (("start", start), ("end", end)):
                url = template.format(pid=pid, team_id=team_id, yr=yr)
                result = fetch(url)
                rows.append({
                    "person_id": pid, "name": player.full_name,
                    "season": player.season, "team_id": team_id,
                    "pattern": name, "year_used": label, "url": url, **result,
                    # The whole point. A 200 that is byte-identical to `latest`
                    # is not a season-specific image.
                    "same_as_latest": bool(
                        result["md5"] and result["md5"] == latest_md5.get(pid)
                    ),
                })
                time.sleep(DELAY_SECONDS)

        print(f"  [{n}/{len(sample)}] {player.full_name} {player.season}")

    frame = pd.DataFrame(rows)
    rows_path = REPORTS / "headshot_probe_rows.csv"
    frame.to_csv(rows_path, index=False)

    # ---- summary -----------------------------------------------------------
    lines: list[str] = []
    lines.append("SEASON-SPECIFIC HEADSHOT PROBE")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"players probed          {frame['person_id'].nunique()}")
    lines.append(f"seasons covered         {sorted(frame['season'].unique())}")
    lines.append(f"requests made           {len(frame)}")
    lines.append("")

    control_rows = frame[frame["pattern"] == CONTROL[0]]
    ok_control = (control_rows["status"] == 200).sum()
    lines.append(f"control (latest) 200s    {ok_control} of {len(control_rows)}")
    if ok_control == 0:
        lines.append("")
        lines.append("THE CONTROL FAILED. The current headshot URL did not load")
        lines.append("either, so this machine cannot reach the CDN and NOTHING")
        lines.append("below means anything. Check the network before reading on.")
    lines.append("")

    lines.append("PER PATTERN")
    lines.append("-" * 72)
    lines.append(f"{'pattern':<26} {'yr':<6} {'200s':>6} {'images':>7} "
                 f"{'distinct':>9} {'verdict'}")

    verdicts = []
    for name, _ in CANDIDATES:
        for label in ("start", "end"):
            subset = frame[(frame["pattern"] == name) & (frame["year_used"] == label)]
            if subset.empty:
                continue
            ok = int((subset["status"] == 200).sum())
            images = int(
                ((subset["status"] == 200)
                 & subset["content_type"].str.contains("image", na=False)
                 & (subset["bytes"] > 1000)).sum()
            )
            distinct = int(
                ((subset["status"] == 200)
                 & subset["content_type"].str.contains("image", na=False)
                 & (~subset["same_as_latest"])).sum()
            )
            if distinct >= max(3, int(0.5 * len(subset))):
                verdict = "USABLE, season-specific images"
            elif images > 0 and distinct == 0:
                verdict = "alias for latest, useless"
            elif ok > 0 and images == 0:
                verdict = "200 but not an image"
            else:
                verdict = "no such path"
            verdicts.append((name, label, verdict, distinct, len(subset)))
            lines.append(f"{name:<26} {label:<6} {ok:>6} {images:>7} "
                         f"{distinct:>9} {verdict}")

    lines.append("")
    lines.append("WHAT THIS MEANS")
    lines.append("-" * 72)
    winners = [v for v in verdicts if v[2].startswith("USABLE")]
    if winners:
        lines.append("At least one pattern returns a genuinely different image")
        lines.append("per season. Send this file back and the dashboard can be")
        lines.append("pointed at it, with the current photo kept as a fallback")
        lines.append("for players and seasons the pattern does not cover.")
        for name, label, _, distinct, total in winners:
            lines.append(f"    {name} ({label} year): {distinct} of {total} distinct")
    else:
        lines.append("No candidate pattern returned a season-specific image.")
        lines.append("On this evidence the public CDN publishes one current")
        lines.append("headshot per player and nothing historical, so the photo")
        lines.append("on the dashboard will always show the player's CURRENT")
        lines.append("team. That is a real limitation, not something to paper")
        lines.append("over: the options are to label it, or to go back to")
        lines.append("jersey numbers in the circles.")
    lines.append("")
    lines.append(f"row-level detail: {rows_path}")

    summary_path = REPORTS / "headshot_probe_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {summary_path}")
    print(f"wrote {rows_path}")


if __name__ == "__main__":
    main()
