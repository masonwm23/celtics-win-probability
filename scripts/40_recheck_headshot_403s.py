"""
Re-check the 403s. Are they real absences, or were we rate limited?

WHY
---
`scripts/39_build_season_headshots.py` came back with 3,680 of 4,009 usable and
329 misses, and every single miss was a 403. That uniformity is the reason to
look twice rather than move on.

Two explanations fit a wall of 403s and they have opposite consequences:

  REAL ABSENCE   The object does not exist for that team-year. S3-backed CDNs
                 answer a missing key with 403 rather than 404 when listing is
                 denied, so this is the expected shape of "no photo". Those
                 rows fall back to the current photo and nothing is wrong.

  RATE LIMITING  Four threads for twenty-four minutes annoyed the CDN and it
                 started refusing us. Then the map is wrong, the misses are an
                 artefact of how we asked, and the fallbacks are hiding photos
                 that do exist.

This project has already been burnt by exactly the second case: a flood of 429s
in the YouTube fill was recorded as "no video", written to the progress file,
and skipped forever on re-run. That is why this script exists instead of a
sentence assuming the first explanation.

THE TEST
--------
Re-request every 403 slowly, one at a time, with a real pause between each. No
threads. If they were rate limiting, a calm re-request now succeeds. If they
are real absences, they stay 403 no matter how politely we ask.

A control runs alongside: an equal number of URLs that DID work first time,
re-requested the same way. If the control also fails now, the problem is the
network today and the whole comparison is void, which the report says outright
rather than letting a clean-looking zero stand.

HOW TO RUN
----------
    python scripts/40_recheck_headshot_403s.py

329 misses plus 329 controls at roughly one second each, so about eleven
minutes. It updates `data/processed/season_headshots.csv` ONLY if a re-request
succeeds, and says exactly how many rows it changed.

OUTPUT
------
    reports/headshot_403_recheck.txt
    data/processed/season_headshots.csv   updated in place if anything recovered
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "data" / "processed" / "season_headshots.csv"
REPORTS = ROOT / "reports"

DELAY_SECONDS = 1.0
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


def main() -> None:
    if not MAP_PATH.exists():
        raise SystemExit(f"missing {MAP_PATH}. Run script 39 first.")

    REPORTS.mkdir(exist_ok=True)
    frame = pd.read_csv(MAP_PATH)

    misses = frame.loc[~frame["usable"].astype(bool)].copy()
    hits = frame.loc[frame["usable"].astype(bool)].copy()

    if misses.empty:
        raise SystemExit("no misses to re-check. Nothing to do.")

    # The control is drawn EVENLY ACROSS SEASONS from the rows that worked, not
    # from the head of a season-sorted frame, which would take every control
    # from 2016-17 and tell us nothing about the rest.
    per_season = max(1, len(misses) // max(1, hits["season"].nunique()))
    control_parts = []
    for season in sorted(hits["season"].unique()):
        control_parts.append(hits.loc[hits["season"].eq(season)].head(per_season))
    control = pd.concat(control_parts).head(len(misses))

    logger.info("re-checking %d misses and %d controls, one at a time",
                len(misses), len(control))
    logger.info("about %d minutes", round(
        (len(misses) + len(control)) * DELAY_SECONDS / 60))

    recovered = []
    still_403 = 0
    other = 0
    for n, row in enumerate(misses.itertuples(), start=1):
        result = fetch(row.url)
        if is_image(result):
            recovered.append((row.Index, result))
        elif result["status"] == 403:
            still_403 += 1
        else:
            other += 1
        if n % 50 == 0 or n == len(misses):
            logger.info("  misses %d/%d, recovered %d",
                        n, len(misses), len(recovered))
        time.sleep(DELAY_SECONDS)

    control_ok = 0
    for n, row in enumerate(control.itertuples(), start=1):
        if is_image(fetch(row.url)):
            control_ok += 1
        if n % 50 == 0 or n == len(control):
            logger.info("  control %d/%d, still working %d",
                        n, len(control), control_ok)
        time.sleep(DELAY_SECONDS)

    # ---- write back only what actually recovered --------------------------
    for index, result in recovered:
        frame.loc[index, "status"] = result["status"]
        frame.loc[index, "content_type"] = result["content_type"]
        frame.loc[index, "bytes"] = result["bytes"]
        frame.loc[index, "md5"] = result["md5"]
        frame.loc[index, "usable"] = True
        frame.loc[index, "error"] = "recovered on recheck"
    if recovered:
        frame.to_csv(MAP_PATH, index=False)

    # ---- report ------------------------------------------------------------
    control_rate = control_ok / max(1, len(control))
    lines: list[str] = []
    lines.append("403 RE-CHECK")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"misses re-requested      {len(misses):,}")
    lines.append(f"  recovered              {len(recovered):,}")
    lines.append(f"  still 403              {still_403:,}")
    lines.append(f"  other outcome          {other:,}")
    lines.append("")
    lines.append(f"control re-requested     {len(control):,}")
    lines.append(f"  still working          {control_ok:,} "
                 f"({100 * control_rate:.1f}%)")
    lines.append("")
    lines.append("VERDICT")
    lines.append("-" * 72)

    if control_rate < 0.9:
        lines.append("THE CONTROL BROKE. URLs that worked twenty minutes ago are")
        lines.append("failing now, so this run says something about the network")
        lines.append("today and nothing about the NBA's archive. Re-run later.")
        lines.append("Do not treat the miss count as settled.")
    elif len(recovered) == 0:
        lines.append("Every 403 is still a 403, while the controls all still")
        lines.append("work. These player-seasons genuinely have no photo at")
        lines.append("that path. The misses are real absences, not our fault,")
        lines.append("and falling back to the current photo is the right")
        lines.append("behaviour for them.")
    elif len(recovered) < 0.05 * len(misses):
        lines.append(f"{len(recovered)} of {len(misses)} came back on a calm")
        lines.append("re-request. That is a small tail rather than systematic")
        lines.append("throttling. The map has been updated with them.")
    else:
        lines.append(f"{len(recovered)} of {len(misses)} recovered, which is too")
        lines.append("many to call a tail. The first run was being throttled,")
        lines.append("so the miss list was partly an artefact of how fast we")
        lines.append("asked. The map is updated, but re-run script 39 with")
        lines.append("WORKERS = 1 before trusting the coverage figure.")
    lines.append("")

    usable_now = int(frame["usable"].astype(bool).sum())
    lines.append(f"usable after recheck     {usable_now:,} of {len(frame):,} "
                 f"({100 * usable_now / max(1, len(frame)):.1f}%)")

    report_path = REPORTS / "headshot_403_recheck.txt"
    report_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {report_path}")


if __name__ == "__main__":
    main()
