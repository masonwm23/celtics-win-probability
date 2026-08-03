"""
Phase 11c: does the clip actually PLAY, as opposed to answering a HEAD request.

WHY THIS EXISTS
---------------
The Phase 11 probe checked playback with a HEAD request and reported 40 out of
40 at HTTP 200, median clip size 31.6 MB. Reading the row-level CSV afterwards
showed something the report did not:

    min 31.6 MB, max 31.6 MB, across 40 DIFFERENT clips whose durations range
    from 4.4 to 16.6 seconds.

Forty distinct videos of wildly different lengths cannot all be exactly the
same number of bytes. That Content-Length is not the clip's size. Something in
front of the file is answering HEAD generically, which means the 40/40 measured
the CDN's willingness to answer a HEAD and not whether the video is there.

So the third of the probe's three questions is still open. This closes it.

WHAT IT DOES INSTEAD
--------------------
A ranged GET for the first two kilobytes of each clip.

  - a 206 with a Content-Range gives the file's REAL total size, which should
    differ from clip to clip;
  - the bytes themselves are checked for the MPEG-4 `ftyp` box, which is the
    first thing in a real .mp4 and cannot be faked by a CDN that is merely
    being agreeable about headers.

It issues the HEAD too, side by side, so the discrepancy is on the record
rather than asserted.

READ ONLY. About 32 requests, downloads roughly 32 KB in total. Writes one
file, reports/video_playback.txt.
"""

import logging
import time
from collections import Counter

import pandas as pd

from src import config

logger = logging.getLogger(__name__)

SAMPLE_PER_SEASON = 2
RANGE_BYTES = 2047          # first 2 KB is far more than enough to see `ftyp`

REQUEST_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"),
    "Referer": "https://www.nba.com/",
}


def looks_like_mp4(head_bytes: bytes) -> bool:
    """
    True if these bytes begin an MPEG-4 file.

    An .mp4 opens with a box whose type is `ftyp`, at offset 4. A CDN returning
    an error page, a redirect body or a zero-filled placeholder will not have
    it, and that is exactly the failure a 200 status hides.
    """
    if not head_bytes or len(head_bytes) < 12:
        return False
    return b"ftyp" in head_bytes[:64]


def total_size_from_content_range(value) -> int:
    """
    Parse the trailing total out of `bytes 0-2047/48213456`.

    Returns 0 when absent or unparseable rather than raising, because a missing
    Content-Range is itself one of the results worth reporting.
    """
    if not value or "/" not in str(value):
        return 0
    tail = str(value).rsplit("/", 1)[-1].strip()
    return int(tail) if tail.isdigit() else 0


def check_one(url: str) -> dict:
    """HEAD and a ranged GET for the same URL. Never raises."""
    import requests
    result = {
        "head_status": None, "head_length": None,
        "range_status": None, "range_bytes_returned": None,
        "real_total_bytes": None, "content_range": None,
        "is_mp4": False, "error": None,
    }
    try:
        head = requests.head(url, headers=REQUEST_HEADERS, timeout=20,
                             allow_redirects=True)
        result["head_status"] = head.status_code
        result["head_length"] = head.headers.get("Content-Length")
    except Exception as exc:                       # noqa: BLE001
        result["error"] = f"HEAD {type(exc).__name__}: {exc}"

    try:
        headers = dict(REQUEST_HEADERS)
        headers["Range"] = f"bytes=0-{RANGE_BYTES}"
        got = requests.get(url, headers=headers, timeout=20,
                           allow_redirects=True)
        body = got.content or b""
        result.update({
            "range_status": got.status_code,
            "range_bytes_returned": len(body),
            "content_range": got.headers.get("Content-Range"),
            "real_total_bytes": total_size_from_content_range(
                got.headers.get("Content-Range")),
            "is_mp4": looks_like_mp4(body),
        })
    except Exception as exc:                       # noqa: BLE001
        prior = result["error"]
        result["error"] = ((prior + "; ") if prior else "") + \
            f"GET {type(exc).__name__}: {exc}"
    return result


def sample_clips(probe: pd.DataFrame) -> pd.DataFrame:
    """Two matched clips per season, oldest first within each."""
    matched = probe.loc[probe["status"].eq("matched") & probe["url"].notna()]
    if matched.empty:
        return matched
    return (matched.sort_values(["season", "game_date", "event_index"])
            .groupby("season", group_keys=False).head(SAMPLE_PER_SEASON))


def build_report(frame: pd.DataFrame) -> str:
    total = len(frame)
    playable = int(frame["is_mp4"].sum())
    sizes = pd.to_numeric(frame["real_total_bytes"], errors="coerce").fillna(0)
    distinct = int(sizes.loc[sizes > 0].nunique())

    lines = [
        "=" * 78,
        "PHASE 11c - DOES THE CLIP ACTUALLY PLAY",
        "=" * 78,
        "",
        "  The Phase 11 probe checked playback with a HEAD request and got",
        "  40/40 at HTTP 200. Every one of those 40 reported the SAME",
        "  Content-Length, 31.6 MB, for clips lasting between 4.4 and 16.6",
        "  seconds. That is impossible, so the HEAD was not measuring the file.",
        "",
        "  This asks for the first 2 KB of each clip instead and looks for the",
        "  MPEG-4 `ftyp` box in the bytes that come back.",
        "",
        f"  clips tested            {total}",
        f"  real MP4 bytes returned {playable}  ({playable / total:.0%})"
        if total else "  clips tested            0",
        f"  distinct real file sizes among them: {distinct}",
        "",
    ]
    if total and distinct <= 1 and playable:
        lines += [
            "  !! Every clip reports the same real size. Treat the result as",
            "  unproven and look at the rows below by hand.",
            "",
        ]

    lines += [
        "=" * 78,
        "HEAD VERSUS RANGED GET, CLIP BY CLIP",
        "=" * 78,
        "",
        f"  {'season':<9}{'HEAD':>6}{'HEAD len':>12}{'GET':>5}{'got':>6}"
        f"{'real size':>12}{'mp4':>5}",
    ]
    for row in frame.itertuples():
        head_len = row.head_length or "-"
        real = (f"{row.real_total_bytes / 1e6:.1f} MB"
                if row.real_total_bytes else "-")
        lines.append(
            f"  {str(row.season):<9}{str(row.head_status):>6}{str(head_len):>12}"
            f"{str(row.range_status):>5}{str(row.range_bytes_returned):>6}"
            f"{real:>12}{'yes' if row.is_mp4 else 'NO':>5}")

    errors = frame.loc[frame["error"].notna()]
    if len(errors):
        lines += ["", f"  errors: {len(errors)}"]
        for row in errors.head(5).itertuples():
            lines.append(f"    {row.season}: {str(row.error)[:90]}")

    lines += ["", "=" * 78, "BY SEASON", "=" * 78, "",
              f"  {'season':<10}{'tested':>8}{'real mp4':>10}{'rate':>8}"]
    for season, group in frame.groupby("season"):
        ok = int(group["is_mp4"].sum())
        lines.append(f"  {str(season):<10}{len(group):>8}{ok:>10}"
                     f"{ok / len(group):>7.0%}")

    statuses = Counter(frame["range_status"].dropna())
    lines += ["", "  ranged GET statuses: " +
              ", ".join(f"{int(k)}x{v}" for k, v in sorted(statuses.items()))]

    lines += [
        "",
        "=" * 78,
        "WHAT THIS STILL DOES NOT TELL YOU",
        "=" * 78,
        "",
        "  Real MP4 bytes over HTTP is strong evidence the clip is there. It",
        "  is not proof a <video> tag in a browser will play it: an expiring",
        "  signature, a stricter Referer check on the real GET, or a codec the",
        "  browser will not decode would all pass this test.",
        "",
        "  The only conclusive test is one clip in an actual page, which is an",
        "  app change and is not being made without your say-so.",
        "",
        "  Nothing was changed by this script.",
        "=" * 78,
    ]
    return "\n".join(lines)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    config.ensure_dirs()

    source = config.INTERIM_DIR / "video_probe.csv"
    if not source.exists():
        raise SystemExit(
            f"{source} not found. Run scripts/24_probe_video.py first.")

    probe = pd.read_csv(source)
    clips = sample_clips(probe)
    if clips.empty:
        raise SystemExit(
            "No matched clips in video_probe.csv, so there is nothing to test.")

    logger.info("testing %d clips across %d seasons",
                len(clips), clips["season"].nunique())

    records = []
    for row in clips.itertuples():
        result = check_one(row.url)
        records.append({"season": row.season, "game_id": row.game_id,
                        "action_number": row.action_number,
                        "duration_ms": getattr(row, "duration", None),
                        "url": row.url, **result})
        logger.info("  %s %s: mp4=%s", row.season, row.game_id,
                    result["is_mp4"])
        time.sleep(0.3)

    frame = pd.DataFrame(records)
    report = build_report(frame)
    print(report)
    out = config.REPORTS_DIR / "video_playback.txt"
    out.write_text(report + "\n", encoding="utf-8")
    frame.to_csv(config.INTERIM_DIR / "video_playback.csv", index=False)
    print(f"\nSaved to: {out}")
    print("\nREAD ONLY. Nothing in the app, the API, the model or the research")
    print("outputs was modified.")
    return frame


if __name__ == "__main__":
    main()
