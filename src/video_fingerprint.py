"""
Phase 11d: are these sixteen URLs sixteen different videos, or one video?

THE PROBLEM
-----------
Phase 11c asked for the first 2 KB of sixteen clips. All sixteen returned HTTP
206 with real MPEG-4 bytes. Good. But every single one reported the same total
size in its Content-Range header:

    31,580,089 bytes, identical across all sixteen, for clips lasting 4.4 to
    16.6 seconds.

That has two possible explanations and they are not close to equivalent:

  A. the CDN reports a fixed, meaningless total in Content-Range, but the bytes
     it serves are the real clip                     -> harmless, build it

  B. every URL is serving the SAME file, and the "clip" we would put on screen
     has nothing to do with the play                 -> fatal, do not build it

31.6 MB for a nine second 960x540 clip is about 28 Mbps, which is implausible
on its face, so A is more likely. Likely is not good enough for something that
would put a video next to a play and imply it IS that play.

HOW THIS TELLS THEM APART
-------------------------
Three ranges per clip, each hashed:

  0 - 2 KB            the file header. Different videos have different moov
                      atoms, durations and track tables, so identical hashes
                      here across sixteen clips is damning.
  1 MB - 1 MB + 2 KB  compressed frame data well inside the file. Two different
                      videos agreeing here by chance is not a thing.
  31 MB               the honesty test for the claimed size. If the file really
                      is 31.6 MB this returns data; if the claim is fiction it
                      returns 416 Range Not Satisfiable.

Then ONE clip is streamed in full, with a hard cap, to measure what it actually
weighs. That settles the size question outright and tells us what a video panel
would really cost to load.

VERDICT RULE, written before the data
-------------------------------------
Distinct header hashes must equal the number of clips tested. Sixteen out of
sixteen, or the feature does not get built. Anything less and this reports
FATAL, because a clip that is not the play is worse than no clip.

READ ONLY. Writes reports/video_fingerprint.txt and
data/interim/video_fingerprint.csv. Nothing else is touched.
"""

import hashlib
import logging
import time

import pandas as pd

from src import config

logger = logging.getLogger(__name__)

CHUNK = 2048

# Byte offsets probed in every clip. The third is deliberately past any
# plausible real size for a ten second clip; it exists to test the 31.6 MB
# claim rather than to fetch anything useful.
OFFSETS = {
    "header": 0,
    "deep": 1_000_000,
    "claimed_tail": 31_000_000,
}

# A full download of one clip, capped so a genuinely huge file cannot run away
# with the connection.
FULL_DOWNLOAD_CAP = 80 * 1024 * 1024

REQUEST_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"),
    "Referer": "https://www.nba.com/",
}


def digest(payload: bytes) -> str:
    """Short md5 of a byte range. Empty input gets a marker, never a hash."""
    if not payload:
        return ""
    return hashlib.md5(payload).hexdigest()[:16]


def fetch_range(url: str, start: int, length: int = CHUNK) -> dict:
    """One ranged GET. Never raises."""
    import requests
    headers = dict(REQUEST_HEADERS)
    headers["Range"] = f"bytes={start}-{start + length - 1}"
    try:
        response = requests.get(url, headers=headers, timeout=25,
                                allow_redirects=True)
        body = response.content or b""
        return {"status": response.status_code, "bytes": len(body),
                "hash": digest(body), "error": None}
    except Exception as exc:                       # noqa: BLE001
        return {"status": None, "bytes": 0, "hash": "",
                "error": f"{type(exc).__name__}: {exc}"}


def measure_full(url: str, cap: int = FULL_DOWNLOAD_CAP) -> dict:
    """
    Stream one clip end to end and count what actually arrives.

    Streamed rather than held in memory, and capped, because the whole reason
    we are here is that the advertised size is not trusted.
    """
    import requests
    try:
        with requests.get(url, headers=REQUEST_HEADERS, timeout=60,
                          stream=True, allow_redirects=True) as response:
            received = 0
            hasher = hashlib.md5()
            for block in response.iter_content(chunk_size=65536):
                received += len(block)
                hasher.update(block)
                if received >= cap:
                    return {"status": response.status_code,
                            "actual_bytes": received, "complete": False,
                            "advertised": response.headers.get("Content-Length"),
                            "hash": hasher.hexdigest()[:16], "error": None}
            return {"status": response.status_code, "actual_bytes": received,
                    "complete": True,
                    "advertised": response.headers.get("Content-Length"),
                    "hash": hasher.hexdigest()[:16], "error": None}
    except Exception as exc:                       # noqa: BLE001
        return {"status": None, "actual_bytes": 0, "complete": False,
                "advertised": None, "hash": "", "error":
                f"{type(exc).__name__}: {exc}"}


def verdict(frame: pd.DataFrame) -> tuple:
    """
    (verdict, reason). Decided by the rule stated in the module docstring, not
    by whatever the numbers turn out to be.
    """
    tested = len(frame)
    if not tested:
        return "FATAL", "no clips were tested"

    header_hashes = {h for h in frame["header_hash"] if h}
    if len(header_hashes) < tested:
        return ("FATAL",
                f"only {len(header_hashes)} distinct file headers across "
                f"{tested} clips: these URLs are not all different videos")

    deep_hashes = {h for h in frame["deep_hash"] if h}
    deep_ok = int(frame["deep_status"].isin([200, 206]).sum())
    if deep_ok and len(deep_hashes) < deep_ok:
        return ("FATAL",
                f"only {len(deep_hashes)} distinct hashes 1 MB into "
                f"{deep_ok} clips")

    return ("PASS",
            f"{tested} clips, {len(header_hashes)} distinct headers, "
            f"{len(deep_hashes)} distinct interiors")


def build_report(frame: pd.DataFrame, full: dict) -> str:
    result, reason = verdict(frame)
    tested = len(frame)

    lines = [
        "=" * 78,
        "PHASE 11d - SIXTEEN CLIPS, OR ONE CLIP SIXTEEN TIMES",
        "=" * 78,
        "",
        "  Phase 11c returned real MP4 bytes for all sixteen clips and an",
        "  IDENTICAL advertised size of 31,580,089 bytes for every one of",
        "  them. Either that header is meaningless, or every URL serves the",
        "  same video. This hashes three byte ranges per clip to find out.",
        "",
        f"  VERDICT: {result}",
        f"  {reason}",
        "",
    ]
    if result == "FATAL":
        lines += [
            "  Do not build the video panel. A clip that is not the play it",
            "  sits beside is worse than no clip at all.",
            "",
        ]

    lines += [
        "=" * 78,
        "HASHES BY BYTE OFFSET",
        "=" * 78,
        "",
        "  header      first 2 KB, the MP4 header. Different videos differ here.",
        "  deep        2 KB from 1 MB in. Chance agreement is not credible.",
        "  tail@31MB   tests the 31.6 MB claim. 416 means the claim is fiction.",
        "",
        f"  {'season':<9}{'game':<12}{'header':<18}{'deep':<18}{'tail':>10}",
    ]
    for row in frame.itertuples():
        tail = ("416" if row.claimed_tail_status == 416
                else str(row.claimed_tail_status))
        lines.append(
            f"  {str(row.season):<9}{str(row.game_id):<12}"
            f"{(row.header_hash or '-'):<18}{(row.deep_hash or '-'):<18}"
            f"{tail:>10}")

    lines += [
        "",
        f"  distinct header hashes : {len({h for h in frame['header_hash'] if h})}"
        f" of {tested}",
        f"  distinct deep hashes   : {len({h for h in frame['deep_hash'] if h})}"
        f" of {int(frame['deep_status'].isin([200, 206]).sum())} that returned data",
    ]

    tail_statuses = frame["claimed_tail_status"].value_counts(dropna=False)
    lines += ["", "  responses at the 31 MB offset:"]
    for status, count in tail_statuses.items():
        note = ""
        if status == 416:
            note = "  <- the advertised 31.6 MB does not exist"
        elif status in (200, 206):
            note = "  <- there really is data that far in"
        lines.append(f"    {status}: {count}{note}")

    lines += ["", "=" * 78, "WHAT ONE CLIP ACTUALLY WEIGHS", "=" * 78, ""]
    if full.get("error"):
        lines.append(f"  download failed: {full['error']}")
    else:
        actual = full.get("actual_bytes") or 0
        lines += [
            f"  HTTP {full.get('status')}",
            f"  advertised Content-Length : {full.get('advertised')}",
            f"  bytes actually received   : {actual:,}  "
            f"({actual / 1e6:.2f} MB)",
            f"  download completed        : {full.get('complete')}",
        ]
        advertised = full.get("advertised")
        if advertised and str(advertised).isdigit():
            claimed = int(advertised)
            if actual and abs(claimed - actual) > max(1024, claimed * 0.01):
                lines += [
                    "",
                    f"  The header claimed {claimed:,} bytes and "
                    f"{actual:,} arrived. The advertised size is wrong, which",
                    "  is consistent with the identical 31.6 MB seen everywhere",
                    "  else and means clip size cannot be read from headers.",
                ]
        if actual:
            lines += [
                "",
                f"  A panel loading one clip per event moves about "
                f"{actual / 1e6:.1f} MB each time at this encoding.",
                "  The 320x180 encoding is also in the probe CSV if that is",
                "  too heavy for scrubbing.",
            ]

    lines += [
        "",
        "=" * 78,
        "WHAT THIS STILL DOES NOT TELL YOU",
        "=" * 78,
        "",
        "  Distinct bytes prove these are distinct videos. They do not prove",
        "  each video shows the play our record says it shows. The evidence",
        "  for that is separate and already collected: the NBA's own",
        "  description matched ours exactly on all 336 matched clips.",
        "",
        "  Nor is this proof a browser <video> tag will play them. Only one",
        "  clip in a real page settles that.",
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

    source = config.INTERIM_DIR / "video_playback.csv"
    if not source.exists():
        raise SystemExit(
            f"{source} not found. Run scripts/26_verify_playback.py first.")

    clips = pd.read_csv(source)
    clips = clips.loc[clips["url"].notna()]
    if clips.empty:
        raise SystemExit("no URLs in video_playback.csv")

    logger.info("fingerprinting %d clips at %d byte offsets each",
                len(clips), len(OFFSETS))

    records = []
    for row in clips.itertuples():
        record = {"season": row.season, "game_id": row.game_id,
                  "action_number": getattr(row, "action_number", None),
                  "url": row.url}
        for name, offset in OFFSETS.items():
            result = fetch_range(row.url, offset)
            record[f"{name}_status"] = result["status"]
            record[f"{name}_hash"] = result["hash"]
            record[f"{name}_bytes"] = result["bytes"]
            record[f"{name}_error"] = result["error"]
            time.sleep(0.2)
        records.append(record)
        logger.info("  %s %s header=%s deep=%s tail=%s", row.season,
                    row.game_id, record["header_hash"] or "-",
                    record["deep_hash"] or "-", record["claimed_tail_status"])

    frame = pd.DataFrame(records)

    logger.info("downloading one clip in full to measure it")
    full = measure_full(clips.iloc[0]["url"])
    logger.info("  received %s bytes", f"{full.get('actual_bytes', 0):,}")

    report = build_report(frame, full)
    print(report)
    out = config.REPORTS_DIR / "video_fingerprint.txt"
    out.write_text(report + "\n", encoding="utf-8")
    frame.to_csv(config.INTERIM_DIR / "video_fingerprint.csv", index=False)
    print(f"\nSaved to: {out}")
    print("\nREAD ONLY. Nothing in the app, the API, the model or the research")
    print("outputs was modified.")
    return frame, full


if __name__ == "__main__":
    main()
