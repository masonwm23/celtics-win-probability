"""
Phase 11f: fetch ALL of them, not a biased sixteen.

WHY THIS EXISTS
---------------
Phase 11d concluded the clips are placeholders. It tested sixteen URLs. Those
sixteen were chosen by sorting on (season, game_date, event_index) and taking
the first two per season, which means they were the two EARLIEST events of the
earliest game in each season: event ids 1, 2, 4 and 7. Opening tips and first
possessions.

That is a biased sample and the bias is in the direction that matters. If the
NBA publishes highlight clips for scoring plays but not for the opening
sequence, a sample of opening-sequence events would show exactly what was
observed and would say nothing about the other 320 matched events.

So this fetches every matched URL and counts how many are the placeholder.

THE REFERENCE HASH IS MEASURED, NOT HARDCODED
---------------------------------------------
A fabricated URL, real in shape but naming a uuid that does not exist, is
fetched first. Whatever comes back IS the placeholder by definition, and its
hash becomes the reference. Hardcoding a hash from an earlier run would break
silently the day the NBA re-encodes the card.

If the fabricated URL returns nothing, the run aborts rather than guessing,
because without a reference every clip would be classified as real.

WHAT COUNTS AS A REAL CLIP
--------------------------
First 2 KB whose md5 differs from the placeholder reference. That is a weak
test in one direction only: two genuinely different clips could in principle
share a header, but nothing that differs from the placeholder is the
placeholder. False negatives are possible, false positives are not, and for
"is there anything here at all" that is the right way round.

READ ONLY. About 340 small ranged requests, 5 to 8 minutes. Writes
reports/video_sweep.txt and data/interim/video_sweep.csv. Nothing else.
"""

import hashlib
import logging
import re
import time

import pandas as pd

from src import config

logger = logging.getLogger(__name__)

CHUNK = 2048
DELAY = 0.15

REQUEST_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"),
    "Referer": "https://www.nba.com/",
}


def digest(payload: bytes) -> str:
    return hashlib.md5(payload).hexdigest()[:16] if payload else ""


def fabricate(url: str) -> str:
    """A real URL shape naming a uuid that does not exist."""
    return re.sub(
        r"/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_",
        "/00000000-0000-0000-0000-000000000000_", url, count=1)


def fetch_head_bytes(url: str) -> dict:
    """First 2 KB. Never raises."""
    import requests
    headers = dict(REQUEST_HEADERS)
    headers["Range"] = f"bytes=0-{CHUNK - 1}"
    try:
        response = requests.get(url, headers=headers, timeout=20,
                                allow_redirects=True)
        body = response.content or b""
        return {"status": response.status_code, "bytes": len(body),
                "hash": digest(body), "error": None}
    except Exception as exc:                       # noqa: BLE001
        return {"status": None, "bytes": 0, "hash": "",
                "error": f"{type(exc).__name__}: {exc}"}


def classify(row_hash: str, placeholder_hash: str) -> str:
    """placeholder | real | no_response"""
    if not row_hash:
        return "no_response"
    return "placeholder" if row_hash == placeholder_hash else "real"


def build_report(frame: pd.DataFrame, placeholder_hash: str) -> str:
    total = len(frame)
    real = int(frame["verdict"].eq("real").sum())
    placeholder = int(frame["verdict"].eq("placeholder").sum())
    missing = int(frame["verdict"].eq("no_response").sum())

    lines = [
        "=" * 78,
        "PHASE 11f - EVERY MATCHED CLIP, NOT A BIASED SIXTEEN",
        "=" * 78,
        "",
        "  Phase 11d tested sixteen URLs and found one placeholder file. Those",
        "  sixteen were the two earliest events of the earliest game in each",
        "  season: event ids 1, 2, 4 and 7. Opening tips. If the NBA publishes",
        "  clips for scoring plays but not for the opening sequence, that",
        "  sample would look exactly the same and mean nothing.",
        "",
        f"  placeholder reference hash: {placeholder_hash}",
        "  (measured this run from a fabricated uuid, not hardcoded)",
        "",
        f"  clips fetched   {total:,}",
        f"  REAL            {real:,}  ({real / total:.1%})" if total else "",
        f"  placeholder     {placeholder:,}  ({placeholder / total:.1%})"
        if total else "",
        f"  no response     {missing:,}",
        "",
    ]

    if real == 0:
        lines += [
            "  Nothing. Every matched clip across every season, every event",
            "  type and every game is the same VIDEO NOT AVAILABLE card. The",
            "  earlier conclusion holds, now on the full set rather than on a",
            "  biased sixteen.",
            "",
        ]
    else:
        lines += [
            f"  {real:,} clips are NOT the placeholder. The earlier conclusion",
            "  was wrong, and it was wrong because of how the sixteen were",
            "  chosen. See the breakdown below for where the real clips are.",
            "",
        ]

    lines += ["=" * 78, "BY SEASON", "=" * 78, "",
              f"  {'season':<10}{'fetched':>9}{'real':>7}{'placeholder':>13}"
              f"{'real rate':>11}"]
    for season, group in frame.groupby("season"):
        r = int(group["verdict"].eq("real").sum())
        lines.append(f"  {str(season):<10}{len(group):>9}{r:>7}"
                     f"{int(group['verdict'].eq('placeholder').sum()):>13}"
                     f"{r / len(group):>10.1%}")

    lines += ["", "=" * 78, "BY EVENT TYPE", "=" * 78, "",
              f"  {'action type':<20}{'fetched':>9}{'real':>7}{'real rate':>11}"]
    for action_type, group in frame.groupby("action_type"):
        r = int(group["verdict"].eq("real").sum())
        lines.append(f"  {str(action_type)[:19]:<20}{len(group):>9}{r:>7}"
                     f"{r / len(group):>10.1%}")

    lines += ["", "=" * 78, "BY PERIOD", "=" * 78, "",
              f"  {'period':<10}{'fetched':>9}{'real':>7}{'real rate':>11}"]
    for period, group in frame.groupby("period"):
        r = int(group["verdict"].eq("real").sum())
        lines.append(f"  {str(period):<10}{len(group):>9}{r:>7}"
                     f"{r / len(group):>10.1%}")

    if real:
        lines += ["", "=" * 78, "REAL CLIPS, OPEN THESE IN A BROWSER",
                  "=" * 78, "",
                  "  Paste a few into Chrome. If they show the play described,",
                  "  the feature is alive for this subset.", ""]
        for row in frame.loc[frame["verdict"].eq("real")].head(8).itertuples():
            lines.append(f"  {row.season}  {row.action_type}  "
                         f"period {row.period}")
            lines.append(f"    {str(row.our_description)[:72]}")
            lines.append(f"    {row.url}")
            lines.append("")

    distinct_real = len({h for h in
                         frame.loc[frame["verdict"].eq("real"), "hash"] if h})
    lines += [
        "",
        f"  distinct hashes among the real clips: {distinct_real}",
    ]
    if real and distinct_real == 1:
        lines.append("  Only one, so these are a SECOND shared file, not real "
                     "clips. Treat as placeholder.")

    lines += [
        "",
        "=" * 78,
        "WHAT THIS DOES NOT TELL YOU",
        "=" * 78,
        "",
        "  A hash differing from the placeholder means the file is not the",
        "  placeholder. It does not prove the file shows the play our record",
        "  names. That evidence is separate and already collected: the NBA's",
        "  description matched ours exactly on all matched responses.",
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
    matched = probe.loc[probe["status"].eq("matched")
                        & probe["url"].notna()].reset_index(drop=True)
    if matched.empty:
        raise SystemExit("no matched clips in video_probe.csv")

    # Establish the placeholder by asking for something that cannot exist.
    reference = fetch_head_bytes(fabricate(matched.iloc[0]["url"]))
    if not reference["hash"]:
        raise SystemExit(
            "The fabricated control URL returned nothing, so there is no "
            "placeholder reference to compare against. Without it every clip "
            "would be classified as real. Stopping rather than guessing.")
    placeholder_hash = reference["hash"]
    logger.info("placeholder reference hash: %s", placeholder_hash)
    logger.info("fetching %d matched clips", len(matched))

    records = []
    for i, row in enumerate(matched.itertuples(), start=1):
        result = fetch_head_bytes(row.url)
        records.append({
            "season": row.season, "game_id": row.game_id,
            "action_number": row.action_number, "action_type": row.action_type,
            "period": row.period, "our_description": row.our_description,
            "url": row.url, "verdict": classify(result["hash"],
                                                placeholder_hash),
            **result})
        if i % 25 == 0:
            so_far = sum(1 for r in records if r["verdict"] == "real")
            logger.info("  %d/%d fetched, %d real so far", i, len(matched),
                        so_far)
        time.sleep(DELAY)

    frame = pd.DataFrame(records)
    report = build_report(frame, placeholder_hash)
    print(report)
    out = config.REPORTS_DIR / "video_sweep.txt"
    out.write_text(report + "\n", encoding="utf-8")
    frame.to_csv(config.INTERIM_DIR / "video_sweep.csv", index=False)
    print(f"\nSaved to: {out}")
    print("\nREAD ONLY. Nothing in the app, the API, the model or the research")
    print("outputs was modified.")
    return frame


if __name__ == "__main__":
    main()
