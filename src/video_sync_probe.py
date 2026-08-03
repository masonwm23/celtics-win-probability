"""
Phase 13a: can a highlight video be seeked to a specific PLAY, legitimately?

THE CORRECTION THIS ANSWERS
---------------------------
The Phase 12 mapping links one game to one ~9 minute recap. That is NOT
play-synchronised video and must never be described as though it were. Seeking
to a play needs a timestamp INSIDE the video, and the mapping contains none.

WHERE A TIMESTAMP COULD LEGITIMATELY COME FROM
----------------------------------------------
Under the project's rules (no downloading, no scraping, embed only) there are
exactly three possibilities, and two of them are dead ends:

  1. CHAPTER MARKERS in the video description. YouTube builds chapters from
     timestamped lines in the description, and the description is a plain
     metadata field on the API. This is free, legitimate, and the only real
     candidate.

  2. CAPTION TRACKS. `captions.list` reveals whether tracks exist, but
     downloading their content requires OAuth as the video's OWNER. We are not
     the NBA. Third-party transcript endpoints are scraping. So existence can
     be reported; content cannot be obtained.

  3. Anything derived from the video itself: on-screen clock OCR, frame
     analysis, audio alignment. All require obtaining the video. Out of bounds,
     and not attempted anywhere in this file.

So this probe asks question 1 first, because if the answer is no, every
downstream timestamp would be inferred from position or duration. That is
guessing, it is forbidden, and the correct output is then "not possible" rather
than a matching algorithm with a plausible-looking accuracy figure.

WHAT IT DOES
------------
  A. Scans EVERY mapped video's description for timestamps. Cheap, and it
     settles feasibility.
  B. For one game, if that game's video has timestamps, matches each chapter
     label to a play-by-play event on player name and description overlap,
     and reports coverage.
  C. Emits a MANUAL SPOT-CHECK list.

WHY C EXISTS, AND WHAT THIS PROBE CANNOT PROVE
----------------------------------------------
Matching a chapter LABEL to a play proves the label describes that play. It
does not prove the timestamp is correct. Nothing available here can prove that,
because verifying it means watching the video at that offset.

So timing accuracy is NOT reported as a computed number. The probe produces
seek URLs for a human to open and confirm. Reporting a timing accuracy figure
this probe cannot measure would be the same error as the earlier phases, where
a well-formed number meant nothing.

READ ONLY. Writes reports/video_sync_probe.txt and
data/interim/video_sync_probe.csv.
"""

import json
import logging
import re

import pandas as pd

from src import config
from src.youtube_probe import load_api_key, safe_api_get

logger = logging.getLogger(__name__)

# A timestamped description line: "1:23 Tatum three" or "0:00 Intro".
TIMESTAMP_LINE = re.compile(
    r"^\s*\[?(\d{1,2}):(\d{2})(?::(\d{2}))?\]?\s*[-–—:|]?\s*(.*\S)?\s*$")

# Captions cost 50 units a call, so only a few videos are checked. Existence is
# all that can be learned anyway: the content needs owner OAuth.
CAPTION_CHECKS = 3

# Share of an event description's words that must appear in a chapter label.
LABEL_AGREEMENT = 0.5

STOPWORDS = {"the", "a", "an", "and", "to", "of", "for", "at", "vs", "with",
             "his", "her", "on", "in", "by", "pts", "ast", "reb"}


# ---------------------------------------------------------------------------
# A. Do timestamps exist at all
# ---------------------------------------------------------------------------

def parse_timestamps(description) -> list:
    """
    [(seconds, label)] from a description, in order.

    Only lines that BEGIN with a timestamp count. A stray "check out 2:15" in
    prose is not a chapter and treating it as one would invent structure.
    """
    if not isinstance(description, str):
        return []
    out = []
    for line in description.splitlines():
        match = TIMESTAMP_LINE.match(line)
        if not match:
            continue
        first, second, third, label = match.groups()
        if third is not None:
            seconds = int(first) * 3600 + int(second) * 60 + int(third)
        else:
            seconds = int(first) * 60 + int(second)
        out.append((seconds, (label or "").strip()))
    return out


def has_real_chapters(stamps: list) -> bool:
    """
    YouTube only renders chapters when there are at least three and the first
    is 0:00. Fewer than that is a description that happens to mention a time.
    """
    return len(stamps) >= 3 and stamps[0][0] == 0


def fetch_descriptions(key: str, video_ids: list) -> dict:
    out = {}
    for start in range(0, len(video_ids), 50):
        payload, error = safe_api_get(
            "videos", key, part="snippet,contentDetails",
            id=",".join(video_ids[start:start + 50]))
        if error or not payload:
            logger.warning("videos.list failed: %s", error)
            continue
        for item in payload.get("items", []):
            out[item["id"]] = item
    return out


def caption_tracks(key: str, video_id: str) -> dict:
    """
    Whether caption tracks exist. Their CONTENT is not obtainable: downloading
    requires OAuth as the video owner, and third-party transcript endpoints are
    scraping. This records existence only.
    """
    payload, error = safe_api_get("captions", key, part="snippet",
                                  videoId=video_id)
    if error or not payload:
        return {"tracks": None, "error": error}
    items = payload.get("items", [])
    return {"tracks": len(items),
            "kinds": sorted({(i.get("snippet") or {}).get("trackKind", "?")
                             for i in items}),
            "error": None}


# ---------------------------------------------------------------------------
# B. Matching a chapter label to a play
# ---------------------------------------------------------------------------

def tokens(text) -> set:
    if not isinstance(text, str):
        return set()
    words = re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()
    return {w for w in words if w not in STOPWORDS and len(w) > 1}


def surname(name) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    return name.strip().split()[-1].lower()


def label_agreement(event_description, label) -> float:
    """Share of the event's words present in the chapter label."""
    ours = tokens(event_description)
    theirs = tokens(label)
    if not ours:
        return 0.0
    return len(ours & theirs) / len(ours)


def match_chapter(label: str, events: pd.DataFrame) -> dict:
    """
    Best play-by-play event for one chapter label, or nothing.

    Requires the event's player surname to appear in the label AND enough word
    overlap, then requires the winner to be unique. A chapter that matches two
    events equally well is not matched: picking one would be a guess about
    which play the viewer is about to be shown.
    """
    label_tokens = tokens(label)
    if not label_tokens:
        return {"matched": False, "reason": "empty label"}

    scored = []
    for event in events.itertuples():
        person = surname(event.player_name)
        if not person or person not in label_tokens:
            continue
        agreement = label_agreement(event.description, label)
        if agreement >= LABEL_AGREEMENT:
            scored.append((agreement, event))

    if not scored:
        return {"matched": False,
                "reason": "no event whose player and words both appear"}

    scored.sort(key=lambda pair: -pair[0])
    best_score = scored[0][0]
    tied = [pair for pair in scored if abs(pair[0] - best_score) < 1e-9]
    if len(tied) > 1:
        return {"matched": False,
                "reason": f"{len(tied)} events tie at {best_score:.2f}"}

    event = scored[0][1]
    return {"matched": True, "agreement": round(best_score, 3),
            "event_index": int(event.event_index),
            "period": int(event.period), "clock": event.clock_raw,
            "celtics_score": int(event.celtics_score),
            "opponent_score": int(event.opponent_score),
            "player": event.player_name,
            "event_description": event.description}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(scan: pd.DataFrame, chapters: pd.DataFrame, game_id,
                 captions: list) -> str:
    with_chapters = scan.loc[scan["has_chapters"]] if len(scan) else scan

    lines = [
        "=" * 78,
        "PHASE 13a - CAN A HIGHLIGHT VIDEO BE SEEKED TO A PLAY",
        "=" * 78,
        "",
        "  The Phase 12 mapping links one game to one ~9 minute recap. That is",
        "  NOT play-synchronised video. Seeking to a play needs a timestamp",
        "  inside the video, and the mapping has none.",
        "",
        "  Within the project's rules there is exactly one legitimate source",
        "  of such timestamps: chapter markers in the video description.",
        "  Caption CONTENT needs OAuth as the video's owner, and anything",
        "  derived from the video itself requires obtaining the video. Neither",
        "  is attempted here.",
        "",
        "=" * 78,
        "A. DO OFFICIAL HIGHLIGHT VIDEOS CARRY CHAPTERS",
        "=" * 78,
        "",
        f"  mapped videos examined      {len(scan)}",
        f"  with any timestamped line   "
        f"{int(scan['timestamp_lines'].gt(0).sum()) if len(scan) else 0}",
        f"  with real chapters (3+, from 0:00)  {len(with_chapters)}",
        "",
    ]

    if not len(with_chapters):
        lines += [
            "  NONE. No mapped video carries chapter markers.",
            "",
            "  That closes it. With no timestamps in the metadata, a play",
            "  offset could only be inferred from position or duration, which",
            "  is guessing, and guessing is exactly what must not happen here.",
            "",
            "  Play-synchronised video is NOT achievable from this source",
            "  under these rules. The synchronised layer stays the",
            "  play-by-play figure animation drawn from our own coordinates,",
            "  which covers every play of all 636 games.",
            "",
        ]
    else:
        lines += [
            f"  {len(with_chapters)} video(s) DO carry chapters. Section B",
            "  tests whether those chapters can be tied to specific plays.",
            "",
        ]
        for row in with_chapters.head(10).itertuples():
            lines.append(f"    {row.game_id}  {row.chapter_count} chapters  "
                         f"{str(row.title)[:44]}")
        lines.append("")

    lines += ["=" * 78, "CAPTION TRACKS", "=" * 78, "",
              "  Existence only. Content requires OAuth as the video owner,",
              "  and third-party transcript endpoints are scraping. Recorded",
              "  because it bounds what a future, properly licensed approach",
              "  could use, not because it is usable now.", ""]
    if not captions:
        lines.append("  Not checked.")
    for entry in captions:
        if entry.get("error"):
            lines.append(f"    {entry['video_id']}: {entry['error'][:60]}")
        else:
            lines.append(f"    {entry['video_id']}: {entry['tracks']} track(s) "
                         f"{entry.get('kinds')}")

    lines += ["", "=" * 78,
              f"B. CHAPTER TO PLAY MATCHING{f' - GAME {game_id}' if game_id else ''}",
              "=" * 78, ""]
    if not len(chapters):
        lines += ["  Not run: the probed game's video has no chapters.", ""]
    else:
        matched = chapters.loc[chapters["matched"]]
        lines += [
            f"  chapters            {len(chapters)}",
            f"  matched to a play   {len(matched)} "
            f"({len(matched) / len(chapters):.0%})",
            "",
            "  A chapter is matched only when a single play-by-play event has",
            "  both its player's surname in the label AND enough word overlap.",
            "  A tie is NOT broken; it is left unmatched, because choosing",
            "  would be a guess about which play the viewer is shown.",
            "",
        ]
        for row in chapters.itertuples():
            mark = "OK " if row.matched else "-- "
            lines.append(f"  {mark}{row.timestamp:>6}s  "
                         f"{str(row.label)[:44]}")
            if row.matched:
                lines.append(f"        -> P{row.period} {row.clock} "
                             f"{row.celtics_score}-{row.opponent_score}  "
                             f"{str(row.event_description)[:50]}")
            else:
                lines.append(f"        -> {row.reason}")

        lines += ["", "=" * 78, "MANUAL SPOT CHECK, REQUIRED", "=" * 78, "",
                  "  This probe can prove a chapter LABEL describes a play. It",
                  "  CANNOT prove the timestamp is right, because verifying",
                  "  that means watching the video at that offset.",
                  "",
                  "  No timing accuracy figure is reported, because none can",
                  "  be measured here. Open these and confirm by eye:", ""]
        for row in matched.head(8).itertuples():
            lines.append(f"    {str(row.event_description)[:58]}")
            lines.append(f"      https://www.youtube.com/watch?v={row.video_id}"
                         f"&t={int(row.timestamp)}s")
        lines.append("")

    lines += [
        "=" * 78,
        "WHAT HAPPENS EITHER WAY",
        "=" * 78,
        "",
        "  Verified play  seek the embedded official video to that offset.",
        "  Unverified     keep the synchronised figure animation and show",
        "                 'No verified video for this play.'",
        "",
        "  The figure animation is not replaced by video under any outcome.",
        "  It is the only layer that covers every play of every game.",
        "",
        "  Nothing was changed. Nothing in data/serving was touched.",
        "=" * 78,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(game_id=None):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    config.ensure_dirs()

    key = load_api_key()
    map_path = config.INTERIM_DIR / "highlights_map.json"
    if not map_path.exists():
        raise SystemExit(
            f"{map_path} not found. Run scripts/34_precompute_highlights.py "
            "first.")
    mapping = json.loads(map_path.read_text(encoding="utf-8"))
    if not mapping:
        raise SystemExit("The highlights mapping is empty.")

    logger.info("scanning %d mapped video(s) for chapter markers",
                len(mapping))
    details = fetch_descriptions(key, [v["video_id"] for v in mapping.values()])

    scan_rows = []
    for gid, entry in mapping.items():
        item = details.get(entry["video_id"], {})
        description = (item.get("snippet") or {}).get("description", "")
        stamps = parse_timestamps(description)
        scan_rows.append({
            "game_id": gid, "video_id": entry["video_id"],
            "title": entry["title"], "timestamp_lines": len(stamps),
            "chapter_count": len(stamps) if has_real_chapters(stamps) else 0,
            "has_chapters": has_real_chapters(stamps),
            "description_chars": len(description or ""),
        })
    scan = pd.DataFrame(scan_rows)
    logger.info("  %d of %d video(s) carry real chapters",
                int(scan["has_chapters"].sum()), len(scan))

    captions = []
    for entry in list(mapping.values())[:CAPTION_CHECKS]:
        result = caption_tracks(key, entry["video_id"])
        captions.append({"video_id": entry["video_id"], **result})

    # B: one game, as instructed. Prefer one that actually has chapters.
    chapters = pd.DataFrame()
    with_chapters = scan.loc[scan["has_chapters"]]
    if game_id is None:
        game_id = (with_chapters["game_id"].iloc[0] if len(with_chapters)
                   else None)

    if game_id and game_id in mapping:
        events = pd.read_parquet(config.EVENTS_PARQUET)
        events["game_id"] = events["game_id"].astype(str).str.zfill(10)
        game_events = events.loc[events["game_id"].eq(str(game_id).zfill(10))]
        item = details.get(mapping[game_id]["video_id"], {})
        stamps = parse_timestamps(
            (item.get("snippet") or {}).get("description", ""))
        rows = []
        for seconds, label in stamps:
            result = match_chapter(label, game_events)
            rows.append({"game_id": game_id,
                         "video_id": mapping[game_id]["video_id"],
                         "timestamp": seconds, "label": label,
                         "reason": result.get("reason", ""), **result})
        chapters = pd.DataFrame(rows)

    report = build_report(scan, chapters, game_id, captions)
    print(report)
    (config.REPORTS_DIR / "video_sync_probe.txt").write_text(
        report + "\n", encoding="utf-8")
    scan.to_csv(config.INTERIM_DIR / "video_sync_probe.csv", index=False)
    if len(chapters):
        chapters.to_csv(config.INTERIM_DIR / "video_sync_chapters.csv",
                        index=False)

    print(f"\nSaved to: {config.REPORTS_DIR / 'video_sync_probe.txt'}")
    print("\nREAD ONLY. No video downloaded, scraped or cached. Nothing in")
    print("data/serving was touched and the dashboard is unchanged.")
    return scan, chapters


if __name__ == "__main__":
    main()
