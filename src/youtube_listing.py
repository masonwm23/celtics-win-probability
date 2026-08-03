"""
Phase 12b: stop guessing search terms. List what the official channels
actually posted.

WHY
---
Phase 12 found a usable reel for 1 of 3 games. The two failures were rejected
with "title does not name both teams", and every rejected candidate was a
"Top 10 Plays of the Night" compilation rather than a game reel.

That is a search problem, not necessarily an availability problem. The query
was `"Celtics {nickname} highlights"`, invented rather than observed, and if
the NBA's title convention in 2016-17 differed from 2020-21 then the right
video could exist and simply never appear in the top ten by relevance.

So this stops querying and instead LISTS every upload from the two official
channels inside each game's date window, ordered by date, and marks which ones
the Phase 12 matcher would accept. That distinguishes three different answers
the previous run could not:

  - the reel exists and the query missed it        -> fix the query
  - the reel exists under a title naming one team  -> fix the title rule
  - nothing was posted for that game               -> genuinely unavailable

THE REGION QUESTION, WHICH MAY MATTER MORE
------------------------------------------
The one video that matched carried `regionRestriction.allowed` with 24
entries. A whitelist of 24 countries out of ~250 is unusual, and if the United
States is not in it the panel is useless where this project is being written,
no matter how many reels exist.

Phase 12 counted those regions but did not print them. This prints the codes
and states plainly whether US is present.

STILL METADATA ONLY. No download, no scraping, no re-hosting. Same API, same
rules.

READ ONLY. Writes reports/youtube_listing.txt and
data/interim/youtube_listing.csv.
"""

import logging
from datetime import timedelta

import pandas as pd

from src import config
from src.youtube_probe import (
    PUBLISH_WINDOW_AFTER, PUBLISH_WINDOW_BEFORE, TEAM_NAMES, classify,
    game_fields, load_api_key, pick_games, resolve_channels, safe_api_get,
    title_names_both_teams,
)

logger = logging.getLogger(__name__)

# Everything in the window, not a top-ten by relevance. This is the whole point.
MAX_RESULTS = 50

# Words that suggest a full game reel rather than a highlight compilation.
GAME_REEL_HINTS = ("full game highlights", "game highlights", "full game",
                   "game recap", "highlights")


def list_uploads(key: str, channel_id: str, after, before) -> tuple:
    """Every video from one channel in one window, newest first."""
    return safe_api_get(
        "search", key, part="snippet", channelId=channel_id, type="video",
        maxResults=MAX_RESULTS, order="date",
        publishedAfter=after.strftime("%Y-%m-%dT%H:%M:%SZ"),
        publishedBefore=before.strftime("%Y-%m-%dT%H:%M:%SZ"))


def hydrate(key: str, video_ids: list) -> dict:
    """videos.list in batches of 50, because search gives no status."""
    out = {}
    for start in range(0, len(video_ids), 50):
        batch = video_ids[start:start + 50]
        payload, error = safe_api_get(
            "videos", key, part="snippet,status,contentDetails",
            id=",".join(batch))
        if error or not payload:
            logger.warning("videos.list failed: %s", error)
            continue
        for item in payload.get("items", []):
            out[item["id"]] = item
    return out


def region_detail(item: dict) -> dict:
    """
    The actual codes, not a count.

    `us_ok` is None when there is no restriction at all, which is the normal
    and best case.
    """
    restriction = (item.get("contentDetails") or {}).get("regionRestriction")
    if not restriction:
        return {"region_mode": "none", "region_codes": "", "us_ok": None}
    if restriction.get("allowed") is not None:
        allowed = restriction["allowed"]
        return {"region_mode": "allowlist",
                "region_codes": ",".join(sorted(allowed)),
                "us_ok": "US" in allowed}
    blocked = restriction.get("blocked") or []
    return {"region_mode": "blocklist",
            "region_codes": ",".join(sorted(blocked)),
            "us_ok": "US" not in blocked}


def looks_like_a_game_reel(title) -> bool:
    text = (title or "").lower()
    return any(hint in text for hint in GAME_REEL_HINTS)


def run(key: str, games: list, channels: dict) -> pd.DataFrame:
    official_ids = {c["channel_id"] for c in channels.values()}
    records = []

    for game in games:
        after = game["game_date"] - PUBLISH_WINDOW_BEFORE
        before = game["game_date"] + PUBLISH_WINDOW_AFTER
        logger.info("%s  %s  %s", game["season"], game["matchup"],
                    game["game_date"].date())

        ids, sources = [], {}
        for handle, channel in channels.items():
            payload, error = list_uploads(key, channel["channel_id"], after,
                                          before)
            if error:
                logger.warning("  %s: %s", handle, error)
                continue
            found = [i["id"]["videoId"] for i in (payload.get("items") or [])
                     if i.get("id", {}).get("videoId")]
            for video_id in found:
                sources[video_id] = handle
            ids.extend(found)
            logger.info("    %s: %d upload(s) in window", handle, len(found))

        details = hydrate(key, ids)
        for video_id in ids:
            item = details.get(video_id)
            if not item:
                continue
            verdict = classify(item, game, official_ids)
            title = (item.get("snippet") or {}).get("title")
            records.append({
                **game_fields(game),
                "handle": sources.get(video_id, ""),
                "video_id": video_id,
                "title": title,
                "published_at": (item.get("snippet") or {}).get("publishedAt"),
                "duration": (item.get("contentDetails") or {}).get("duration"),
                "embeddable": bool((item.get("status") or {}).get("embeddable")),
                "privacy": (item.get("status") or {}).get("privacyStatus"),
                "names_both_teams": title_names_both_teams(
                    title or "", game["opponent_tricode"]),
                "looks_like_reel": looks_like_a_game_reel(title),
                "phase12_verdict": verdict["verdict"],
                "phase12_reasons": verdict["reasons"],
                "watch_url": f"https://www.youtube.com/watch?v={video_id}",
                **region_detail(item),
            })

    return pd.DataFrame(records)


def build_report(frame: pd.DataFrame, games: list, channels: dict) -> str:
    lines = [
        "=" * 78,
        "PHASE 12b - WHAT THE OFFICIAL CHANNELS ACTUALLY POSTED",
        "=" * 78,
        "",
        "  Phase 12 searched with an invented query and matched 1 of 3 games.",
        "  This lists EVERY upload from the two official channels inside each",
        "  game's date window instead, so the question 'does a reel exist' is",
        "  answered by observation rather than by whether my search terms",
        "  happened to rank it in the top ten.",
        "",
        "  Metadata only. No download, no scraping, no re-hosting.",
        "",
        f"  uploads listed: {len(frame)}",
        "",
    ]

    for game in games:
        rows = (frame.loc[frame["game_id"].eq(game["game_id"])]
                if len(frame) else frame)
        lines += ["=" * 78,
                  f"{game['season']}   {game['matchup']}   "
                  f"{game['game_date'].date()}",
                  "=" * 78, ""]
        if not len(rows):
            lines += ["  Nothing was posted by either official channel in the",
                      "  window. That is a genuine absence, not a search miss.",
                      ""]
            continue

        reels = rows.loc[rows["looks_like_reel"] & rows["names_both_teams"]]
        lines.append(f"  {len(rows)} upload(s) in window. "
                     f"{len(reels)} look like a reel naming both teams.")
        lines.append("")
        lines.append(f"  {'ok':<4}{'both':<6}{'reel':<6}{'embed':<7}"
                     f"{'duration':<11}title")
        for row in rows.itertuples():
            lines.append(
                f"  {'YES' if row.phase12_verdict == 'matched' else '-':<4}"
                f"{'y' if row.names_both_teams else '-':<6}"
                f"{'y' if row.looks_like_reel else '-':<6}"
                f"{'y' if row.embeddable else 'NO':<7}"
                f"{str(row.duration or '-'):<11}{str(row.title)[:44]}")
        lines.append("")

        matched = rows.loc[rows["phase12_verdict"].eq("matched")]
        if len(matched):
            lines.append("  MATCHED BY THE PHASE 12 RULE:")
            for row in matched.itertuples():
                lines.append(f"    {row.title}")
                lines.append(f"    {row.watch_url}")
            lines.append("")
        elif len(reels):
            lines += [
                "  A reel naming both teams EXISTS but the Phase 12 rule",
                "  rejected it. The rule is wrong for this game, not the",
                "  availability. Reasons given:",
            ]
            for row in reels.head(3).itertuples():
                lines.append(f"    {str(row.title)[:60]}")
                lines.append(f"      {row.phase12_reasons}")
            lines.append("")
        else:
            lines += [
                "  No upload in the window both looks like a reel and names",
                "  both teams. For this game the answer is availability, not",
                "  matching.",
                "",
            ]

    lines += ["=" * 78, "REGION RESTRICTIONS", "=" * 78, "",
              "  The question that decides whether any of this works from the",
              "  United States, where this project is being written.",
              ""]
    restricted = (frame.loc[frame["region_mode"].ne("none")]
                  if len(frame) else frame)
    if not len(frame):
        lines.append("  Nothing listed.")
    elif not len(restricted):
        lines.append("  No upload in the window carries any region "
                     "restriction.")
    else:
        for row in restricted.itertuples():
            lines += [
                f"  {str(row.title)[:66]}",
                f"    mode {row.region_mode}, {len(str(row.region_codes).split(','))} code(s)",
                f"    US playable: {row.us_ok}",
                f"    {str(row.region_codes)[:200]}",
                "",
            ]
        us_blocked = restricted.loc[restricted["us_ok"].eq(False)]
        if len(us_blocked):
            lines += [
                f"  !! {len(us_blocked)} of {len(restricted)} restricted",
                "  upload(s) are NOT playable in the United States. A panel",
                "  built on those would be blank for you and for anyone",
                "  reviewing this project here.",
                "",
            ]

    lines += [
        "=" * 78,
        "WHAT THIS DOES NOT TELL YOU",
        "=" * 78,
        "",
        "  Three games. Whatever pattern shows here has to be confirmed at",
        "  scale before a panel is worth building, and that is a separate run.",
        "",
        "  Region data from the API describes the video, not your network. A",
        "  video with no restriction can still be unavailable for other",
        "  reasons, and the panel must degrade to nothing when it is.",
        "",
        "  Nothing was changed. The dashboard does not know this exists.",
        "=" * 78,
    ]
    return "\n".join(lines)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    config.ensure_dirs()

    key = load_api_key()
    index_path = config.RAW_DIR / "game_index.csv"
    if not index_path.exists():
        raise SystemExit(f"{index_path} not found.")

    games = pick_games(pd.read_csv(index_path))
    logger.info("resolving official channels")
    channels = resolve_channels(key)
    if not channels:
        raise SystemExit("Could not resolve either official channel.")

    frame = run(key, games, channels)
    report = build_report(frame, games, channels)
    print(report)

    out = config.REPORTS_DIR / "youtube_listing.txt"
    out.write_text(report + "\n", encoding="utf-8")
    if len(frame):
        frame.to_csv(config.INTERIM_DIR / "youtube_listing.csv", index=False)
    print(f"\nSaved to: {out}")
    print("\nREAD ONLY. No video was downloaded, scraped or cached.")
    return frame


if __name__ == "__main__":
    main()
