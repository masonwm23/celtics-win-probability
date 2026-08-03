"""
Phase 12c: search for the title the NBA actually uses, not one I invented.

TWO CORRECTIONS TO MY OWN WORK
------------------------------
1. Phase 12 queried `"Celtics {nickname} highlights"`. Invented. It matched 1
   of 3 games.

2. Phase 12b listed channel uploads with `search.list` and `order=date`,
   believing that enumerated the channel. It does not. `search.list` is a
   SEARCH INDEX and is not guaranteed to return every upload. The output
   proved it: one upload from @NBA in a five-day window of January 2024, for a
   channel that posts dozens of videos a day. So 12b's "no reel for this game"
   was never a finding, only a failure to surface one.

WHAT 12b DID ESTABLISH
----------------------
The convention, observed rather than assumed. In the March 2021 window roughly
thirty uploads carried exactly this shape, one per game played that night:

    CELTICS at NETS | FULL GAME HIGHLIGHTS | March 11, 2021
    76ERS at BULLS | FULL GAME HIGHLIGHTS | March 11, 2021

AWAY at HOME, the phrase FULL GAME HIGHLIGHTS, then the date. Every field of
that is derivable from `game_index.csv`, including which side is home, so the
expected title can be CONSTRUCTED for any of the 636 games instead of guessed
at.

WHAT THIS DOES
--------------
For each game, builds several candidate titles covering the conventions seen
across eight seasons, searches YouTube for each, and filters the results to the
official channels. Searching without a channel filter lets YouTube's relevance
do the work; the official-channel test is then applied to the results, so
precision is not lost.

The five acceptance rules from Phase 12 are unchanged: official channel,
embeddable, public, published in the game-date window, title names both teams.

It also records WHICH title variant found the match, because a run across all
636 games needs to know which convention applies to which era.

STILL METADATA ONLY. No download, no scraping, no re-hosting.

READ ONLY. Writes reports/youtube_targeted.txt and
data/interim/youtube_targeted.csv.
"""

import logging

import pandas as pd

from src import config
from src.youtube_probe import (
    PUBLISH_WINDOW_AFTER, PUBLISH_WINDOW_BEFORE, TEAM_NAMES, classify,
    game_fields, load_api_key, pick_games, resolve_channels, safe_api_get,
)

logger = logging.getLogger(__name__)

RESULTS_PER_VARIANT = 8

MONTHS = ("January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December")


def home_and_away(game: dict) -> tuple:
    """
    (away_nickname, home_nickname).

    The NBA titles these AWAY at HOME, so getting this backwards would search
    for a title that does not exist. `IS_HOME` is Boston's perspective.
    """
    opponent = TEAM_NAMES.get(game["opponent_tricode"],
                              (game["opponent_tricode"], ""))[0]
    celtics = TEAM_NAMES["BOS"][0]
    if game["is_home"]:
        return opponent, celtics
    return celtics, opponent


def title_variants(game: dict) -> list:
    """
    Candidate titles, most specific first.

    Several conventions are covered because eight seasons is long enough for
    the format to have changed, and 12b only observed the 2021 one directly.
    """
    away, home = home_and_away(game)
    date = game["game_date"]
    long_date = f"{MONTHS[date.month - 1]} {date.day}, {date.year}"
    return [
        (f"{away} at {home} | FULL GAME HIGHLIGHTS | {long_date}",
         "2021 convention: AWAY at HOME | FULL GAME HIGHLIGHTS | date"),
        (f"{away} vs {home} Full Game Highlights {long_date}",
         "vs form with date"),
        (f"{away} vs {home} full game highlights",
         "vs form, no date"),
        (f"{home} vs {away} highlights {date.year}",
         "reversed order fallback"),
    ]


def search_titles(key: str, query: str, after, before) -> tuple:
    """
    No channel filter. Relevance across YouTube finds the reel; the official
    channel test is then applied to what comes back.
    """
    return safe_api_get(
        "search", key, part="snippet", q=query, type="video",
        maxResults=RESULTS_PER_VARIANT, order="relevance",
        publishedAfter=after.strftime("%Y-%m-%dT%H:%M:%SZ"),
        publishedBefore=before.strftime("%Y-%m-%dT%H:%M:%SZ"))


def hydrate(key: str, video_ids: list) -> dict:
    out = {}
    unique = list(dict.fromkeys(video_ids))
    for start in range(0, len(unique), 50):
        payload, error = safe_api_get(
            "videos", key, part="snippet,status,contentDetails",
            id=",".join(unique[start:start + 50]))
        if error or not payload:
            logger.warning("videos.list failed: %s", error)
            continue
        for item in payload.get("items", []):
            out[item["id"]] = item
    return out


def run(key: str, games: list, channels: dict) -> pd.DataFrame:
    official_ids = {c["channel_id"] for c in channels.values()}
    records = []

    for game in games:
        after = game["game_date"] - PUBLISH_WINDOW_BEFORE
        before = game["game_date"] + PUBLISH_WINDOW_AFTER
        away, home = home_and_away(game)
        logger.info("%s  %s  %s  (%s at %s)", game["season"], game["matchup"],
                    game["game_date"].date(), away, home)

        found_by = {}
        for query, label in title_variants(game):
            payload, error = search_titles(key, query, after, before)
            if error:
                logger.warning("    %s: %s", label, error)
                continue
            ids = [i["id"]["videoId"] for i in (payload.get("items") or [])
                   if i.get("id", {}).get("videoId")]
            for video_id in ids:
                found_by.setdefault(video_id, label)
            logger.info("    %-46s %d hit(s)", query[:46], len(ids))

        if not found_by:
            records.append({**game_fields(game), "video_id": None,
                            "found_by": "", "verdict": "no_candidates",
                            "reasons": "no search variant returned anything"})
            continue

        details = hydrate(key, list(found_by))
        for video_id, label in found_by.items():
            item = details.get(video_id)
            if not item:
                continue
            records.append({**game_fields(game), "found_by": label,
                            **classify(item, game, official_ids)})

    return pd.DataFrame(records)


def build_report(frame: pd.DataFrame, games: list) -> str:
    matched = (frame.loc[frame["verdict"].eq("matched")]
               if len(frame) else frame)
    games_with = matched["game_id"].nunique() if len(matched) else 0

    lines = [
        "=" * 78,
        "PHASE 12c - SEARCHING FOR THE TITLE THE NBA ACTUALLY USES",
        "=" * 78,
        "",
        "  Two corrections to earlier runs in this phase.",
        "",
        "  Phase 12 searched an invented query and matched 1 of 3.",
        "",
        "  Phase 12b listed channel uploads and treated that as complete. It",
        "  is not: search.list is a search index, not an enumeration, and it",
        "  returned ONE upload from @NBA across five days of January 2024 for",
        "  a channel that posts dozens a day. Its 'no reel for this game' was",
        "  never a finding.",
        "",
        "  What 12b did establish is the convention, from ~30 uploads in the",
        "  March 2021 window: AWAY at HOME | FULL GAME HIGHLIGHTS | date.",
        "  Every field of that comes from game_index.csv, so the title is",
        "  constructed here rather than guessed.",
        "",
        f"  games tested             {len(games)}",
        f"  games with a usable reel {games_with} of {len(games)}",
        f"  candidates examined      {len(frame)}",
        "",
    ]

    for game in games:
        rows = (frame.loc[frame["game_id"].eq(game["game_id"])]
                if len(frame) else frame)
        hits = (rows.loc[rows["verdict"].eq("matched")]
                if len(rows) else rows)
        away, home = home_and_away(game)
        lines += ["=" * 78,
                  f"{game['season']}   {game['matchup']}   "
                  f"{game['game_date'].date()}   ({away} at {home})",
                  "=" * 78, ""]
        lines.append("  titles searched:")
        for query, label in title_variants(game):
            lines.append(f"    {query}")
        lines.append("")

        if len(hits):
            for row in hits.itertuples():
                lines += [
                    f"  MATCHED  {row.title}",
                    f"    channel    {row.channel_title}",
                    f"    published  {row.published_at}",
                    f"    duration   {row.duration}",
                    f"    found by   {row.found_by}",
                    f"    {row.watch_url}",
                ]
                if row.region_restriction:
                    lines.append(f"    regions    {row.region_restriction}")
                lines.append("")
        else:
            lines.append("  NO USABLE REEL FROM ANY VARIANT")
            if len(rows):
                lines.append("  closest candidates:")
                for row in rows.head(5).itertuples():
                    lines.append(f"    {str(getattr(row, 'title', ''))[:60]}")
                    lines.append(f"      {getattr(row, 'channel_title', '')}"
                                 f" | {row.reasons}")
            lines.append("")

    if len(matched):
        lines += ["=" * 78, "WHICH TITLE CONVENTION WORKED", "=" * 78, "",
                  "  A run across all 636 games needs this per era.", ""]
        for label, group in matched.groupby("found_by"):
            seasons = ", ".join(sorted(group["season"].unique()))
            lines.append(f"  {label}")
            lines.append(f"    worked for: {seasons}")
        lines.append("")

    lines += [
        "=" * 78,
        "WHAT THIS DOES NOT TELL YOU",
        "=" * 78,
        "",
        "  Three games. A convention that works for three is a hypothesis",
        "  about 636, not a coverage figure.",
        "",
        "  A game with no reel found here may still have one under a title no",
        "  variant covers. Absence of a match is weaker evidence than",
        "  presence of one, and should be reported that way.",
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
    report = build_report(frame, games)
    print(report)

    out = config.REPORTS_DIR / "youtube_targeted.txt"
    out.write_text(report + "\n", encoding="utf-8")
    if len(frame):
        frame.to_csv(config.INTERIM_DIR / "youtube_targeted.csv", index=False)
    print(f"\nSaved to: {out}")
    print("\nREAD ONLY. No video was downloaded, scraped or cached.")
    return frame


if __name__ == "__main__":
    main()
