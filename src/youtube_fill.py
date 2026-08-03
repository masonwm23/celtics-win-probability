"""
Phase 12f: fill the gap the uploads playlist cannot reach, resumably.

WHY THIS IS NEEDED
------------------
Phase 12e enumerated both official uploads playlists at one quota unit per
fifty videos. It worked, and inside the window it could reach it was perfect:

    @NBA        18,955 uploads, back only to 2024-03-08 (a hard depth cap)
    @celtics     3,176 uploads, back to 2017-04-01, FULLY enumerated
    matched      20 of the 20 Boston games in that window

Two conclusions from that, one of which closes a question permanently:

  - The Celtics channel's entire history was enumerated and contains no
    full-game reels at all. That is tested, not inferred. Every match will
    come from @NBA.

  - Everything before 2024-03-08 is unreachable by enumeration. Its absence
    from the mapping is a limit of the method, not a finding.

So the remaining ~616 games need search.list, which Phase 12d showed works at
roughly 89% from 2018-19 onward. Search is the expensive call, so this is
built to be run repeatedly across days rather than once.

RESUMABLE BY DESIGN
-------------------
Every attempted game is recorded. A re-run skips them and continues where the
last one stopped, so the job is "run it once a day until it says finished"
rather than one long session that loses everything on a quota error.

The mapping is rewritten after every single game, so a crash costs one game.

ORDERING
--------
Seasons Phase 12d found productive go first: 2018-19 through 2023-24. The two
seasons it found empty, 2016-17 and 2017-18, go last, so a limited daily quota
is spent where it is most likely to return something. They are still attempted
rather than assumed empty, because 12d only tested three games each.

VERIFICATION IS UNCHANGED
-------------------------
The same four axes as Phase 12e: teams parsed from the title, date in the
title or a tight upload window, the title reading as a game reel, and official
plus public plus embeddable. Plus uniqueness within the game's own candidates.
Anything uncertain goes to review and is never written to the mapping.

Unofficial channels cannot enter the mapping: candidates are filtered to the
two official channel ids before assessment.

STILL METADATA ONLY. No download, no scraping, no re-hosting.
"""

import json
import logging
import time
from datetime import timedelta

import pandas as pd

from src import config
from src.youtube_probe import (
    PUBLISH_WINDOW_AFTER, PUBLISH_WINDOW_BEFORE, load_api_key,
    resolve_channels, safe_api_get,
)
from src.youtube_targeted import hydrate, title_variants
from src.youtube_precompute import assess, load_games, teams_in_title
from src.youtube_coverage import looks_like_quota_exhaustion

logger = logging.getLogger(__name__)

# Phase 12d measured these. Productive first, empty-looking last, but nothing
# is skipped on the strength of a three-game sample.
SEASON_PRIORITY = ["2023-24", "2022-23", "2021-22", "2020-21", "2019-20",
                   "2018-19", "2017-18", "2016-17"]

VARIANTS = 2
RESULTS_PER_VARIANT = 8
DELAY = 0.25

MAP_PATH = "highlights_map.json"
PROGRESS_PATH = "highlights_progress.csv"
REVIEW_PATH = "highlights_review.csv"


def load_json(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            logger.warning("%s is not valid JSON, starting fresh", path)
    return default


def load_progress(path) -> set:
    if not path.exists():
        return set()
    try:
        return set(pd.read_csv(path)["game_id"].astype(str).str.zfill(10))
    except Exception:                              # noqa: BLE001
        return set()


def build_queue(games: list, mapped: set, attempted: set) -> list:
    """Unmatched, unattempted games, productive seasons first."""
    order = {season: i for i, season in enumerate(SEASON_PRIORITY)}
    pending = [g for g in games
               if g["game_id"] not in mapped and g["game_id"] not in attempted]
    return sorted(pending,
                  key=lambda g: (order.get(g["season"], 99), g["game_date"]))


def search_variant(key: str, query: str, after, before):
    return safe_api_get(
        "search", key, part="snippet", q=query, type="video",
        maxResults=RESULTS_PER_VARIANT, order="relevance",
        publishedAfter=after.strftime("%Y-%m-%dT%H:%M:%SZ"),
        publishedBefore=before.strftime("%Y-%m-%dT%H:%M:%SZ"))


def attempt_game(key: str, game: dict, official_ids: set) -> tuple:
    """
    (result, quota_dead). `result` carries a verdict of confirmed, review or
    unmatched, and the row that produced it.
    """
    after = game["game_date"] - PUBLISH_WINDOW_BEFORE
    before = game["game_date"] + PUBLISH_WINDOW_AFTER

    seen, reviews = {}, []
    for query, label in title_variants(game)[:VARIANTS]:
        payload, error = search_variant(key, query, after, before)
        if error:
            if looks_like_quota_exhaustion(error):
                return None, True
            logger.warning("    %s", error)
            continue

        ids = [i["id"]["videoId"] for i in (payload.get("items") or [])
               if i.get("id", {}).get("videoId")]
        fresh = [i for i in ids if i not in seen]
        for video_id in fresh:
            seen[video_id] = label
        if not fresh:
            continue

        details = hydrate(key, fresh)
        confirmed = []
        for video_id in fresh:
            item = details.get(video_id)
            if not item:
                continue
            snippet = item.get("snippet") or {}
            # Unofficial channels are removed before assessment, so they can
            # never reach the mapping by any path.
            if snippet.get("channelId") not in official_ids:
                continue
            candidate = {
                "handle": snippet.get("channelTitle"),
                "channel_id": snippet.get("channelId"),
                "video_id": video_id,
                "title": snippet.get("title"),
                "published_at": snippet.get("publishedAt"),
                "embeddable": bool((item.get("status") or {}).get("embeddable")),
                "privacy": (item.get("status") or {}).get("privacyStatus"),
                "duration": (item.get("contentDetails") or {}).get("duration"),
            }
            verdict = assess(candidate, game)
            row = {**candidate, **verdict, "found_by": label,
                   "game_id": game["game_id"], "season": game["season"],
                   "game_date": game["game_date"].date().isoformat(),
                   "matchup": game["matchup"]}
            if verdict["verdict"] == "confirmed":
                confirmed.append(row)
            elif verdict["verdict"] == "review":
                reviews.append(row)

        if len(confirmed) == 1:
            return {"verdict": "confirmed", "row": confirmed[0],
                    "reviews": reviews}, False
        if len(confirmed) > 1:
            # Two reels both confirm. Picking one would be a guess on screen.
            for row in confirmed:
                row["verdict"] = "review"
                row["problems"] = "more than one confirmed candidate"
            return {"verdict": "review", "row": None,
                    "reviews": reviews + confirmed}, False
        time.sleep(DELAY)

    return {"verdict": "review" if reviews else "unmatched", "row": None,
            "reviews": reviews}, False


def run(key: str, games: list, official_ids: set, paths: dict,
        max_games=None) -> dict:
    mapping = load_json(paths["map"], {})
    attempted = load_progress(paths["progress"])
    queue = build_queue(games, set(mapping), attempted)
    if max_games:
        queue = queue[:max_games]

    logger.info("%d already mapped, %d already attempted, %d queued",
                len(mapping), len(attempted), len(queue))

    progress_rows, review_rows = [], []
    stopped = False

    for i, game in enumerate(queue, start=1):
        result, quota_dead = attempt_game(key, game, official_ids)
        if quota_dead:
            logger.error("QUOTA EXHAUSTED after %d game(s) this run. Stopping "
                         "cleanly. Re-run tomorrow to continue.", i - 1)
            stopped = True
            break

        review_rows.extend(result["reviews"])
        progress_rows.append({
            "game_id": game["game_id"], "season": game["season"],
            "game_date": game["game_date"].date().isoformat(),
            "matchup": game["matchup"], "verdict": result["verdict"]})

        if result["verdict"] == "confirmed":
            row = result["row"]
            mapping[game["game_id"]] = {
                "video_id": row["video_id"], "title": row["title"],
                "channel": row["handle"],
                "published_at": row["published_at"],
                "duration_seconds": int(row["duration_seconds"]),
                "label": "Game Highlights",
                "note": ("Game-level recap published by the official channel. "
                         "Not synchronised with the probability cursor."),
            }

        # Written every game, so a crash or a kill costs one game.
        paths["map"].write_text(json.dumps(mapping, indent=2), encoding="utf-8")
        append_csv(paths["progress"], progress_rows[-1:])
        if result["reviews"]:
            append_csv(paths["review"], result["reviews"])

        if i % 10 == 0 or i == len(queue):
            got = sum(1 for r in progress_rows if r["verdict"] == "confirmed")
            logger.info("  %d/%d this run, %d confirmed, mapping now %d",
                        i, len(queue), got, len(mapping))
        time.sleep(DELAY)

    return {"mapping": mapping, "progress": pd.DataFrame(progress_rows),
            "reviews": pd.DataFrame(review_rows), "stopped": stopped,
            "queued": len(queue)}


def append_csv(path, rows):
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def build_report(result: dict, games: list) -> str:
    mapping = result["mapping"]
    progress = result["progress"]
    remaining = len(games) - len(mapping)

    lines = [
        "=" * 78,
        "PHASE 12f - GAP FILL, RESUMABLE",
        "=" * 78,
        "",
        "  The uploads playlist reached only 2024-03-08 on @NBA, a hard depth",
        "  cap. Inside that window it matched 20 of 20 games, so the method is",
        "  sound and simply cannot see further back. This fills the rest with",
        "  search, which is the expensive call, across as many runs as it",
        "  takes.",
        "",
        "  The Celtics channel was enumerated in full, 2017-04 to now, and",
        "  contains no full-game reels. That question is closed: every match",
        "  comes from @NBA.",
        "",
        f"  attempted this run   {len(progress)}",
        f"  confirmed this run   "
        f"{int(progress['verdict'].eq('confirmed').sum()) if len(progress) else 0}",
        f"  mapping total        {len(mapping)} of {len(games)} games "
        f"({len(mapping) / len(games):.1%})",
        f"  still unmapped       {remaining}",
        "",
    ]

    if result["stopped"]:
        lines += [
            "  !! STOPPED ON QUOTA. Nothing was lost: the mapping and the",
            "  progress file are written after every game. Run the same",
            "  script again tomorrow and it will continue from here.",
            "",
        ]
    elif remaining:
        lines += [
            f"  Queue not empty. {remaining} game(s) remain. Run again to",
            "  continue; already-attempted games are skipped.",
            "",
        ]
    else:
        lines += ["  Queue empty. Every game has been attempted.", ""]

    if len(progress):
        lines += ["=" * 78, "THIS RUN, BY SEASON", "=" * 78, "",
                  f"  {'season':<10}{'tried':>7}{'confirmed':>11}"
                  f"{'review':>8}{'unmatched':>11}{'rate':>8}"]
        for season, group in progress.groupby("season"):
            got = int(group["verdict"].eq("confirmed").sum())
            lines.append(
                f"  {season:<10}{len(group):>7}{got:>11}"
                f"{int(group['verdict'].eq('review').sum()):>8}"
                f"{int(group['verdict'].eq('unmatched').sum()):>11}"
                f"{got / len(group):>7.0%}")

    by_season = {}
    for game in games:
        by_season.setdefault(game["season"], []).append(game["game_id"])
    lines += ["", "=" * 78, "MAPPING TOTAL, BY SEASON", "=" * 78, "",
              f"  {'season':<10}{'games':>7}{'mapped':>8}{'rate':>8}"]
    for season in sorted(by_season):
        ids = by_season[season]
        got = sum(1 for i in ids if i in mapping)
        lines.append(f"  {season:<10}{len(ids):>7}{got:>8}"
                     f"{got / len(ids):>7.0%}")

    reviews = result["reviews"]
    lines += ["", "=" * 78, "SENT TO REVIEW THIS RUN", "=" * 78, "",
              "  Official channel, right two teams, something else wrong.",
              "  NONE are in the mapping and NONE would be displayed.", ""]
    if not len(reviews):
        lines.append("  None this run.")
    else:
        for row in reviews.head(25).itertuples():
            lines += [f"  {row.season}  {row.matchup}  {row.game_date}",
                      f"    {str(row.title)[:70]}",
                      f"    WHY: {row.problems}",
                      f"    https://www.youtube.com/watch?v={row.video_id}",
                      ""]
        if len(reviews) > 25:
            lines.append(f"  ... and {len(reviews) - 25} more in "
                         f"{REVIEW_PATH}")

    lines += [
        "",
        "=" * 78,
        "  Nothing in data/serving was touched. The dashboard is unchanged",
        "  and does not read the mapping yet.",
        "=" * 78,
    ]
    return "\n".join(lines)


def main(max_games=None):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    config.ensure_dirs()

    key = load_api_key()
    index_path = config.RAW_DIR / "game_index.csv"
    if not index_path.exists():
        raise SystemExit(f"{index_path} not found.")

    games = load_games(pd.read_csv(index_path))
    channels = resolve_channels(key)
    if not channels:
        raise SystemExit("Could not resolve either official channel.")
    official_ids = {c["channel_id"] for c in channels.values()}

    paths = {
        "map": config.INTERIM_DIR / MAP_PATH,
        "progress": config.INTERIM_DIR / PROGRESS_PATH,
        "review": config.INTERIM_DIR / REVIEW_PATH,
    }

    result = run(key, games, official_ids, paths, max_games=max_games)
    report = build_report(result, games)
    print(report)
    (config.REPORTS_DIR / "youtube_fill.txt").write_text(report + "\n",
                                                         encoding="utf-8")
    print(f"\nMapping : {paths['map']}  ({len(result['mapping'])} games)")
    print(f"Progress: {paths['progress']}")
    print(f"Report  : {config.REPORTS_DIR / 'youtube_fill.txt'}")
    print("\nREAD ONLY. No video downloaded, scraped or cached. Nothing in")
    print("data/serving was touched and the dashboard is unchanged.")
    return result


if __name__ == "__main__":
    main()
