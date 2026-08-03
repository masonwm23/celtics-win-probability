"""
Phase 12d: which SEASONS have official, embeddable game reels?

THE QUESTION
------------
Phase 12c tested three games and found an official reel for one, the 2020-21
game. The two failures both had full game highlights on YouTube, but only on
re-upload channels (Ximo Pierto, FreeDawkins, Motion Station and similar),
which are exactly the unauthorised mirrors this project will not use.

So official coverage looks partial and era-dependent. This measures that
properly: three games per season across all eight, using the title convention
that 12b observed and 12c confirmed.

SAMPLING, AND A MISTAKE MADE TWICE ALREADY
------------------------------------------
Games are drawn at fixed fractions through each season by date: 0.25, 0.50,
0.75. Deterministic, and deliberately spread.

Twice in Phase 11 a conclusion was drawn from a sample that had quietly
clustered: once at the start of the earliest game of each season, once at the
first quarter. Both times the bias pointed the same way as the conclusion. The
spread here is explicit for that reason.

THREE OUTCOMES, NOT TWO
-----------------------
A game is one of:

  official_reel     an official channel published a matching, embeddable reel
  unofficial_only   a matching reel exists but ONLY on unofficial channels
  nothing_found     no variant surfaced a matching reel anywhere

The middle case matters. It says the highlights exist and the NBA simply did
not publish them to YouTube itself, which is a different fact from the game
having no coverage, and it is the case where the answer is "we will not use
that" rather than "there is nothing".

QUOTA, AND THE FAILURE THAT WOULD LOOK LIKE A FINDING
-----------------------------------------------------
The free tier is 10,000 units a day and search.list is the expensive call. At
two variants for each of 24 games this run costs roughly 4,800 units, which
fits in one day.

If the quota runs out mid-run, every remaining game would return nothing and
the later seasons would read as having no coverage. That is a failure dressed
as a result, which this project has now been caught by three times. So quota
exhaustion is detected explicitly, the run stops, untested games are recorded
as `not_tested` rather than as absences, and the report says so at the top.

Partial results are written to disk as the run proceeds, so a stop loses
nothing.

STILL METADATA ONLY. No download, no scraping, no re-hosting.

READ ONLY. Writes reports/youtube_coverage.txt and
data/interim/youtube_coverage.csv.
"""

import logging
import time

import pandas as pd

from src import config
from src.youtube_probe import (
    PUBLISH_WINDOW_AFTER, PUBLISH_WINDOW_BEFORE, classify, load_api_key,
    resolve_channels, safe_api_get, title_names_both_teams,
)
from src.youtube_targeted import (
    RESULTS_PER_VARIANT, hydrate, home_and_away, search_titles, title_variants,
)
from src.youtube_listing import looks_like_a_game_reel

logger = logging.getLogger(__name__)

GAMES_PER_SEASON = 3
SEASON_POSITIONS = (0.25, 0.50, 0.75)

# Only the two most specific variants, to keep the run inside one day's quota.
VARIANTS_PER_GAME = 2

# Rough public figure for search.list. Used only to warn before spending.
UNITS_PER_SEARCH = 100
DAILY_FREE_QUOTA = 10_000

DELAY = 0.2

QUOTA_MARKERS = ("quotaexceeded", "dailylimitexceeded", "http error 403")


def looks_like_quota_exhaustion(error) -> bool:
    """
    A spent quota returns 403. Treating that as "no video" would turn the tail
    of the run into a fabricated absence.
    """
    if not error:
        return False
    text = str(error).lower()
    return any(marker in text for marker in QUOTA_MARKERS)


def pick_games(index: pd.DataFrame) -> list:
    """Three per season at fixed fractions through the schedule."""
    games = []
    for season in sorted(index["SEASON"].unique()):
        group = (index.loc[index["SEASON"].eq(season)]
                 .sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True))
        n = len(group)
        for position in SEASON_POSITIONS:
            row = group.loc[min(int(round(position * (n - 1))), n - 1)]
            games.append({
                "season": season,
                "game_id": str(row["GAME_ID"]).zfill(10),
                "game_date": pd.Timestamp(row["GAME_DATE"]).tz_localize("UTC")
                .to_pydatetime(),
                "opponent_tricode": row["OPPONENT_ABBREV"],
                "matchup": row["MATCHUP"],
                "is_home": bool(row["IS_HOME"]),
                "position_in_season": position,
            })
    # Deduplicate in case a short season collapses two positions onto one game.
    seen, unique = set(), []
    for game in games:
        if game["game_id"] in seen:
            continue
        seen.add(game["game_id"])
        unique.append(game)
    return unique


def probe_game(key: str, game: dict, official_ids: set) -> dict:
    """One game. Returns a row, plus whether the quota died."""
    after = game["game_date"] - PUBLISH_WINDOW_BEFORE
    before = game["game_date"] + PUBLISH_WINDOW_AFTER

    candidates, quota_dead = {}, False
    for query, label in title_variants(game)[:VARIANTS_PER_GAME]:
        payload, error = search_titles(key, query, after, before)
        if error:
            if looks_like_quota_exhaustion(error):
                quota_dead = True
                break
            logger.warning("    %s", error)
            continue
        for entry in payload.get("items") or []:
            video_id = entry.get("id", {}).get("videoId")
            if video_id:
                candidates.setdefault(video_id, label)
        time.sleep(DELAY)

    if quota_dead:
        return {**base_row(game), "outcome": "not_tested",
                "note": "quota exhausted before this game"}, True

    if not candidates:
        return {**base_row(game), "outcome": "nothing_found",
                "note": "no variant returned anything"}, False

    details = hydrate(key, list(candidates))
    official, unofficial = [], []
    for video_id, label in candidates.items():
        item = details.get(video_id)
        if not item:
            continue
        title = (item.get("snippet") or {}).get("title", "")
        verdict = classify(item, game, official_ids)
        if verdict["verdict"] == "matched":
            official.append({**verdict, "found_by": label})
        elif (looks_like_a_game_reel(title)
              and title_names_both_teams(title, game["opponent_tricode"])):
            unofficial.append(verdict)

    if official:
        best = official[0]
        return {**base_row(game), "outcome": "official_reel",
                "video_id": best["video_id"], "title": best["title"],
                "channel_title": best["channel_title"],
                "published_at": best["published_at"],
                "duration": best["duration"],
                "found_by": best["found_by"],
                "watch_url": best["watch_url"],
                "region_restriction": best["region_restriction"],
                "unofficial_candidates": len(unofficial),
                "note": ""}, False

    if unofficial:
        return {**base_row(game), "outcome": "unofficial_only",
                "unofficial_candidates": len(unofficial),
                "note": f"{len(unofficial)} matching reel(s), none official"}, \
            False

    return {**base_row(game), "outcome": "nothing_found",
            "note": "candidates returned, none matched this game"}, False


def base_row(game: dict) -> dict:
    away, home = home_and_away(game)
    return {"season": game["season"], "game_id": game["game_id"],
            "game_date": game["game_date"].date().isoformat(),
            "matchup": game["matchup"],
            "opponent_tricode": game["opponent_tricode"],
            "away": away, "home": home,
            "position_in_season": game["position_in_season"],
            "video_id": None, "title": None, "channel_title": None,
            "published_at": None, "duration": None, "found_by": None,
            "watch_url": None, "region_restriction": None,
            "unofficial_candidates": 0}


def run(key: str, games: list, channels: dict, out_csv=None) -> pd.DataFrame:
    official_ids = {c["channel_id"] for c in channels.values()}
    rows, stopped = [], False

    for i, game in enumerate(games, start=1):
        if stopped:
            rows.append({**base_row(game), "outcome": "not_tested",
                         "note": "run stopped before this game"})
            continue
        logger.info("[%2d/%d] %s  %s  %s", i, len(games), game["season"],
                    game["matchup"], game["game_date"].date())
        row, quota_dead = probe_game(key, game, official_ids)
        rows.append(row)
        logger.info("        %s", row["outcome"])
        if quota_dead:
            logger.error("QUOTA EXHAUSTED. Stopping. Remaining games are "
                         "recorded as not_tested, NOT as absences.")
            stopped = True
        # Written as we go, so a stop loses nothing.
        if out_csv is not None:
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    return pd.DataFrame(rows)


def build_report(frame: pd.DataFrame) -> str:
    tested = frame.loc[frame["outcome"].ne("not_tested")]
    not_tested = frame.loc[frame["outcome"].eq("not_tested")]
    official = frame.loc[frame["outcome"].eq("official_reel")]

    lines = [
        "=" * 78,
        "PHASE 12d - OFFICIAL HIGHLIGHT COVERAGE BY SEASON",
        "=" * 78,
        "",
        "  Three games per season, drawn at fixed fractions through each",
        "  schedule (0.25, 0.50, 0.75). Deterministic and deliberately spread:",
        "  two earlier conclusions in this project came from samples that had",
        "  quietly clustered, both times in the direction of the conclusion.",
        "",
        "  Metadata only. No download, no scraping, no re-hosting.",
        "",
    ]

    if len(not_tested):
        lines += [
            f"  !! {len(not_tested)} of {len(frame)} games were NOT TESTED.",
            "  Those are recorded as untested, never as absences. A spent",
            "  quota returns 403 for everything, which would otherwise make",
            "  the tail of the run look like a coverage cliff.",
            "",
        ]

    lines += [
        f"  games tested       {len(tested)} of {len(frame)}",
        f"  official reel      {len(official)}"
        + (f"  ({len(official) / len(tested):.0%} of tested)"
           if len(tested) else ""),
        f"  unofficial only    {int(frame['outcome'].eq('unofficial_only').sum())}",
        f"  nothing found      {int(frame['outcome'].eq('nothing_found').sum())}",
        "",
        "  'unofficial only' means a matching full-game reel EXISTS but sits",
        "  on a re-upload channel. Those are not usable here, by instruction",
        "  and on the merits, but it is a different fact from no coverage.",
        "",
        "=" * 78,
        "BY SEASON",
        "=" * 78,
        "",
        f"  {'season':<10}{'tested':>8}{'official':>10}{'unoff':>8}"
        f"{'none':>7}{'untested':>10}{'rate':>8}",
    ]
    for season, group in frame.groupby("season"):
        done = group.loc[group["outcome"].ne("not_tested")]
        off = int(group["outcome"].eq("official_reel").sum())
        rate = f"{off / len(done):.0%}" if len(done) else "n/a"
        lines.append(
            f"  {season:<10}{len(done):>8}{off:>10}"
            f"{int(group['outcome'].eq('unofficial_only').sum()):>8}"
            f"{int(group['outcome'].eq('nothing_found').sum()):>7}"
            f"{int(group['outcome'].eq('not_tested').sum()):>10}{rate:>8}")

    if len(official):
        lines += ["", "=" * 78, "WHICH CONVENTION WORKED, BY SEASON",
                  "=" * 78, "",
                  "  A run across all 636 games needs this per era.", ""]
        for label, group in official.groupby("found_by"):
            lines.append(f"  {label}")
            lines.append(f"    {', '.join(sorted(group['season'].unique()))}")

        lines += ["", "=" * 78, "OFFICIAL REELS FOUND", "=" * 78, ""]
        for row in official.itertuples():
            lines.append(f"  {row.season}  {row.matchup}  {row.game_date}")
            lines.append(f"    {str(row.title)[:68]}")
            lines.append(f"    {row.channel_title} | {row.duration} | "
                         f"{row.watch_url}")

    usable_seasons = sorted(
        official["season"].unique()) if len(official) else []
    lines += ["", "=" * 78, "WHAT THIS MEANS FOR THE PANEL", "=" * 78, ""]
    if not usable_seasons:
        lines += ["  No season produced an official reel. There is nothing to",
                  "  build a highlights panel on."]
    else:
        lines += [
            f"  Seasons with at least one official reel: "
            f"{', '.join(usable_seasons)}",
            "",
            "  A panel would show 'Game highlights' for games in those",
            "  seasons and NOTHING AT ALL elsewhere. It must never be",
            "  labelled 'Current play', must never block the dashboard, and",
            "  must degrade silently when a video is pulled or made private.",
            "",
            "  A full 636-game precompute at two searches per game is roughly",
            f"  {636 * VARIANTS_PER_GAME * UNITS_PER_SEARCH:,} quota units, "
            f"about {636 * VARIANTS_PER_GAME * UNITS_PER_SEARCH // DAILY_FREE_QUOTA + 1}"
            " days of the free tier.",
            "  Restricting it to the seasons above would cut that "
            "proportionally.",
        ]

    lines += [
        "",
        "=" * 78,
        "WHAT THIS DOES NOT TELL YOU",
        "=" * 78,
        "",
        "  Three games a season is an estimate with wide error bars, not a",
        "  coverage figure. A season showing 3 of 3 could still be 60% across",
        "  its 82 games.",
        "",
        "  A game with no official reel found may still have one under a title",
        "  neither variant covers. Absence is weaker evidence than presence.",
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
    estimate = len(games) * VARIANTS_PER_GAME * UNITS_PER_SEARCH
    print("=" * 70)
    print(f"  {len(games)} games, {VARIANTS_PER_GAME} searches each")
    print(f"  estimated quota: ~{estimate:,} units of {DAILY_FREE_QUOTA:,}/day")
    print("=" * 70)
    if estimate > DAILY_FREE_QUOTA:
        print("  WARNING: this may exceed one day's free quota. The run will")
        print("  stop cleanly and mark the rest untested if it does.")
    print()

    logger.info("resolving official channels")
    channels = resolve_channels(key)
    if not channels:
        raise SystemExit("Could not resolve either official channel.")

    out_csv = config.INTERIM_DIR / "youtube_coverage.csv"
    frame = run(key, games, channels, out_csv=out_csv)

    report = build_report(frame)
    print(report)
    out = config.REPORTS_DIR / "youtube_coverage.txt"
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\nRow detail: {out_csv}")
    print(f"Saved to  : {out}")
    print("\nREAD ONLY. No video was downloaded, scraped or cached.")
    return frame


if __name__ == "__main__":
    main()
