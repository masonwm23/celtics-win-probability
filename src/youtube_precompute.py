"""
Phase 12e: build a game_id -> video_id mapping, cheaply and verifiably.

WHY NOT SEARCH
--------------
Phase 12d used search.list, at 100 quota units per call. Mapping 636 games that
way is ~127,000 units, about thirteen days of the free tier.

playlistItems.list costs ONE unit per page of fifty. Every channel has an
"uploads" playlist containing everything it has ever posted, so enumerating the
NBA and Celtics channels and matching titles LOCALLY costs on the order of a
thousand units. One run instead of thirteen days, for the same data.

There is a known limitation: an uploads playlist can stop paginating before a
very large channel's full history is reached. That is not assumed either way.
The run records how far back it actually got, and if it stops short of
2016-17 the report says so rather than reporting the shortfall as an absence.

VERIFICATION: TEAMS, DATE, TITLE
--------------------------------
A reel under the wrong game is the failure this whole phase exists to avoid, so
a candidate is CONFIRMED only when every one of these holds:

  teams   exactly two distinct NBA teams parse out of the title, one of them
          Boston and the other this game's opponent
  date    if the title carries a date it must equal the game date; if it does
          not, the upload must fall in a tight window after tip-off
  title   the title reads as a game reel, not a player mixtape or a top-ten
  status  official channel, public, and embeddable
  unique  exactly one candidate for the game, and that video is not also the
          best candidate for a different game

Anything that satisfies teams and status but fails the rest goes to a REVIEW
report. It is never written to the mapping and never displayed. An uncertain
match is treated as no match.

Unofficial channels are not considered at any point. They are not in the
candidate pool at all, because the pool is built from two official uploads
playlists.

WHAT IT WRITES
--------------
  data/interim/highlights_map.json    game_id -> video, CONFIRMED only
  data/interim/youtube_uploads.csv    the raw enumeration, so a re-run is free
  data/interim/youtube_precompute.csv every candidate and its verdict
  reports/youtube_precompute.txt      coverage by season
  reports/youtube_review.txt          the ambiguity report

NOTHING IN data/serving IS TOUCHED. The dashboard does not read any of this.
Moving the mapping into the serving layer is a separate, later step that
happens only after the reports are read and approved.

STILL METADATA ONLY. No download, no scraping, no re-hosting.
"""

import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from src import config
from src.youtube_probe import load_api_key, safe_api_get

logger = logging.getLogger(__name__)

OFFICIAL_HANDLES = ("@NBA", "@celtics")

PAGE_SIZE = 50
DELAY = 0.05

# Stop paging once the enumeration is comfortably older than the first game.
EARLIEST_GAME_BUFFER = timedelta(days=30)

# A reel published the same night runs past midnight UTC, so the tight window
# is asymmetric. The loose window only ever downgrades to review, never to a
# confirmed match.
TIGHT_WINDOW_BEFORE = timedelta(hours=6)
TIGHT_WINDOW_AFTER = timedelta(days=2)

# Full game reels run about ten minutes. Well outside that is a different kind
# of video and goes to review rather than into the mapping.
MIN_SECONDS = 240
MAX_SECONDS = 1500

REEL_WORDS = ("full game highlights", "game highlights", "full game recap",
              "game recap", "full highlights")

# Words that mean the video is about one player or one moment, not the game.
NOT_A_REEL = ("mixtape", "top 10", "top ten", "top 5", "top five",
              "every bucket", "all buckets", "career high", "highlights vs",
              "full highlights vs", "best plays", "reacts", "reaction",
              "press conference", "postgame", "interview", "mic'd up",
              "micd up", "1st half", "first half", "2nd half", "half highlights")

# Tokens that identify a team unambiguously. City names shared between two
# franchises are deliberately absent: "Los Angeles" and "LA" cannot pick
# between the Lakers and the Clippers, so only the nicknames are trusted.
TEAM_TOKENS = {
    "ATL": ["hawks", "atlanta"], "BKN": ["nets", "brooklyn"],
    "BOS": ["celtics", "boston"], "CHA": ["hornets", "charlotte"],
    "CHI": ["bulls", "chicago"], "CLE": ["cavaliers", "cavs", "cleveland"],
    "DAL": ["mavericks", "mavs", "dallas"], "DEN": ["nuggets", "denver"],
    "DET": ["pistons", "detroit"], "GSW": ["warriors", "golden state"],
    "HOU": ["rockets", "houston"], "IND": ["pacers", "indiana"],
    "LAC": ["clippers"], "LAL": ["lakers"],
    "MEM": ["grizzlies", "memphis"], "MIA": ["heat", "miami"],
    "MIL": ["bucks", "milwaukee"], "MIN": ["timberwolves", "wolves",
                                           "minnesota"],
    "NOP": ["pelicans", "new orleans"], "NYK": ["knicks", "new york"],
    "OKC": ["thunder", "oklahoma city"], "ORL": ["magic", "orlando"],
    "PHI": ["76ers", "sixers", "philadelphia"], "PHX": ["suns", "phoenix"],
    "POR": ["trail blazers", "blazers", "portland"],
    "SAC": ["kings", "sacramento"], "SAS": ["spurs", "san antonio"],
    "TOR": ["raptors", "toronto"], "UTA": ["jazz", "utah"],
    "WAS": ["wizards", "washington"],
}

MONTHS = {m.lower(): i for i, m in enumerate(
    ("January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"), start=1)}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})

DATE_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(MONTHS, key=len, reverse=True)) +
    r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------

def uploads_playlists(key: str) -> dict:
    """{handle: (channel_id, uploads_playlist_id, title)}."""
    out = {}
    for handle in OFFICIAL_HANDLES:
        payload, error = safe_api_get("channels", key,
                                      part="snippet,contentDetails",
                                      forHandle=handle)
        if error or not payload or not payload.get("items"):
            logger.warning("could not resolve %s: %s", handle,
                           error or "no items")
            continue
        item = payload["items"][0]
        playlist = (item["contentDetails"]["relatedPlaylists"]
                    .get("uploads"))
        out[handle] = (item["id"], playlist, item["snippet"]["title"])
        logger.info("  %s -> channel %s, uploads %s", handle, item["id"],
                    playlist)
    return out


def enumerate_uploads(key: str, handle: str, playlist_id: str,
                      channel_id: str, stop_before: datetime) -> tuple:
    """
    Page an uploads playlist newest first until older than `stop_before`.

    Returns (rows, deepest_date, exhausted). `exhausted` is True when the
    playlist ran out of pages before reaching the target, which is the known
    depth limitation and must not be read as "the channel posted nothing".
    """
    rows, token, pages = [], None, 0
    deepest = None
    while True:
        params = {"part": "snippet,contentDetails", "playlistId": playlist_id,
                  "maxResults": PAGE_SIZE}
        if token:
            params["pageToken"] = token
        payload, error = safe_api_get("playlistItems", key, **params)
        if error:
            logger.warning("  %s page %d: %s", handle, pages, error)
            return rows, deepest, True
        pages += 1

        for entry in payload.get("items", []):
            snippet = entry.get("snippet") or {}
            details = entry.get("contentDetails") or {}
            published = (details.get("videoPublishedAt")
                         or snippet.get("publishedAt") or "")
            rows.append({
                "handle": handle, "channel_id": channel_id,
                "video_id": details.get("videoId"),
                "title": snippet.get("title"),
                "published_at": published,
            })
            parsed = parse_iso(published)
            if parsed and (deepest is None or parsed < deepest):
                deepest = parsed

        if pages % 10 == 0:
            logger.info("    %s: %d pages, %d videos, back to %s", handle,
                        pages, len(rows),
                        deepest.date() if deepest else "?")

        token = payload.get("nextPageToken")
        if not token:
            return rows, deepest, True          # playlist genuinely ended
        if deepest and deepest < stop_before:
            return rows, deepest, False         # reached far enough
        time.sleep(DELAY)


def parse_iso(text):
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
    return None


# ---------------------------------------------------------------------------
# Title parsing
# ---------------------------------------------------------------------------

def teams_in_title(title) -> set:
    """
    Every NBA team named in a title, as tricodes.

    Deduplicating by tricode is what lets "Full Game Recap: Celtics vs Heat |
    Vintage Wade On Display In Miami" resolve to {BOS, MIA} rather than three
    teams, and "TRAIL BLAZERS vs CELTICS | ... To Lead Portland" to {POR, BOS}.
    """
    text = (title or "").lower()
    found = set()
    for tricode, tokens in TEAM_TOKENS.items():
        for token in tokens:
            if re.search(rf"(?<![a-z]){re.escape(token)}(?![a-z])", text):
                found.add(tricode)
                break
    return found


def date_in_title(title):
    match = DATE_PATTERN.search(title or "")
    if not match:
        return None
    month = MONTHS.get(match.group(1).lower())
    if not month:
        return None
    try:
        return datetime(int(match.group(3)), month, int(match.group(2)),
                        tzinfo=timezone.utc)
    except ValueError:
        return None


def reads_as_a_game_reel(title) -> bool:
    text = (title or "").lower()
    if any(bad in text for bad in NOT_A_REEL):
        return False
    return any(word in text for word in REEL_WORDS)


def iso_duration_seconds(text) -> int:
    match = re.match(r"^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$", text or "")
    if not match:
        return 0
    hours, minutes, seconds = (int(g or 0) for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def assess(candidate: dict, game: dict) -> dict:
    """
    One candidate against one game. Returns the verdict and every reason.

    `confirmed` requires all four axes. `review` means it named the right two
    teams on an official channel but something else did not line up, and it is
    never written to the mapping.
    """
    title = candidate["title"]
    teams = teams_in_title(title)
    wanted = {"BOS", game["opponent_tricode"]}
    published = parse_iso(candidate["published_at"])
    stated = date_in_title(title)
    seconds = iso_duration_seconds(candidate.get("duration"))

    problems = []
    if teams != wanted:
        problems.append(
            f"title teams {sorted(teams) or 'none'} != {sorted(wanted)}")
    if not candidate.get("embeddable"):
        problems.append("not embeddable")
    if candidate.get("privacy") != "public":
        problems.append(f"privacy {candidate.get('privacy')}")
    if not reads_as_a_game_reel(title):
        problems.append("title does not read as a game reel")

    if stated is not None:
        if stated.date() != game["game_date"].date():
            problems.append(f"title date {stated.date()} != "
                            f"{game['game_date'].date()}")
    elif not published or not (
            game["game_date"] - TIGHT_WINDOW_BEFORE <= published
            <= game["game_date"] + TIGHT_WINDOW_AFTER):
        problems.append("no date in title and upload outside the tight window")

    if seconds and not (MIN_SECONDS <= seconds <= MAX_SECONDS):
        problems.append(f"duration {seconds}s outside {MIN_SECONDS}-"
                        f"{MAX_SECONDS}s")

    right_teams_official = (teams == wanted and candidate.get("embeddable")
                            and candidate.get("privacy") == "public")
    if not problems:
        verdict = "confirmed"
    elif right_teams_official:
        verdict = "review"
    else:
        verdict = "rejected"

    return {"verdict": verdict, "problems": "; ".join(problems),
            "title_date": stated.date().isoformat() if stated else "",
            "duration_seconds": seconds,
            "teams_parsed": ",".join(sorted(teams))}


def match_games(candidates: pd.DataFrame, games: list) -> pd.DataFrame:
    """
    Every candidate against every plausible game, then uniqueness.

    Two collisions are checked, because either would put a reel under the wrong
    scoreboard: two candidates confirmed for one game, and one candidate
    confirmed for two games.
    """
    rows = []
    by_opponent = {}
    for game in games:
        by_opponent.setdefault(game["opponent_tricode"], []).append(game)

    for candidate in candidates.to_dict("records"):
        teams = teams_in_title(candidate["title"])
        if "BOS" not in teams:
            continue
        opponents = teams - {"BOS"}
        for opponent in opponents:
            for game in by_opponent.get(opponent, []):
                published = parse_iso(candidate["published_at"])
                stated = date_in_title(candidate["title"])
                near = (stated and stated.date() == game["game_date"].date())
                if not near and published:
                    near = (game["game_date"] - timedelta(days=1) <= published
                            <= game["game_date"] + timedelta(days=5))
                if not near:
                    continue
                rows.append({
                    "game_id": game["game_id"], "season": game["season"],
                    "game_date": game["game_date"].date().isoformat(),
                    "matchup": game["matchup"],
                    "opponent_tricode": game["opponent_tricode"],
                    **candidate, **assess(candidate, game)})

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    confirmed = frame["verdict"].eq("confirmed")
    per_game = frame.loc[confirmed].groupby("game_id")["video_id"].nunique()
    per_video = frame.loc[confirmed].groupby("video_id")["game_id"].nunique()

    contested_games = set(per_game[per_game > 1].index)
    contested_videos = set(per_video[per_video > 1].index)

    def downgrade(row):
        if row["verdict"] != "confirmed":
            return row["verdict"], row["problems"]
        if row["game_id"] in contested_games:
            return "review", "more than one confirmed candidate for this game"
        if row["video_id"] in contested_videos:
            return "review", "this video is confirmed for more than one game"
        return "confirmed", row["problems"]

    applied = frame.apply(downgrade, axis=1, result_type="expand")
    frame["verdict"], frame["problems"] = applied[0], applied[1]
    return frame


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def build_coverage_report(frame, games, deepest, exhausted, uploads) -> str:
    confirmed = (frame.loc[frame["verdict"].eq("confirmed")]
                 if len(frame) else frame)
    mapped = set(confirmed["game_id"]) if len(confirmed) else set()
    seasons = sorted({g["season"] for g in games})

    lines = [
        "=" * 78,
        "PHASE 12e - HIGHLIGHT PRECOMPUTE, COVERAGE",
        "=" * 78,
        "",
        "  Built by enumerating the official uploads playlists and matching",
        "  titles locally. No search.list, so the whole run costs roughly a",
        "  thousand quota units instead of the ~127,000 a per-game search",
        "  would have cost.",
        "",
        "  Metadata only. No download, no scraping, no re-hosting. Unofficial",
        "  channels are not in the candidate pool at all.",
        "",
        f"  uploads enumerated  {len(uploads):,}",
        f"  oldest reached      {deepest.date() if deepest else 'n/a'}",
        f"  games               {len(games)}",
        f"  CONFIRMED           {len(mapped)}  "
        f"({len(mapped) / len(games):.1%})" if games else "",
        f"  sent to review      "
        f"{frame['verdict'].eq('review').sum() if len(frame) else 0}",
        "",
    ]

    if exhausted:
        lines += [
            "  !! The uploads playlist stopped paginating before reaching the",
            "  target date. An uploads playlist has a depth limit, so games",
            "  older than the date above were NEVER TESTED. Their absence",
            "  from the mapping is not evidence that no reel exists.",
            "",
        ]

    lines += ["=" * 78, "BY SEASON", "=" * 78, "",
              f"  {'season':<10}{'games':>7}{'confirmed':>11}{'review':>8}"
              f"{'unmatched':>11}{'rate':>8}"]
    for season in seasons:
        in_season = [g for g in games if g["season"] == season]
        ids = {g["game_id"] for g in in_season}
        got = len(ids & mapped)
        reviewed = (frame.loc[frame["season"].eq(season)
                              & frame["verdict"].eq("review"), "game_id"]
                    .nunique() if len(frame) else 0)
        lines.append(f"  {season:<10}{len(in_season):>7}{got:>11}"
                     f"{reviewed:>8}{len(ids) - got:>11}"
                     f"{got / len(ids):>7.0%}")

    lines += [
        "",
        "=" * 78,
        "WHAT THE PANEL WOULD DO",
        "=" * 78,
        "",
        "  Games in the mapping get an optional panel labelled",
        "  'Game Highlights', stating plainly that it is a game-level recap",
        "  and NOT synchronised with the probability cursor.",
        "",
        "  Every other game hides the panel entirely. No placeholder, no",
        "  error, no unofficial fallback. That includes 2016-17 and 2017-18,",
        "  where the NBA did not publish per-game reels to its own channel.",
        "",
        "  The synchronised play-by-play shot animation is a separate feature",
        "  and is not replaced by this.",
        "",
        "  Nothing in data/serving was touched by this run. The dashboard",
        "  does not read the mapping yet.",
        "=" * 78,
    ]
    return "\n".join(lines)


def build_review_report(frame) -> str:
    review = frame.loc[frame["verdict"].eq("review")] if len(frame) else frame
    lines = [
        "=" * 78,
        "PHASE 12e - AMBIGUITY REVIEW",
        "=" * 78,
        "",
        "  Candidates on an official channel that named the right two teams",
        "  but failed verification for some other reason.",
        "",
        "  NONE of these are in the mapping and NONE would be displayed. An",
        "  uncertain match is treated as no match. They are listed so the",
        "  verification rules can be judged rather than trusted.",
        "",
        f"  candidates in review: {len(review)}",
        f"  games affected      : "
        f"{review['game_id'].nunique() if len(review) else 0}",
        "",
    ]
    if not len(review):
        lines += ["  Nothing to review.", "=" * 78]
        return "\n".join(lines)

    lines += ["  reasons, most common first:"]
    reasons = {}
    for problems in review["problems"]:
        for problem in str(problems).split("; "):
            key = re.sub(r"\d+", "N", problem)
            reasons[key] = reasons.get(key, 0) + 1
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        lines.append(f"    {count:>4}  {reason}")

    lines += ["", "=" * 78, "EVERY CANDIDATE IN REVIEW", "=" * 78, ""]
    for row in review.sort_values(["season", "game_date"]).itertuples():
        lines += [
            f"  {row.season}  {row.matchup}  {row.game_date}",
            f"    {str(row.title)[:70]}",
            f"    {row.handle} | published {row.published_at} | "
            f"{row.duration_seconds}s",
            f"    https://www.youtube.com/watch?v={row.video_id}",
            f"    WHY: {row.problems}",
            "",
        ]
    lines.append("=" * 78)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_games(index: pd.DataFrame) -> list:
    """
    Every game, with `is_home` included.

    `is_home` is not used by this module, but Phase 12f builds search titles
    from these dicts and the NBA titles reels AWAY at HOME, so omitting it
    raises a KeyError on the first game. Caught by a test rather than by a
    user watching a multi-day job die on its first call.
    """
    games = []
    for row in index.itertuples():
        games.append({
            "season": row.SEASON,
            "game_id": str(row.GAME_ID).zfill(10),
            "game_date": pd.Timestamp(row.GAME_DATE).tz_localize("UTC")
            .to_pydatetime(),
            "opponent_tricode": row.OPPONENT_ABBREV,
            "matchup": row.MATCHUP,
            "is_home": bool(row.IS_HOME),
        })
    return games


def hydrate_candidates(key: str, uploads: pd.DataFrame) -> pd.DataFrame:
    """videos.list only for titles that could plausibly be a game reel."""
    plausible = uploads.loc[
        uploads["title"].apply(reads_as_a_game_reel)
        & uploads["title"].apply(lambda t: "BOS" in teams_in_title(t))]
    ids = [v for v in plausible["video_id"].dropna().unique()]
    logger.info("hydrating %d plausible candidate(s) of %d upload(s)",
                len(ids), len(uploads))

    details = {}
    for start in range(0, len(ids), 50):
        payload, error = safe_api_get(
            "videos", key, part="snippet,status,contentDetails",
            id=",".join(ids[start:start + 50]))
        if error or not payload:
            logger.warning("videos.list failed: %s", error)
            continue
        for item in payload.get("items", []):
            details[item["id"]] = item
        time.sleep(DELAY)

    rows = []
    for row in plausible.to_dict("records"):
        item = details.get(row["video_id"])
        if not item:
            continue
        rows.append({
            **row,
            "embeddable": bool((item.get("status") or {}).get("embeddable")),
            "privacy": (item.get("status") or {}).get("privacyStatus"),
            "duration": (item.get("contentDetails") or {}).get("duration"),
        })
    return pd.DataFrame(rows)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    config.ensure_dirs()

    key = load_api_key()
    index_path = config.RAW_DIR / "game_index.csv"
    if not index_path.exists():
        raise SystemExit(f"{index_path} not found.")

    games = load_games(pd.read_csv(index_path))
    earliest = min(g["game_date"] for g in games) - EARLIEST_GAME_BUFFER
    logger.info("%d games, earliest %s", len(games), earliest.date())

    cache = config.INTERIM_DIR / "youtube_uploads.csv"
    if cache.exists():
        uploads = pd.read_csv(cache)
        deepest = min((parse_iso(p) for p in uploads["published_at"]
                       if parse_iso(p)), default=None)
        exhausted = bool(deepest and deepest > earliest)
        logger.info("reusing cached enumeration: %d uploads back to %s",
                    len(uploads), deepest.date() if deepest else "?")
    else:
        playlists = uploads_playlists(key)
        if not playlists:
            raise SystemExit("Could not resolve either official channel.")
        all_rows, deepest, exhausted = [], None, False
        for handle, (channel_id, playlist_id, title) in playlists.items():
            logger.info("enumerating %s (%s)", handle, title)
            rows, got_to, ran_out = enumerate_uploads(
                key, handle, playlist_id, channel_id, earliest)
            logger.info("  %s: %d uploads, oldest %s, exhausted=%s", handle,
                        len(rows), got_to.date() if got_to else "?", ran_out)
            all_rows.extend(rows)
            if got_to and (deepest is None or got_to < deepest):
                deepest = got_to
            exhausted = exhausted or (ran_out and got_to and got_to > earliest)
        uploads = pd.DataFrame(all_rows)
        uploads.to_csv(cache, index=False)

    if uploads.empty:
        raise SystemExit("No uploads enumerated.")

    candidates = hydrate_candidates(key, uploads)
    frame = match_games(candidates, games) if len(candidates) else pd.DataFrame()

    mapping = {}
    if len(frame):
        for row in frame.loc[frame["verdict"].eq("confirmed")].itertuples():
            mapping[row.game_id] = {
                "video_id": row.video_id,
                "title": row.title,
                "channel": row.handle,
                "published_at": row.published_at,
                "duration_seconds": int(row.duration_seconds),
                "label": "Game Highlights",
                "note": ("Game-level recap published by the official channel. "
                         "Not synchronised with the probability cursor."),
            }

    out_map = config.INTERIM_DIR / "highlights_map.json"
    out_map.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    if len(frame):
        frame.to_csv(config.INTERIM_DIR / "youtube_precompute.csv", index=False)

    coverage = build_coverage_report(frame, games, deepest, exhausted, uploads)
    review = build_review_report(frame)
    print(coverage)
    print()
    print(review)

    (config.REPORTS_DIR / "youtube_precompute.txt").write_text(
        coverage + "\n", encoding="utf-8")
    (config.REPORTS_DIR / "youtube_review.txt").write_text(
        review + "\n", encoding="utf-8")

    print(f"\nMapping   : {out_map}  ({len(mapping)} games)")
    print(f"Coverage  : {config.REPORTS_DIR / 'youtube_precompute.txt'}")
    print(f"Review    : {config.REPORTS_DIR / 'youtube_review.txt'}")
    print("\nREAD ONLY. No video downloaded, scraped or cached. Nothing in")
    print("data/serving was touched and the dashboard is unchanged.")
    return frame, mapping


if __name__ == "__main__":
    main()
