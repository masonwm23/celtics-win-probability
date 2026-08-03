"""
Phase 12: is there an OFFICIAL, EMBEDDABLE game highlight video per game?

WHAT THIS IS ALLOWED TO DO, AND WHAT IT IS NOT
----------------------------------------------
Not allowed, and not attempted anywhere in this file:

  - downloading video
  - scraping YouTube pages
  - re-hosting, mirroring or caching any footage

This module talks only to the official YouTube Data API v3 and reads only
METADATA: title, channel, publish date, and the `status.embeddable` flag. It
never requests a video stream. If a highlight panel is eventually built it
would use YouTube's own iframe player, which is the embed mechanism YouTube
publishes for this purpose, so the footage is served by YouTube and stays under
their player, their ads and their controls.

GAME LEVEL, NOT PLAY LEVEL
--------------------------
Phase 11 established there is no play-level video available. This is a
different and much weaker claim: one highlight reel per GAME, matched by
season, date and opponent, and labelled **"Game highlights"**. It is not
synchronised to the probability cursor and must never be labelled "Current
play", because it is not the current play.

The synchronised visualisation stays the play-by-play figure animation drawn
from our own coordinates. Video, if it happens at all, is an optional extra
panel that the dashboard works fine without.

WHY MATCHING IS STILL THE HARD PART
-----------------------------------
A highlight reel for the wrong game, sitting under a scoreboard for this game,
is the same failure as a clip of the wrong play. So a candidate counts as
matched only if ALL of these hold:

  1. it is on an official channel, resolved at run time from the @NBA and
     @celtics handles rather than hardcoded;
  2. `status.embeddable` is true;
  3. `status.privacyStatus` is "public";
  4. it was published within the window around the game date;
  5. its title names BOTH teams.

Anything failing any test is reported with the reason and never counted as
available. Region restrictions are recorded separately, because a video that
is embeddable but blocked in the viewer's country is not usable either.

SCOPE OF THIS RUN
-----------------
Three games from three different seasons, as instructed. This reports
availability only. No dashboard change is made and none should be made until
the numbers are read.

READ ONLY. Writes reports/youtube_probe.txt and data/interim/youtube_probe.csv.
"""

import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from src import config

logger = logging.getLogger(__name__)

API_ROOT = "https://www.googleapis.com/youtube/v3"

# Resolved at run time via channels.list?forHandle. Hardcoding channel ids is
# how a probe silently starts trusting the wrong channel after a rebrand.
OFFICIAL_HANDLES = ("@NBA", "@celtics")

# Highlights normally post within hours. The window is generous on the late
# side for time zones and slow uploads, and one day early because a game
# starting late Eastern can be published on the same UTC date it began.
PUBLISH_WINDOW_BEFORE = timedelta(days=1)
PUBLISH_WINDOW_AFTER = timedelta(days=4)

RESULTS_PER_SEARCH = 10

TEAM_NAMES = {
    "ATL": ("Hawks", "Atlanta"), "BKN": ("Nets", "Brooklyn"),
    "BOS": ("Celtics", "Boston"), "CHA": ("Hornets", "Charlotte"),
    "CHI": ("Bulls", "Chicago"), "CLE": ("Cavaliers", "Cleveland"),
    "DAL": ("Mavericks", "Dallas"), "DEN": ("Nuggets", "Denver"),
    "DET": ("Pistons", "Detroit"), "GSW": ("Warriors", "Golden State"),
    "HOU": ("Rockets", "Houston"), "IND": ("Pacers", "Indiana"),
    "LAC": ("Clippers", "LA Clippers"), "LAL": ("Lakers", "Los Angeles"),
    "MEM": ("Grizzlies", "Memphis"), "MIA": ("Heat", "Miami"),
    "MIL": ("Bucks", "Milwaukee"), "MIN": ("Timberwolves", "Minnesota"),
    "NOP": ("Pelicans", "New Orleans"), "NYK": ("Knicks", "New York"),
    "OKC": ("Thunder", "Oklahoma City"), "ORL": ("Magic", "Orlando"),
    "PHI": ("76ers", "Philadelphia"), "PHX": ("Suns", "Phoenix"),
    "POR": ("Trail Blazers", "Portland"), "SAC": ("Kings", "Sacramento"),
    "SAS": ("Spurs", "San Antonio"), "TOR": ("Raptors", "Toronto"),
    "UTA": ("Jazz", "Utah"), "WAS": ("Wizards", "Washington"),
}


# ---------------------------------------------------------------------------
# Credentials. The key is never printed, never logged, never stored by this
# code. It is read from the environment or from a gitignored file the user
# creates themselves.
# ---------------------------------------------------------------------------

KEY_FILE = ".youtube_api_key"


def load_api_key() -> str:
    key = (os.environ.get("YOUTUBE_API_KEY") or "").strip()
    if key:
        return key
    path = config.PROJECT_ROOT / KEY_FILE
    if path.exists():
        key = path.read_text(encoding="utf-8").strip()
        if key:
            return key
    raise SystemExit(
        "No YouTube Data API key found.\n\n"
        "This probe reads only public metadata from the official YouTube Data\n"
        "API v3. It never downloads or scrapes video.\n\n"
        "To get a key (free):\n"
        "  1. console.cloud.google.com -> create or pick a project\n"
        "  2. APIs & Services -> Library -> enable 'YouTube Data API v3'\n"
        "  3. APIs & Services -> Credentials -> Create credentials -> API key\n\n"
        f"Then put it in a file called {KEY_FILE} in the project root:\n"
        f"  echo 'YOUR_KEY_HERE' > {config.PROJECT_ROOT / KEY_FILE}\n\n"
        f"{KEY_FILE} is in .gitignore, so it will not be committed.\n"
        "Do not paste the key into a chat window or into any source file.")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def api_get(endpoint: str, key: str, **params) -> dict:
    """One GET against the YouTube Data API. Raises on transport failure."""
    params["key"] = key
    url = f"{API_ROOT}/{endpoint}?{urlencode(params)}"
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def safe_api_get(endpoint: str, key: str, **params) -> tuple:
    """(payload, error). Never raises, and never echoes the key in the error."""
    try:
        return api_get(endpoint, key, **params), None
    except Exception as exc:                       # noqa: BLE001
        message = str(exc).replace(key, "<redacted>") if key else str(exc)
        return None, f"{type(exc).__name__}: {message}"


def resolve_channels(key: str) -> dict:
    """{handle: channel_id} for the official channels, looked up not assumed."""
    resolved = {}
    for handle in OFFICIAL_HANDLES:
        payload, error = safe_api_get("channels", key, part="snippet",
                                      forHandle=handle)
        if error or not payload or not payload.get("items"):
            logger.warning("could not resolve %s: %s", handle,
                           error or "no items")
            continue
        item = payload["items"][0]
        resolved[handle] = {"channel_id": item["id"],
                            "title": item["snippet"]["title"]}
        logger.info("  %s -> %s (%s)", handle, item["id"],
                    item["snippet"]["title"])
    return resolved


def search_channel(key: str, channel_id: str, query: str,
                   after: datetime, before: datetime) -> tuple:
    return safe_api_get(
        "search", key, part="snippet", channelId=channel_id, q=query,
        type="video", maxResults=RESULTS_PER_SEARCH, order="relevance",
        publishedAfter=after.strftime("%Y-%m-%dT%H:%M:%SZ"),
        publishedBefore=before.strftime("%Y-%m-%dT%H:%M:%SZ"))


def video_details(key: str, video_ids: list) -> dict:
    """{video_id: item} with status and region restrictions."""
    if not video_ids:
        return {}
    payload, error = safe_api_get(
        "videos", key, part="snippet,status,contentDetails",
        id=",".join(video_ids[:50]))
    if error or not payload:
        logger.warning("videos.list failed: %s", error)
        return {}
    return {item["id"]: item for item in payload.get("items", [])}


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def normalise(text) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^a-z0-9 ]", " ", text.lower())


def title_names_both_teams(title: str, opponent_tricode: str) -> bool:
    """
    Both teams must appear. A reel titled only "Celtics highlights" could be
    any of 82 games that season.
    """
    text = normalise(title)
    celtics = "celtics" in text or "boston" in text
    nickname, city = TEAM_NAMES.get(opponent_tricode, (opponent_tricode, ""))
    opponent = (normalise(nickname) in text
                or (bool(city) and normalise(city) in text))
    return celtics and opponent


def region_note(item: dict) -> str:
    restriction = (item.get("contentDetails") or {}).get("regionRestriction")
    if not restriction:
        return ""
    if restriction.get("blocked"):
        return f"blocked in {len(restriction['blocked'])} region(s)"
    if restriction.get("allowed"):
        return f"allowed in only {len(restriction['allowed'])} region(s)"
    return ""


def classify(item: dict, game: dict, official_ids: set) -> dict:
    """
    One candidate video against one game. Reasons are accumulated so a near
    miss is legible rather than just absent.
    """
    snippet = item.get("snippet") or {}
    status = item.get("status") or {}
    published = snippet.get("publishedAt", "")
    try:
        published_at = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ") \
            .replace(tzinfo=timezone.utc)
    except ValueError:
        published_at = None

    game_date = game["game_date"]
    window_ok = bool(published_at) and (
        game_date - PUBLISH_WINDOW_BEFORE <= published_at
        <= game_date + PUBLISH_WINDOW_AFTER)

    failures = []
    if snippet.get("channelId") not in official_ids:
        failures.append("not an official channel")
    if not status.get("embeddable"):
        failures.append("embedding disabled")
    if status.get("privacyStatus") != "public":
        failures.append(f"privacy is {status.get('privacyStatus')}")
    if not window_ok:
        failures.append("published outside the game-date window")
    if not title_names_both_teams(snippet.get("title", ""),
                                  game["opponent_tricode"]):
        failures.append("title does not name both teams")

    return {
        "video_id": item.get("id"),
        "title": snippet.get("title"),
        "channel_title": snippet.get("channelTitle"),
        "channel_id": snippet.get("channelId"),
        "published_at": published,
        "embeddable": bool(status.get("embeddable")),
        "privacy": status.get("privacyStatus"),
        "duration": (item.get("contentDetails") or {}).get("duration"),
        "region_restriction": region_note(item),
        "verdict": "matched" if not failures else "rejected",
        "reasons": "; ".join(failures),
        "watch_url": (f"https://www.youtube.com/watch?v={item.get('id')}"
                      if item.get("id") else ""),
    }


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

def pick_games(index: pd.DataFrame, count=3) -> list:
    """
    One game from each of `count` well-separated seasons, mid-season so the
    sample is not an opening night or a meaningless end-of-year game.

    Deterministic. The earlier phases of this project were twice misled by
    samples that clustered, so the spread is explicit rather than incidental.
    """
    seasons = sorted(index["SEASON"].unique())
    chosen_seasons = [seasons[0],
                      seasons[len(seasons) // 2],
                      seasons[-1]][:count]
    games = []
    for season in chosen_seasons:
        group = (index.loc[index["SEASON"].eq(season)]
                 .sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True))
        row = group.loc[len(group) // 2]
        games.append({
            "season": season,
            "game_id": str(row["GAME_ID"]).zfill(10),
            "game_date": pd.Timestamp(row["GAME_DATE"]).to_pydatetime()
            .replace(tzinfo=timezone.utc),
            "opponent_tricode": row["OPPONENT_ABBREV"],
            "matchup": row["MATCHUP"],
            "is_home": row["IS_HOME"],
        })
    return games


def query_for(game: dict) -> str:
    nickname, city = TEAM_NAMES.get(game["opponent_tricode"],
                                    (game["opponent_tricode"], ""))
    return f"Celtics {nickname} highlights"


def run_probe(key: str, games: list, channels: dict) -> pd.DataFrame:
    official_ids = {c["channel_id"] for c in channels.values()}
    records = []

    for game in games:
        after = game["game_date"] - PUBLISH_WINDOW_BEFORE
        before = game["game_date"] + PUBLISH_WINDOW_AFTER
        query = query_for(game)
        logger.info("%s  %s  %s", game["season"], game["matchup"],
                    game["game_date"].date())

        candidate_ids = []
        for handle, channel in channels.items():
            payload, error = search_channel(key, channel["channel_id"], query,
                                            after, before)
            if error:
                records.append({**game_fields(game), "video_id": None,
                                "verdict": "search_error", "reasons": error,
                                "searched_channel": handle})
                continue
            found = [i["id"]["videoId"] for i in (payload.get("items") or [])
                     if i.get("id", {}).get("videoId")]
            logger.info("    %s: %d candidate(s)", handle, len(found))
            candidate_ids.extend(found)

        if not candidate_ids:
            records.append({**game_fields(game), "video_id": None,
                            "verdict": "no_candidates",
                            "reasons": "no videos on either official channel "
                                       "in the date window",
                            "searched_channel": ""})
            continue

        details = video_details(key, candidate_ids)
        for video_id in candidate_ids:
            item = details.get(video_id)
            if not item:
                continue
            records.append({**game_fields(game),
                            **classify(item, game, official_ids),
                            "searched_channel": ""})

    return pd.DataFrame(records)


def game_fields(game: dict) -> dict:
    return {"season": game["season"], "game_id": game["game_id"],
            "game_date": game["game_date"].date().isoformat(),
            "matchup": game["matchup"],
            "opponent_tricode": game["opponent_tricode"]}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(frame: pd.DataFrame, games: list, channels: dict) -> str:
    lines = [
        "=" * 78,
        "PHASE 12 - OFFICIAL, EMBEDDABLE GAME HIGHLIGHTS (READ ONLY)",
        "=" * 78,
        "",
        "  METADATA ONLY. This probe called the official YouTube Data API and",
        "  read titles, channels, publish dates and the embeddable flag. It",
        "  downloaded no video, scraped no page and cached no footage.",
        "",
        "  Scope: ONE reel per GAME, matched on season, date and opponent, to",
        "  be labelled 'Game highlights'. Never 'Current play'. It is not",
        "  synchronised to the cursor and is not a substitute for one.",
        "",
        "  The synchronised visualisation remains the play-by-play figure",
        "  animation drawn from our own shot coordinates.",
        "",
        "=" * 78,
        "OFFICIAL CHANNELS, RESOLVED THIS RUN",
        "=" * 78,
        "",
    ]
    if not channels:
        lines.append("  NONE RESOLVED. Every result below is therefore a "
                     "rejection, and the run proves nothing.")
    for handle, channel in channels.items():
        lines.append(f"  {handle:<12} {channel['channel_id']}  "
                     f"{channel['title']}")

    matched = frame.loc[frame["verdict"].eq("matched")] if len(frame) \
        else frame
    games_with = matched["game_id"].nunique() if len(matched) else 0

    lines += [
        "",
        "=" * 78,
        "RESULT",
        "=" * 78,
        "",
        f"  games tested            {len(games)}",
        f"  games with a usable reel {games_with} of {len(games)}",
        f"  candidates examined     {len(frame)}",
        "",
    ]

    for game in games:
        rows = frame.loc[frame["game_id"].eq(game["game_id"])] if len(frame) \
            else frame
        hits = rows.loc[rows["verdict"].eq("matched")] if len(rows) else rows
        lines += ["-" * 78,
                  f"  {game['season']}   {game['matchup']}   "
                  f"{game['game_date'].date()}",
                  ""]
        if len(hits):
            for row in hits.itertuples():
                lines += [
                    f"    MATCHED  {row.title}",
                    f"      channel   {row.channel_title}",
                    f"      published {row.published_at}",
                    f"      duration  {row.duration}   "
                    f"embeddable {row.embeddable}",
                    f"      {row.watch_url}",
                ]
                if row.region_restriction:
                    lines.append(f"      NOTE: {row.region_restriction}")
                lines.append("")
        else:
            lines.append("    NO USABLE REEL")
            if len(rows):
                lines.append("    nearest candidates and why each was "
                             "rejected:")
                for row in rows.head(4).itertuples():
                    lines.append(f"      {str(getattr(row, 'title', None))[:62]}")
                    lines.append(f"        {row.reasons}")
            lines.append("")

    lines += [
        "=" * 78,
        "WHAT THIS DOES NOT TELL YOU",
        "=" * 78,
        "",
        "  Three games is a feasibility check, not a coverage figure. If all",
        "  three work, the next question is what share of 636 games work, and",
        "  that is a separate run.",
        "",
        "  `embeddable` is YouTube's own flag and is the right gate, but a",
        "  video can still be pulled, made private or region-locked later. A",
        "  highlight panel must therefore degrade to nothing on failure and",
        "  the dashboard must not depend on it in any way.",
        "",
        "  Nothing was changed. The dashboard does not know this probe exists.",
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

    index = pd.read_csv(index_path)
    games = pick_games(index)
    logger.info("resolving official channels")
    channels = resolve_channels(key)
    if not channels:
        raise SystemExit(
            "Could not resolve either official channel. That usually means "
            "the API key is not enabled for YouTube Data API v3, or the "
            "daily quota is spent. Nothing else was attempted.")

    frame = run_probe(key, games, channels)
    report = build_report(frame, games, channels)
    print(report)

    out = config.REPORTS_DIR / "youtube_probe.txt"
    out.write_text(report + "\n", encoding="utf-8")
    if len(frame):
        frame.to_csv(config.INTERIM_DIR / "youtube_probe.csv", index=False)
    print(f"\nSaved to: {out}")
    print("\nREAD ONLY. No video was downloaded, scraped or cached. Nothing in")
    print("the app, the API, the model or the research outputs was modified.")
    return frame


if __name__ == "__main__":
    main()
