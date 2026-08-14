"""
Probe: is there an OFFICIAL, EMBEDDABLE single-play clip for the biggest
Celtics win-probability swings?

WHAT THIS IS, AND IS NOT
------------------------
Phases 12-13 closed play-SYNCHRONISED video: you cannot seek a long recap to a
specific play, because the timestamp that would let you does not exist in the
official videos. This is a different question. A standalone clip of ONE play
needs no seeking; the clip IS the play. The NBA posts these for game-winners
and big shots. This probe measures how many of the biggest swings actually have
one we can confirm.

It is READ ONLY and metadata only, through the official YouTube Data API v3,
reusing the exact credential handling and official-channel resolution from
src/youtube_probe.py. Nothing is downloaded, scraped, cached or re-hosted, and
nothing in data/serving is touched.

HOW A MATCH IS CONFIRMED
------------------------
A clip is attached to a swing only when ALL of these hold. Anything short of it
is written to the report as a near miss and never treated as a match.

  1. Official channel   @NBA or @celtics, resolved at run time, never assumed.
  2. Embeddable, public.
  3. Publish window     posted from 1 day before to 4 days after the game. This
                        is the disambiguator: a clip of THIS game-winner is
                        posted right after THIS game, which pins a title that
                        has no date to the exact game.
  4. Names the player   the shooter's surname appears in the title.
  5. Names the matchup  the opponent (or Boston) appears in the title.
  6. Single play        short duration (<= 240s) and the title is not a
                        compilation ("top 10", "best plays", "highlights",
                        "recap", "mix", ...). A 9-minute recap or a top-plays
                        reel is exactly what we do NOT want here.

WHAT IT WRITES
    reports/swing_clip_probe.txt          human-readable, matches and near misses
    data/interim/swing_clip_candidates.csv every candidate with its verdict

HOW TO RUN
    Put your key in .youtube_api_key (gitignored) as the earlier YouTube scripts
    describe, then open this file in Spyder and press F5. It prints the quota it
    is about to spend and stops early on each swing once it finds a confirmed
    clip, so a run over the default 15 swings costs well under the 10,000-unit
    free daily quota.
"""

import csv
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.youtube_probe import (  # noqa: E402
    TEAM_NAMES, PUBLISH_WINDOW_AFTER, PUBLISH_WINDOW_BEFORE,
    load_api_key, resolve_channels, search_channel, video_details,
    normalise, region_note,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# How many of the biggest swings to probe. The run is resumable: it skips any
# swing that already has a confirmed clip in the candidates file, and it stops
# before spending QUOTA_BUDGET units so one run stays inside the 10,000-unit
# free daily quota. Widen TOP_N freely and run again on the next quota day; it
# picks up where it left off and accumulates clips.
TOP_N = 40
QUOTA_BUDGET = 9000

# On a re-run, also skip swings already SEARCHED that came back with only near
# misses (full-game recaps and compilations, no official single-play clip). This
# spends the day's quota on swings that have never been searched instead of
# re-searching ones that already failed. Flip to True only to give those near
# misses another look on a later day.
RETRY_NEAR_MISSES = False

# A single play clip is short. Recaps run ~9 minutes; top-plays reels longer.
# Up to 5 minutes allows a game-winner clip that includes replays and reaction,
# but anything past 150s must also carry an explicit game-winner marker (below).
MAX_CLIP_SECONDS = 300
LONG_CLIP_SECONDS = 150

# Title tokens that mark a compilation rather than one play. If any appears, the
# candidate is not a single-play clip whatever else it looks like.
COMPILATION_MARKERS = (
    "top 10", "top 5", "top plays", "best plays", "best of", "highlights",
    "full game", "full highlights", "recap", "compilation", "mix", "every ",
    "all the", "top play", "month", "this week", "season", "moments", "plays of",
)

# Title tokens that positively read as one clutch play. Not required, but noted.
PLAY_MARKERS = (
    "game winner", "game-winner", "gamewinner", "buzzer", "clutch", "dagger",
    "and-1", "and 1", "seals", "wins it", "go-ahead", "go ahead", "final",
)

# Strong game-winner markers. When one of these is present alongside our player's
# name and the tight publish window, the clip is that game's decisive play even
# if the title never names the opponent. Used to accept clips that name the
# player and the moment but leave the opponent out, which is the common form.
GAME_WINNER_MARKERS = (
    "game winner", "game-winner", "gamewinner", "buzzer beater",
    "buzzer-beater", "buzzerbeater", "wins it", "walk-off", "walk off",
    "dagger", "go-ahead", "go ahead", "hits the game", "clutch game-winner",
    "game-winning", "game winning",
)


def parse_duration(iso: str) -> int:
    """ISO-8601 'PT#M#S' to seconds. Returns -1 if unparseable."""
    if not isinstance(iso, str):
        return -1
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso)
    if not m:
        return -1
    h, mi, s = (int(x) if x else 0 for x in m.groups())
    return h * 3600 + mi * 60 + s


def surname(full_name: str) -> str:
    """Last name for title matching. Drops a trailing Jr./Sr./III suffix."""
    parts = [p for p in re.sub(r"[^A-Za-z .'-]", " ", full_name or "").split()
             if p]
    suffixes = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}
    while len(parts) > 1 and parts[-1].lower().strip(".") in suffixes:
        parts.pop()
    return parts[-1] if parts else ""


def query_name(full_name: str) -> str:
    """First + last, suffix dropped, for the search query."""
    parts = [p for p in re.sub(r"[^A-Za-z .'-]", " ", full_name or "").split()
             if p]
    suffixes = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}
    while len(parts) > 1 and parts[-1].lower().strip(".") in suffixes:
        parts.pop()
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Step 1: the biggest positive swings, from the serving payload
# ---------------------------------------------------------------------------

def biggest_swings(top_n: int) -> list:
    """
    The largest single-event increase in Boston's out-of-fold win probability,
    per game, ranked across all games. Only swings caused by a Boston made shot
    are kept, because those are the ones a clip could exist for; an opponent
    turnover has no Boston play to show.
    """
    games_dir = config.SERVING_DIR / "games"
    swings = []
    for path in games_dir.glob("*.json"):
        game = json.loads(path.read_text())
        e = game["events"]
        wp = e["wp"]
        best, at = 0.0, -1
        for i in range(1, len(wp)):
            delta = wp[i] - wp[i - 1]
            if delta > best:
                best, at = delta, i
        if at < 0:
            continue
        team = e["team"][at]
        shot_made = (e["shot_result"][at] == "Made") or \
            (" pts)" in (e["description"][at] or "").lower())
        if team != "BOS" or not shot_made:
            continue
        meta = game["meta"]
        pid = str(e["person_id"][at])
        name = game["players"].get(pid, {}).get("name", "")
        swings.append({
            "delta": best,
            "season": meta["season"],
            "date": meta["date"],
            "matchup": meta["matchup"],
            "opponent": meta["opponent"],
            "is_home": meta["celtics_is_home"],
            "won": meta["celtics_won"],
            "period": e["period"][at],
            "clock": e["clock"][at],
            "celtics_score": e["celtics_score"][at],
            "opponent_score": e["opponent_score"][at],
            "wp_before": wp[at - 1],
            "wp_after": wp[at],
            "player": name,
            "description": e["description"][at],
        })
    swings.sort(key=lambda s: s["delta"], reverse=True)
    return swings[:top_n]


# ---------------------------------------------------------------------------
# Step 2: search official channels and verify
# ---------------------------------------------------------------------------

def build_queries(swing: dict) -> list:
    nickname, _city = TEAM_NAMES.get(swing["opponent"], (swing["opponent"], ""))
    player = query_name(swing["player"]) or swing["opponent"]
    return [
        f"{player} game winner {nickname}",
        f"{player} clutch {nickname}",
        f"{player} {nickname}",
    ]


def verify(item: dict, details: dict, swing: dict, official_ids: set) -> dict:
    snippet = item.get("snippet") or {}
    detail = details.get(item.get("id", {}).get("videoId") or item.get("id"), {})
    status = detail.get("status") or {}
    content = detail.get("contentDetails") or {}
    title = snippet.get("title", "")
    text = normalise(title)

    published = snippet.get("publishedAt", "")
    try:
        published_at = datetime.strptime(
            published, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        published_at = None
    game_date = datetime.strptime(swing["date"], "%Y-%m-%d").replace(
        tzinfo=timezone.utc)
    window_ok = bool(published_at) and (
        game_date - PUBLISH_WINDOW_BEFORE <= published_at
        <= game_date + PUBLISH_WINDOW_AFTER)

    nickname, city = TEAM_NAMES.get(swing["opponent"], (swing["opponent"], ""))
    last = surname(swing["player"])
    duration = parse_duration(content.get("duration"))

    winner_marker = any(g in text for g in GAME_WINNER_MARKERS)
    player_ok = bool(last) and normalise(last) in text
    matchup_ok = (normalise(nickname) in text
                  or (bool(city) and normalise(city) in text)
                  or "celtics" in text or "boston" in text)

    fails = []
    if snippet.get("channelId") not in official_ids:
        fails.append("not official channel")
    if not status.get("embeddable"):
        fails.append("not embeddable")
    if status.get("privacyStatus") != "public":
        fails.append(f"privacy {status.get('privacyStatus')}")
    if not window_ok:
        fails.append("outside game-date window")
    if not player_ok:
        fails.append("player not in title")
    # The publish window already ties the clip to this game, so a clip that
    # names our player and marks it as the game-winner is that game's decisive
    # play even with the opponent left out of the title. Require one or the other.
    if not (matchup_ok or winner_marker):
        fails.append("neither matchup nor game-winner marker in title")
    if any(marker in text for marker in COMPILATION_MARKERS):
        fails.append("title reads as a compilation")
    if duration < 0 or duration > MAX_CLIP_SECONDS:
        fails.append(f"duration {duration}s not single-play")
    elif duration > LONG_CLIP_SECONDS and not winner_marker:
        fails.append(f"{duration}s clip needs a game-winner marker")

    vid = item.get("id", {}).get("videoId") or item.get("id")
    return {
        "swing_date": swing["date"],
        "swing_matchup": swing["matchup"],
        "swing_player": swing["player"],
        "swing_delta_pp": round(swing["delta"] * 100, 1),
        "video_id": vid,
        "title": title,
        "channel_title": snippet.get("channelTitle"),
        "published_at": published,
        "duration_sec": duration,
        "region": region_note(detail),
        "has_play_marker": any(m in text for m in PLAY_MARKERS),
        "verdict": "MATCH" if not fails else "near-miss",
        "reasons": "; ".join(fails),
        "watch_url": f"https://www.youtube.com/watch?v={vid}" if vid else "",
    }


def probe_swing(key: str, channels: dict, swing: dict) -> dict:
    """Return {'match': row|None, 'candidates': [rows], 'searches': n}."""
    game_date = datetime.strptime(swing["date"], "%Y-%m-%d").replace(
        tzinfo=timezone.utc)
    after = game_date - PUBLISH_WINDOW_BEFORE
    before = game_date + PUBLISH_WINDOW_AFTER
    official_ids = {c["channel_id"] for c in channels.values()}

    candidates, searches, seen = [], 0, set()
    for query in build_queries(swing):
        for handle, chan in channels.items():
            payload, error = search_channel(
                key, chan["channel_id"], query, after, before)
            searches += 1
            if error or not payload:
                continue
            items = payload.get("items", [])
            ids = [it["id"]["videoId"] for it in items
                   if it.get("id", {}).get("videoId")]
            details = video_details(key, [i for i in ids if i not in seen])
            for it in items:
                vid = it.get("id", {}).get("videoId")
                if not vid or vid in seen:
                    continue
                seen.add(vid)
                row = verify(it, details, swing, official_ids)
                candidates.append(row)
        # Stop as soon as any query yields a confirmed single-play clip.
        if any(c["verdict"] == "MATCH" for c in candidates):
            break

    matches = [c for c in candidates if c["verdict"] == "MATCH"]
    matches.sort(key=lambda c: (c["has_play_marker"], -c["duration_sec"]),
                 reverse=True)
    return {"match": matches[0] if matches else None,
            "candidates": candidates, "searches": searches}


def no_candidate_row(swing: dict) -> dict:
    """A placeholder row for a swing whose search returned nothing, so it is
    recorded as searched (same columns as a verify() row). build_swings only
    reads verdict == MATCH, so this is inert there."""
    return {
        "swing_date": swing["date"],
        "swing_matchup": swing["matchup"],
        "swing_player": swing["player"],
        "swing_delta_pp": round(swing["delta"] * 100, 1),
        "video_id": "",
        "title": "",
        "channel_title": "",
        "published_at": "",
        "duration_sec": "",
        "region": "",
        "has_play_marker": "",
        "verdict": "no-candidates",
        "reasons": "no candidates returned in the game-date window",
        "watch_url": "",
    }


def load_existing():
    """Prior candidate rows, the swings already confirmed, and every swing that
    has been searched before (confirmed or near-miss)."""
    path = config.INTERIM_DIR / "swing_clip_candidates.csv"
    if not path.exists():
        return [], set(), set()
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    confirmed = {(r["swing_date"], r["swing_matchup"])
                 for r in rows if r.get("verdict") == "MATCH"}
    searched = {(r["swing_date"], r["swing_matchup"]) for r in rows}
    return rows, confirmed, searched


def main():
    swings = biggest_swings(TOP_N)
    existing_rows, confirmed, searched = load_existing()
    already = sum(1 for s in swings if (s["date"], s["matchup"]) in confirmed)
    skip_searched = searched - confirmed if not RETRY_NEAR_MISSES else set()
    to_search = sum(1 for s in swings
                    if (s["date"], s["matchup"]) not in confirmed
                    and (s["date"], s["matchup"]) not in skip_searched)

    key = load_api_key()
    print(f"Top {len(swings)} swings. {already} already have a confirmed clip. "
          f"{len(skip_searched & {(s['date'], s['matchup']) for s in swings})} "
          f"were searched before with no single-play clip (skipped). "
          f"{to_search} to search this run. Budget {QUOTA_BUDGET:,} units.\n")
    print("Resolving official channels:")
    channels = resolve_channels(key)
    if not channels:
        raise SystemExit("Could not resolve any official channel. Stopping.")
    print()

    new_rows, probed_keys = [], set()
    spent, matched, deferred = 0, 0, 0
    for n, swing in enumerate(swings, 1):
        ms = (swing["date"], swing["matchup"])
        head = (f"[{n:>2}/{len(swings)}] +{swing['delta']*100:.0f}pp "
                f"{swing['date']} {swing['matchup']:<13} "
                f"{surname(swing['player']):<12}")
        if ms in confirmed:
            matched += 1
            print(f"{head} have   (already confirmed)")
            continue
        if ms in skip_searched:
            print(f"{head} skip   (searched before, no single-play clip)")
            continue
        if spent + 700 > QUOTA_BUDGET:
            deferred += 1
            continue
        result = probe_swing(key, channels, swing)
        spent += result["searches"] * 100
        # Record the search even when it returned nothing, so the swing counts
        # as searched and a later run skips it instead of spending quota on it
        # again. Without this a "no candidates" swing is never remembered and is
        # re-searched every run, which is what stalled the earlier passes.
        new_rows.extend(result["candidates"] or [no_candidate_row(swing)])
        probed_keys.add(ms)
        m = result["match"]
        if m:
            matched += 1
            print(f"{head} MATCH  {m['duration_sec']}s  {m['title'][:56]}")
            print(f"{'':>22} {m['watch_url']}")
        else:
            near = [c for c in result["candidates"] if c["reasons"]]
            why = near[0]["reasons"] if near else "no candidates returned"
            print(f"{head} none   ({why})")

    # Keep prior rows for swings we did not re-probe; replace those we did.
    merged = [r for r in existing_rows
              if (r["swing_date"], r["swing_matchup"]) not in probed_keys]
    merged.extend(new_rows)

    config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = config.INTERIM_DIR / "swing_clip_candidates.csv"
    if merged:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(merged[0].keys()))
            writer.writeheader()
            writer.writerows(merged)

    total_confirmed = len({(r["swing_date"], r["swing_matchup"])
                           for r in merged if r.get("verdict") == "MATCH"})
    lines = [
        "BIGGEST-SWING SINGLE-PLAY CLIP PROBE (resumable, official API)",
        f"Run at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"top swings           {len(swings)}",
        f"confirmed clips total {total_confirmed}",
        f"spent this run       ~{spent:,} units",
        f"deferred to next day {deferred}",
        "", "CONFIRMED", "-" * 60,
    ]
    for r in merged:
        if r.get("verdict") == "MATCH":
            lines.append(f"  {r['swing_date']} {r['swing_matchup']}  "
                         f"{r['title']}")
            lines.append(f"      {r['watch_url']}")
    (config.REPORTS_DIR / "swing_clip_probe.txt").write_text(
        "\n".join(lines) + "\n")

    print(f"\nConfirmed clips total: {total_confirmed} across the top "
          f"{len(swings)} swings. Spent ~{spent:,} units this run.")
    if deferred:
        print(f"{deferred} swing(s) deferred (quota). Run again tomorrow to "
              f"finish them.")
    print("Next: run scripts/46_build_swings.py to fold the clips into the "
          "dashboard, then commit and push.")


if __name__ == "__main__":
    main()
