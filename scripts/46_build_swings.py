"""
Build the biggest-swing list the dashboard's Swings view reads.

WHAT A SWING IS
---------------
The single largest one-event increase in Boston's OUT-OF-FOLD win probability in
a game, kept only when the event that caused it is a Boston made shot (an
opponent turnover has no Boston play to show or to clip). Ranked across every
game. Out of fold matters here exactly as it does for the comeback list: each
probability came from a model that never saw that game's season, so the jump is
a real forecast moving, not a model reciting an outcome.

THE CLIPS
---------
scripts/45_probe_swing_clips.py searched the official NBA and Celtics channels
for a verified single-play clip of each top swing and wrote every candidate,
with its verdict, to data/interim/swing_clip_candidates.csv. This step attaches
ONLY the rows that probe confirmed (verdict == MATCH): official channel,
embeddable, public, posted right after that game, names the player and the
game-winning moment, short, not a compilation. No clip is attached that the
probe did not confirm, and nothing is constructed here.

OUTPUT
    data/serving/swings.json          the ranked list, with clips where confirmed
    web/public/data/swings.json       the copy the static app loads
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

TOP_N = 30
WEB_DATA_DIR = config.PROJECT_ROOT / "web" / "public" / "data"


def load_confirmed_clips() -> dict:
    """(date, matchup) -> clip dict, for probe rows with verdict == MATCH."""
    path = config.INTERIM_DIR / "swing_clip_candidates.csv"
    clips = {}
    if not path.exists():
        return clips
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("verdict") != "MATCH":
                continue
            key = (row["swing_date"], row["swing_matchup"])
            # First confirmed clip wins; the probe already ordered them so the
            # best single-play match is first for a given swing.
            clips.setdefault(key, {
                "video_id": row["video_id"],
                "title": row["title"],
                "url": row["watch_url"],
                "channel": row.get("channel_title", ""),
                "duration_sec": int(row["duration_sec"]) if row.get(
                    "duration_sec") else None,
            })
    return clips


def biggest_swing(game: dict):
    """Return (event_index, delta) of the largest Boston-made-shot WP jump."""
    e = game["events"]
    wp = e["wp"]
    best, at = 0.0, -1
    for i in range(1, len(wp)):
        delta = wp[i] - wp[i - 1]
        if delta <= best:
            continue
        if e["team"][i] != "BOS":
            continue
        made = (e["shot_result"][i] == "Made") or \
            (" pts)" in (e["description"][i] or "").lower())
        if not made:
            continue
        best, at = delta, i
    return at, best


def main():
    clips = load_confirmed_clips()
    games_dir = config.SERVING_DIR / "games"

    swings = []
    for path in sorted(games_dir.glob("*.json")):
        game = json.loads(path.read_text())
        at, delta = biggest_swing(game)
        if at < 0:
            continue
        e, meta = game["events"], game["meta"]
        pid = str(e["person_id"][at])
        clip = clips.get((meta["date"], meta["matchup"]))
        swings.append({
            "game_id": meta["game_id"],
            "event_index": int(e["event_index"][at]),
            "season": meta["season"],
            "date": meta["date"],
            "matchup": meta["matchup"],
            "opponent": meta["opponent"],
            "opponent_logo": meta.get("opponent_logo"),
            "celtics_is_home": meta["celtics_is_home"],
            "celtics_won": meta["celtics_won"],
            "celtics_final": meta["celtics_final"],
            "opponent_final": meta["opponent_final"],
            "periods": meta["periods"],
            "player": game["players"].get(pid, {}).get("name", ""),
            "description": e["description"][at],
            "period": int(e["period"][at]),
            "clock": e["clock"][at],
            "celtics_score": int(e["celtics_score"][at]),
            "opponent_score": int(e["opponent_score"][at]),
            "wp_before": round(float(game["events"]["wp"][at - 1]), 5),
            "wp_after": round(float(game["events"]["wp"][at]), 5),
            "delta": round(delta, 5),
            "clip": clip,
        })

    swings.sort(key=lambda s: s["delta"], reverse=True)
    top = swings[:TOP_N]

    payload = {
        "swings": top,
        "count": len(top),
        "with_clip": sum(1 for s in top if s["clip"]),
        "probability_source": (
            "out-of-fold tier3_celtics: the jump is a forecast from a model that "
            "never saw this game's season"),
        "note": (
            "Biggest single-play increase in Boston's out-of-fold win "
            "probability per game, Boston made shots only, ranked across all "
            "games. Clips are official single-play videos the probe confirmed; "
            "everything else opens in the play reconstruction."),
    }

    for out_dir in (config.SERVING_DIR, WEB_DATA_DIR):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "swings.json").write_text(
            json.dumps(payload, separators=(",", ":"), ensure_ascii=False))

    print(f"biggest swings: {len(swings)} games, top {len(top)} kept")
    print(f"confirmed clips attached: {payload['with_clip']}")
    print("top 8:")
    for i, s in enumerate(top[:8], 1):
        tag = "[clip]" if s["clip"] else "      "
        print(f"  {i:>2} +{s['delta']*100:4.0f}pp {tag} {s['date']} "
              f"{s['matchup']:<13} {s['player']:<18} "
              f"{s['wp_before']*100:4.1f}->{s['wp_after']*100:4.1f}")
    print(f"\nwrote {config.SERVING_DIR / 'swings.json'}")
    print(f"wrote {WEB_DATA_DIR / 'swings.json'}")


if __name__ == "__main__":
    main()
