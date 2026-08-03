"""
Add the comeback fields to data/serving/index.json.

WHAT A COMEBACK IS HERE
-----------------------
The largest points deficit Boston faced at any point in a game they went on to
win. Nothing about it is modelled: `margin` in each game payload is
`celtics_score - opponent_score` at every event, both of which come from the
play-by-play, so the deficit is arithmetic on the scoreboard.

    largest_deficit = max(0, -min(margin))

Losses get the same number computed, because a deficit is a deficit, but the
dashboard only ranks wins. Trailing by 20 and losing is not a comeback, it is
just a loss, and putting the two in one list would be the most misleading
possible ordering.

WHY THIS IS A SEPARATE SCRIPT
-----------------------------
`src/build_serving.py` now writes these fields itself, so a full rebuild
produces them. This script exists so you do not have to run a full rebuild to
get them once: it reads the 636 game payloads that are already on disk and
rewrites only index.json. It changes no other file and computes no new
probability.

Run either one. They produce the same numbers, and this script checks that the
fields it writes agree with the payload it read them from.

HOW TO RUN
----------
    python scripts/42_comeback_index.py

OUTPUT
------
    data/serving/index.json      rewritten in place, with four fields added
                                 per game and every existing field preserved
    reports/comebacks.txt        the ranked list, by season, for the record
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("comebacks")

INDEX_PATH = config.SERVING_DIR / "index.json"
GAMES_DIR = config.SERVING_DIR / "games"
REPORT_PATH = config.REPORTS_DIR / "comebacks.txt"

ADDED_FIELDS = (
    "largest_deficit",
    "deficit_period",
    "deficit_clock",
    "deficit_event",
)


def comeback_fields(events: dict) -> dict:
    """
    The worst deficit in one game, and the moment it was reached.

    The moment is the FIRST event at the low point. A team can sit at -18 for
    six possessions; the interesting one is when they got there.
    """
    margin = list(events["margin"])
    if not margin:
        raise ValueError("empty margin series")

    worst = min(margin)
    deficit = max(0, -int(worst))
    if deficit == 0:
        # Never trailed. Real, and it happens: a wire-to-wire win has no
        # comeback in it, and it is not the same thing as missing data.
        return {
            "largest_deficit": 0,
            "deficit_period": None,
            "deficit_clock": None,
            "deficit_event": None,
        }

    at = margin.index(worst)
    return {
        "largest_deficit": deficit,
        "deficit_period": int(events["period"][at]),
        "deficit_clock": str(events["clock"][at]),
        "deficit_event": int(at),
    }


def main() -> None:
    if not INDEX_PATH.exists():
        raise SystemExit(
            f"{INDEX_PATH} does not exist. Run scripts/20_build_serving.py first."
        )

    index = json.loads(INDEX_PATH.read_text())
    games = index["games"]
    logger.info("index carries %d games", len(games))

    missing: list[str] = []
    for row in games:
        payload_path = GAMES_DIR / f"{row['game_id']}.json"
        if not payload_path.exists():
            missing.append(row["game_id"])
            continue
        payload = json.loads(payload_path.read_text())
        events = payload["events"]

        fields = comeback_fields(events)

        # Cross-check against the payload's own final score rather than
        # trusting the index row. If these ever disagree the index is stale and
        # ranking off it would be wrong.
        meta = payload["meta"]
        if meta["celtics_final"] != row["celtics_final"] or \
           meta["opponent_final"] != row["opponent_final"]:
            raise SystemExit(
                f"{row['game_id']}: index and payload disagree on the final "
                f"score. Rebuild with scripts/20_build_serving.py."
            )
        final_margin = events["margin"][-1]
        if (final_margin > 0) != bool(meta["celtics_won"]):
            raise SystemExit(
                f"{row['game_id']}: final margin {final_margin} does not match "
                f"celtics_won={meta['celtics_won']}."
            )

        row.update(fields)

    if missing:
        raise SystemExit(
            f"{len(missing)} games in the index have no payload on disk, "
            f"first: {missing[0]}. Rebuild before ranking anything."
        )

    INDEX_PATH.write_text(json.dumps(index, indent=1))
    logger.info("wrote %s", INDEX_PATH)

    # ---------------------------------------------------------------- report
    wins = [g for g in games if g["celtics_won"] and g["largest_deficit"] > 0]
    wins.sort(key=lambda g: (-g["largest_deficit"], g["lowest_wp"]))

    lines = [
        "Largest deficits erased in Celtics wins, 2016-17 to 2023-24",
        "",
        "Deficit is max(0, -min(margin)) over the play-by-play score. "
        "Lowest WP is",
        "the out-of-fold probability, shown for context and not used to rank.",
        "",
    ]
    lines.append(f"{len(wins)} of {sum(1 for g in games if g['celtics_won'])} "
                 f"wins involved trailing at some point.")
    lines.append("")
    lines.append(f"{'rank':>4}  {'date':<10}  {'matchup':<14}  {'down':>4}  "
                 f"{'when':<10}  {'final':>8}  {'low wp':>7}")
    for rank, game in enumerate(wins[:60], start=1):
        when = (f"Q{game['deficit_period']} {game['deficit_clock']}"
                if game["deficit_period"] else "")
        lines.append(
            f"{rank:>4}  {game['date']:<10}  {game['matchup']:<14}  "
            f"{game['largest_deficit']:>4}  {when:<10}  "
            f"{game['celtics_final']:>3}-{game['opponent_final']:<3}   "
            f"{game['lowest_wp']:>7.4f}"
        )

    lines.append("")
    lines.append("By season, the largest:")
    for season in index["seasons"]:
        best = [g for g in wins if g["season"] == season]
        if not best:
            lines.append(f"  {season}   no win involved trailing")
            continue
        top = best[0]
        lines.append(
            f"  {season}   down {top['largest_deficit']} "
            f"({top['matchup']}, {top['date']})"
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    logger.info("wrote %s", REPORT_PATH)
    logger.info("")
    for line in lines[:18]:
        logger.info(line)


if __name__ == "__main__":
    main()
