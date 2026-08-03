"""
Phase 2 build: turn cached raw payloads into parsed tables.

Reads only from data/raw, which is never modified. Writes:

    data/interim/events.parquet          one row per play-by-play event
    data/interim/rosters.parquet         one row per player per team per game
    data/interim/lineups.parquet         both teams' on-court five at every event
    data/interim/lineup_anomalies.csv    every incoherent substitution
    data/interim/derived_minutes.csv     derived vs boxscore minutes per player

This module BUILDS. It does not vouch for the result. src/validate_phase2.py is
the independent audit, kept separate on purpose.
"""

import logging
import time

import pandas as pd

from src import config, lineups, parse_events, rosters

logger = logging.getLogger(__name__)


def run(game_ids=None):
    config.ensure_dirs()
    started = time.time()

    print("Parsing events...")
    events = parse_events.build_events(game_ids)
    events.to_parquet(config.EVENTS_PARQUET, index=False)
    print(f"  {len(events):,} events across {events.game_id.nunique()} games "
          f"-> {config.EVENTS_PARQUET.name}")

    print("Building rosters...")
    roster = rosters.build_rosters(game_ids)
    roster.to_parquet(config.ROSTERS_PARQUET, index=False)
    print(f"  {len(roster):,} player-game rows -> {config.ROSTERS_PARQUET.name}")

    print("Reconstructing lineups...")
    lineup_df, anomalies, minutes, methods = lineups.build_lineups(game_ids)
    # Lineups are tuples of ints; parquet needs a stable representation.
    for column in ("home_lineup", "away_lineup"):
        lineup_df[column] = lineup_df[column].apply(
            lambda ids: ",".join(str(i) for i in ids))
    lineup_df.to_parquet(config.LINEUPS_PARQUET, index=False)
    anomalies.to_csv(config.LINEUP_ANOMALIES_CSV, index=False)
    minutes.to_csv(config.DERIVED_MINUTES_CSV, index=False)
    print(f"  {len(lineup_df):,} event lineups -> {config.LINEUPS_PARQUET.name}")
    print(f"  {len(anomalies):,} anomalies -> {config.LINEUP_ANOMALIES_CSV.name}")
    print(f"  {len(minutes):,} minute comparisons -> "
          f"{config.DERIVED_MINUTES_CSV.name}")
    print(f"  substitution resolution methods: {methods}")

    print(f"\nBuild finished in {time.time() - started:.1f}s")
    print("Next: run scripts/08_validate_phase2.py")
    return events, roster, lineup_df, anomalies, minutes, methods


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    run()


if __name__ == "__main__":
    main()
