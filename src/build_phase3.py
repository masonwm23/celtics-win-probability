"""
Phase 3 build: live game-state features from the validated event table.

Reads data/interim/events.parquet and writes data/processed/model_frame.parquet.

Lineup strength is NOT written here. It is an average over games, so it must be
computed inside each cross-validation fold from that fold's training seasons
only. Writing it into a shared file is exactly how that guarantee gets lost.
See src/lineup_strength.py and src/splits.py.
"""

import logging
import time

import pandas as pd

from src import config, features

logger = logging.getLogger(__name__)


def run():
    config.ensure_dirs()
    if not config.EVENTS_PARQUET.exists():
        raise FileNotFoundError(
            f"{config.EVENTS_PARQUET} missing. Run the Phase 2 build first.")

    started = time.time()
    events = pd.read_parquet(config.EVENTS_PARQUET)
    print(f"Loaded {len(events):,} events across {events.game_id.nunique()} games")

    print("Building game-state features...")
    frame = features.build_features(events)
    frame.to_parquet(config.MODEL_FRAME_PARQUET, index=False)

    print(f"  {len(frame):,} rows -> {config.MODEL_FRAME_PARQUET.name}")
    print(f"  feature columns: {len(features.FEATURE_COLUMNS)}")
    for name in features.FEATURE_COLUMNS:
        print(f"    {name}")
    print(f"\nBuild finished in {time.time() - started:.1f}s")
    print("Next: run scripts/10_validate_phase3.py")
    return frame


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    run()


if __name__ == "__main__":
    main()
