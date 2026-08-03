"""
Phase 3, step 1 runner: build live game-state features.

HOW TO RUN IN SPYDER
  Open this file and press F5 (Run file).

No network. Reads data/interim/events.parquet, writes
data/processed/model_frame.parquet. Takes well under a minute.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.build_phase3 import main  # noqa: E402

if __name__ == "__main__":
    main()
