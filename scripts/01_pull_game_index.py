"""
Phase 1, step 2 runner: pull the Celtics game index.

HOW TO RUN IN SPYDER
  Open this file and press F5 (Run file).

Takes well under a minute. Makes 8 API calls, one per season.
Writes data/raw/game_index.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pull_game_index import main  # noqa: E402

if __name__ == "__main__":
    main()
