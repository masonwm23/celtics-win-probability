"""
Phase 1, step 3 runner: audit the game index.

HOW TO RUN IN SPYDER
  Open this file and press F5 (Run file).

Makes no network calls. Reads data/raw/game_index.csv and writes
reports/phase1_game_index_validation.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.validate_game_index import main  # noqa: E402

if __name__ == "__main__":
    main()
