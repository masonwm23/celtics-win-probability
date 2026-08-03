"""
Phase 2, step 1 runner: parse raw payloads into tables.

HOW TO RUN IN SPYDER
  Open this file and press F5 (Run file).

No network. Reads data/raw only, which is never modified. Expect a few minutes
for all 636 games. Writes events, rosters and lineups into data/interim.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.build_phase2 import main  # noqa: E402

if __name__ == "__main__":
    main()
