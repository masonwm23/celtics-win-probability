"""
Phase 2, step 2 runner: independently audit the parsed tables.

HOW TO RUN IN SPYDER
  Open this file and press F5 (Run file).

No network. The check to watch is "Derived on-court minutes match boxscore
minutes": it compares lineup reconstruction against a source that knows nothing
about it, so agreement is real evidence rather than the code agreeing with
itself.

Writes reports/phase2_validation.txt and data/lineup_risk_games.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.validate_phase2 import main  # noqa: E402

if __name__ == "__main__":
    main()
