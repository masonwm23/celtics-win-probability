"""
Phase 15 runner: build the dashboard's pregame prior.

HOW TO RUN
      python scripts/24_build_pregame_prior.py

WHAT IT WRITES
      data/interim/pregame_prior.csv   one row per game: prior + decay constant

WHAT IT DOES NOT TOUCH
  The model, its metadata, the out-of-fold predictions and every figure in the
  paper. This reads them and writes one new file. Re-run
  scripts/20_build_serving.py afterwards so the dashboard picks it up.

  The blend happens at DISPLAY time. The timeline keeps the model's own
  probability; the interface blends the prior in with a weight that decays, and
  says on screen which is which.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import pregame_prior  # noqa: E402

if __name__ == "__main__":
    pregame_prior.main()
