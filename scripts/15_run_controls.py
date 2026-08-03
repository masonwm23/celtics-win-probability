"""
Phase 6 runner: controls for the game-constant memorisation artefact.

HOW TO RUN IN SPYDER
  Open this file and press F5 (Run file).

No network. Fits 12 model specifications across the same 8 leave-one-season-out
folds, so it takes several minutes. Longer than 11_train_model.py.

WHAT THIS IS FOR
  Phase 5 reported that every opponent formulation made the model dramatically
  worse. A single pregame column moved Brier 0.1630 -> 0.1998, which is far too
  much damage for one constant-per-game column to do honestly.

  The suspicion is that the opponent feature acts as a game identifier: it takes
  608 distinct values across 636 games, a game averages 486 events that all share
  one label, and min_child_weight is 20, so the tree can isolate a single
  training game into a pure leaf and memorise it.

  The decisive control adds a RANDOM number, constant within each game and
  carrying no information about anything. If that does comparable damage, the
  Phase 5 result is a property of the model settings rather than a statement
  about opponent quality.

  Every prediction is written down in src/controls.py before the run.

Writes reports/phase6_controls.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import run_controls  # noqa: E402

if __name__ == "__main__":
    run_controls.main()
