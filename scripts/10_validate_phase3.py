"""
Phase 3, step 2 runner: audit the features and the split design.

HOW TO RUN IN SPYDER
  Open this file and press F5 (Run file).

No network. The check to watch is the SHUFFLED TARGET LEAK TEST. It permutes
which outcome each game carries and refits. If the model still predicts well on
scrambled labels, a feature is carrying the answer and every later number in this
project is worthless.

Takes a few minutes: it fits a model twice per season fold.

Writes reports/phase3_validation.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.validate_phase3 import main  # noqa: E402

if __name__ == "__main__":
    main()
