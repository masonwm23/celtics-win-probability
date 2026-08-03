"""
Phase 4, step 2 runner: audit the saved model and its reproducibility.

HOW TO RUN IN SPYDER
  Open this file and press F5 (Run file).

No network. Confirms the model reloads, that a fresh fit with the same seed
reproduces it exactly, and that the saved feature order is load-bearing.

Writes reports/phase4_validation.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.validate_phase4 import main  # noqa: E402

if __name__ == "__main__":
    main()
