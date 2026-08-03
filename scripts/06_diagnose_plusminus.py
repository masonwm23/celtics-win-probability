"""
Phase 1 diagnostic runner: check whether PLUS_MINUS is trustworthy and,
more importantly, whether the WL column (the model's target variable) is correct.

HOW TO RUN IN SPYDER
  Open this file and press F5 (Run file).

Makes 15 API calls, well under a minute: the 5 games with a fractional
PLUS_MINUS, plus 10 randomly chosen clean games as controls.

This script only reads. It changes no data and repairs nothing. Its job is to
tell us what the correct fix is, so we fix the right thing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.diagnose_plusminus import main  # noqa: E402

if __name__ == "__main__":
    main()
