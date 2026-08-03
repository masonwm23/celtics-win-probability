"""
Phase 4 runner: train and evaluate the four model tiers.

HOW TO RUN IN SPYDER
  Open this file and press F5 (Run file).

No network. Fits 4 tiers across 8 leave-one-season-out folds, then repeats the
whole thing for the sensitivity check, then fits the deliverable model on all
seasons. Expect several minutes.

Every reported metric is OUT OF FOLD. The saved model is fitted on everything,
which is right for deployment and wrong for evaluation; the two are kept apart
deliberately.

Writes reports/phase4_results.txt, data/processed/oof_predictions.parquet,
and models/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train_model import main  # noqa: E402

if __name__ == "__main__":
    main()
