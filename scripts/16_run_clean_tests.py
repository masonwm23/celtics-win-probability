"""
Phase 7 runner: clean re-tests after the Phase 6 artefact.

HOW TO RUN IN SPYDER
  Open this file and press F5 (Run file).

No network. 15 specs across the same 8 folds, so roughly 3 to 4 minutes.

WHAT IT ANSWERS
  1. DOSE-RESPONSE. Damage against feature resolution, for real opponent
     quality and for a meaningless random column, at matched rungs of about 5,
     20, 100 and one-value-per-game. If cardinality is the mechanism, both
     curves rise and they rise together. The shape is predicted in advance, so
     it can fail.

  2. LINEUP, CLEANLY. Phase 4 claimed lineup strength genuinely hurts. Phase 6
     showed most of that penalty is reproducible with noise, and that the
     min_child_weight fix did not work. Here lineup strength is tested at five
     bins cut from TRAINING quantiles only, and as a plain term in a linear
     model that has no splits to memorise with.

  3. WHAT SHIPS. A logistic regression on margin, time, their interaction and
     one opponent number beat tier 3 on Brier, AUC, calibration and seven of
     eight game phases, but has never been bootstrapped against it. The rule is
     fixed in code before the run: switch only if the interval excludes zero in
     the challenger's favour. Same standard that stopped tier 4 shipping.

  4. A NULL CONTROL. A random per-game value added to the linear model. If that
     improves anything, the opponent result cannot be trusted.

Writes reports/phase7_clean_tests.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import run_clean_tests  # noqa: E402

if __name__ == "__main__":
    run_clean_tests.main()
