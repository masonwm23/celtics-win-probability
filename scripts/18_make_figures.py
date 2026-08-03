"""
Phase 8b runner: build the paper's figures.

HOW TO RUN IN SPYDER
  Open this file and press F5 (Run file).

No network, no model fitting, a few seconds. Every figure is built from a saved
prediction table or score file, so if a result changes the figures change with
it. Nothing here is hand-drawn or hand-typed.

  Figure 1  Reliability, tier 2 against tier 3
  Figure 2  THE SIGNATURE FIGURE. Damage against feature resolution, real
            opponent quality against a meaningless random column
  Figure 3  Brier skill by game phase
  Figure 4  One game's win probability trace, the largest genuine comeback,
            chosen by rule rather than by eye
  Figure 5  Training against out-of-fold Brier, the memorisation signature

If an input is missing the figure is SKIPPED with the script to run, rather than
failing halfway through. Figures 2 and 5 need reports/phase7_scores.csv, which
is written by an updated scripts/16_run_clean_tests.py, so run that again first
if you have not.

Writes figures/fig1..fig5 as PNG at 200 dpi.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import figures  # noqa: E402

if __name__ == "__main__":
    figures.main()
