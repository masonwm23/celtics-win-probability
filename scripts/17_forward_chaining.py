"""
Phase 8a runner: forward chaining versus leave-one-season-out.

HOW TO RUN IN SPYDER
  Open this file and press F5 (Run file).

No network. Two split designs, three specifications, roughly a minute.

WHY IT MATTERS
  Every result so far uses leave-one-season-out, which lets a fold train on
  2023-24 to predict 2016-17. That measures how much team-specific structure
  exists in the data. It is not how a live model would be used, because on any
  given night you only have the past.

  The paper currently lists this as a limitation. Running it converts a
  statement into a measurement, which is the difference between a caveat and a
  result, and it is the kind of thing a reader notices.

  Both designs are scored on exactly the SAME held-out seasons, so the
  comparison is of split designs rather than of different test sets.

  One confound is reported rather than hidden: forward chaining also trains on
  less data, three to seven seasons instead of a constant seven, so a gap cannot
  be attributed to the direction of time alone.

Writes reports/phase8_forward_chaining.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import forward_chaining  # noqa: E402

if __name__ == "__main__":
    forward_chaining.main()
