"""
Phase 11f runner: fetch every matched clip, not a biased sixteen.

HOW TO RUN IN SPYDER
  Open this file and press F5 (Run file).

  About 340 small ranged requests, 5 to 8 minutes. Downloads roughly 700 KB in
  total. Needs data/interim/video_probe.csv from scripts/24_probe_video.py.

WHY, AND THIS IS A CORRECTION TO MY OWN WORK
  Phase 11d concluded the clips are all one placeholder file. It tested sixteen
  URLs. Those sixteen were picked by sorting on (season, game_date,
  event_index) and taking the first two per season, so they were the two
  EARLIEST events of the earliest game in each season. Event ids 1, 2, 4 and 7.
  Opening tips and first possessions.

  If the NBA publishes clips for scoring plays but not for the opening
  sequence, that sample would look exactly as it did and would tell us nothing
  about the other 320 matched events. The conclusion was drawn from a sample
  biased in precisely the direction that matters.

  This fetches all of them.

  The placeholder reference is measured at the start of the run by asking for a
  uuid that cannot exist, rather than hardcoded from an earlier run. If that
  control returns nothing the run aborts instead of guessing, because without a
  reference every clip would be misclassified as real.

READ ONLY
  Writes reports/video_sweep.txt and data/interim/video_sweep.csv. Nothing
  else is touched.

WHAT TO DO
  Paste the output back. If any clips come back real, the report prints their
  URLs so you can open a few in Chrome and see.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import video_sweep  # noqa: E402

if __name__ == "__main__":
    video_sweep.main()
