"""
Phase 11 runner: probe whether NBA play clips exist, match, and play.

RE-RUN, AFTER A CORRECTED ENDPOINT
  The first run of this reported 0 clips out of 535 events with 0 errors. That
  was a false negative. It asked `videoevents`, which answers HTTP 200 with a
  playlist that is right about everything except that it contains no URLs.
  Phase 11b (scripts/25_diagnose_video.py) proved the clips exist by printing
  the raw bodies from both endpoints side by side.

  This version asks `videoeventsasset`, records the small, medium and large
  encodings plus thumbnail and captions, and spreads the playback checks evenly
  across seasons instead of taking the first 40 rows, which all came from
  2016-17.

  It OVERWRITES reports/video_probe.txt and data/interim/video_probe.csv. Both
  currently hold the wrong answer, so that is the point.

HOW TO RUN IN SPYDER
  Open this file and press F5 (Run file).

  Needs network. It calls stats.nba.com once per probed event: 24 games (three
  per season) times roughly 24 sampled events each, so about 550 to 620 calls.
  With the same 0.8 second delay every other pull in this project uses, expect
  15 to 25 minutes, longer if calls have to be retried. Leave it running.

  It is resumable only by re-running from the start. There is no cache here,
  because a probe of availability should not be answered from a stale copy.

THIS CHANGES NOTHING
  It writes exactly two files, both new:

    reports/video_probe.txt        the report
    data/interim/video_probe.csv   one row per probed event

  It does not touch the app, the API, the model, the serving data, the
  out-of-fold predictions or any existing report. The dashboard does not know
  it exists. Nothing here is wired into anything.

WHAT IT ANSWERS
  1. Availability   do clips exist, and for which seasons and event types
  2. Matching       when a clip comes back, is it demonstrably THIS play
  3. Playback       do the URLs actually resolve

  Matching is the one that decides the feature. The video endpoint is keyed by
  the play-by-play event number, and Phase 2 established that number is not
  unique within a game. A clip shown against the wrong play would be worse than
  showing no clip at all, so a clip that cannot be verified is counted as a
  MISMATCH and never as coverage.

WHAT TO DO WITH THE OUTPUT
  Paste the printed report back. Nothing gets built until you have read it and
  said so.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import video_probe  # noqa: E402

if __name__ == "__main__":
    video_probe.main()
