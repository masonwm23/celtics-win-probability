"""
Phase 11b runner: find out why the video probe returned nothing.

HOW TO RUN IN SPYDER
  Open this file and press F5 (Run file).

  About 14 network calls, well under a minute.

WHY THIS EXISTS
  The probe reported 535 events, 0 errors, 0 clips. That is a suspiciously
  clean zero. The same output is produced by "the NBA has no clip for this
  play" and by "we asked the wrong endpoint", and the probe recorded a verdict
  rather than the raw response, so it cannot tell them apart.

  This calls both videoevents and videoeventsasset for the same made field
  goals, plus one plain HTTP request with nba_api taken out of the loop, and
  prints the untouched response bodies.

  nba.com itself uses videoeventsasset for a single play. The probe used
  videoevents. That is the leading suspect, but it is a suspicion until the
  output says so.

READ ONLY
  Writes reports/video_diagnose.txt and nothing else. No app code, no API, no
  model, no serving data, no existing report is touched.

WHAT TO DO
  Paste the output back. The report ends with a HOW TO READ THIS section that
  maps each possible outcome to what it means.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import video_diagnose  # noqa: E402

if __name__ == "__main__":
    video_diagnose.main()
