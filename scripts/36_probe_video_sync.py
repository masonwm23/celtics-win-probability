"""
Phase 13a runner: can a highlight video be seeked to a specific PLAY?

HOW TO RUN IN SPYDER
  Open this file and press F5 (Run file). Same .youtube_api_key.
  A handful of API calls, seconds. Needs data/interim/highlights_map.json.

  You can run this NOW on the 20 games already mapped. It does not need the
  full mapping, and it answers the feasibility question before you spend five
  days finishing script 35.

THE CORRECTION THIS ANSWERS
  The Phase 12 mapping links one game to one ~9 minute recap. That is NOT
  play-synchronised video, and I should not have let it drift toward sounding
  like progress toward one. Seeking to a play needs a timestamp INSIDE the
  video, and the mapping contains none.

WHY THIS PROBE STARTS WHERE IT DOES
  Under "no downloading, no scraping, embed only" there is exactly one
  legitimate source of in-video timestamps: chapter markers in the video
  description, which the API returns as a plain metadata field.

  Caption tracks can be detected but not read: downloading their content
  requires OAuth as the video's OWNER, and third-party transcript endpoints
  are scraping.

  Everything else (on-screen clock OCR, frame or audio analysis) requires
  obtaining the video, which is out of bounds and is not attempted.

  So the first question is whether chapters exist at all. If they do not, any
  play offset would be inferred from position or duration. That is guessing,
  and the honest answer is then "not possible from this source" rather than a
  matching algorithm with a plausible-looking accuracy number.

WHAT IT REPORTS
  A. Chapter markers across every mapped video.
  B. For one game, each chapter label matched to a play-by-play event on
     player surname plus description overlap. Ties are left UNMATCHED rather
     than broken.
  C. A manual spot-check list of seek URLs.

  Section C exists because this probe can prove a chapter LABEL describes a
  play but CANNOT prove the timestamp is right. Verifying that means watching
  the video at that offset. So NO timing accuracy figure is reported, because
  none can be measured here. Reporting one would repeat the earlier mistake in
  this project of publishing a well-formed number that meant nothing.

WHATEVER THE OUTCOME
  Verified play  -> seek the embedded official video to that offset.
  Unverified     -> keep the synchronised figure animation and show
                    "No verified video for this play."

  The figure animation is not replaced by video under any outcome. It is the
  only layer that covers every play of every one of the 636 games.

READ ONLY
  Writes reports/video_sync_probe.txt and two interim CSVs. Nothing in
  data/serving is touched and the dashboard is unchanged.

WHAT TO DO
  Paste the output back.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import video_sync_probe  # noqa: E402

if __name__ == "__main__":
    # Pass a game_id to probe a specific game, e.g. main("0022301227").
    video_sync_probe.main()
