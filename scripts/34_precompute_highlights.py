"""
Phase 12e runner: build the game_id -> video_id mapping.

HOW TO RUN IN SPYDER
  Open this file and press F5 (Run file). Same .youtube_api_key.

  Roughly 1,000 to 1,500 quota units of the 10,000/day free tier, and a few
  minutes, most of it paging. Compare with ~127,000 units for a per-game
  search: playlistItems.list costs ONE unit per fifty videos, search.list
  costs a hundred per call.

  The raw enumeration is cached to data/interim/youtube_uploads.csv, so a
  second run costs almost nothing. Delete that file to force a fresh pull.

WHAT IT DOES
  1. Resolves the uploads playlist for @NBA and @celtics.
  2. Pages both playlists back past the earliest game in the dataset.
  3. Keeps only titles that read as a game reel and name Boston.
  4. Looks up embeddable and privacy status for those.
  5. Verifies each against each plausible game on TEAMS, DATE and TITLE.
  6. Writes a mapping of CONFIRMED matches only.

  A match is confirmed only when the title names exactly Boston and this
  game's opponent, the date in the title equals the game date (or, absent a
  date, the upload falls in a tight window), the title reads as a game reel
  rather than a player mixtape, the video is public and embeddable on an
  official channel, and no other candidate or game contests it.

  Anything that names the right teams on an official channel but fails
  anything else goes to reports/youtube_review.txt. It is NEVER written to the
  mapping and never displayed. An uncertain match is treated as no match.

  Unofficial channels are not considered anywhere. The candidate pool is built
  from two official uploads playlists, so re-upload channels cannot enter it.

WHAT IT WRITES
  data/interim/highlights_map.json     the mapping, confirmed only
  data/interim/youtube_uploads.csv     cached enumeration
  data/interim/youtube_precompute.csv  every candidate and verdict
  reports/youtube_precompute.txt       coverage by season
  reports/youtube_review.txt           the ambiguity report

  NOTHING in data/serving is touched. The dashboard does not read any of this
  and is unchanged. Moving the mapping into the serving layer happens later,
  only after you have read both reports.

STILL METADATA ONLY
  No download, no scraping, no re-hosting.

WHAT TO DO
  Paste both reports back. The panel is not built until you have seen them.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import youtube_precompute  # noqa: E402

if __name__ == "__main__":
    youtube_precompute.main()
