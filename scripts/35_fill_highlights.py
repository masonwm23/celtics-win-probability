"""
Phase 12f runner: fill the games the uploads playlist could not reach.

HOW TO RUN IN SPYDER
  Open this file and press F5 (Run file). Same .youtube_api_key.

  RUN IT ONCE A DAY UNTIL IT SAYS THE QUEUE IS EMPTY.

  Search is the expensive call, so a day's free quota covers roughly a hundred
  games. With about 616 left that is five or six runs across five or six days.
  Nothing is lost between runs: the mapping and the progress file are written
  after EVERY game, and a re-run skips whatever has already been attempted.

WHY THIS IS NEEDED
  Phase 12e enumerated the official uploads playlists at one quota unit per
  fifty videos, which is a hundred times cheaper than searching. It worked
  exactly as intended and then hit a wall:

      @NBA       18,955 uploads, reaching back only to 2024-03-08 (depth cap)
      @celtics    3,176 uploads, back to 2017-04-01, FULLY enumerated
      matched     20 of the 20 Boston games inside that window

  So the method is sound and simply cannot see further back. Two things follow:

    - Everything before 2024-03-08 was never tested. Its absence from the
      mapping is a limit of the method, not evidence that no reel exists.

    - The Celtics channel's whole history was enumerated and holds no
      full-game reels. That question is closed. Every match comes from @NBA.

ORDER OF WORK
  Seasons Phase 12d found productive are attempted first: 2023-24 back to
  2018-19. The two it found empty, 2017-18 and 2016-17, are attempted last so
  a limited daily quota goes where it is most likely to pay off. They are not
  skipped, because 12d only sampled three games from each.

VERIFICATION IS UNCHANGED
  Teams parsed from the title, date in the title or a tight upload window,
  the title reading as a game reel, and official plus public plus embeddable,
  plus uniqueness. Unofficial channels are filtered out before assessment, so
  they cannot enter the mapping by any route. Uncertain goes to review and is
  never displayed.

IF IT STOPS ON QUOTA
  That is expected and handled. It prints how far it got and exits cleanly.
  Run it again the next day.

STILL METADATA ONLY
  No download, no scraping, no re-hosting. Nothing in data/serving is touched
  and the dashboard is unchanged.

WHAT TO DO
  Paste the output back after each run, or just after the last one. The panel
  gets built once the mapping is as complete as it is going to get.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import youtube_fill  # noqa: E402

if __name__ == "__main__":
    # Set a number here to cap a single run, for example youtube_fill.main(25)
    # if you want to spend only part of the day's quota.
    youtube_fill.main()
