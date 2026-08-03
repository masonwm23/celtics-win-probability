"""
Phase 12b runner: list what the official channels actually posted.

HOW TO RUN IN SPYDER
  Open this file and press F5 (Run file). Uses the same .youtube_api_key.
  About 12 API calls, seconds.

WHY, AND THIS CORRECTS MY OWN WORK
  Phase 12 matched 1 of 3 games. Both failures were rejected with "title does
  not name both teams", and every rejected candidate was a "Top 10 Plays of
  the Night" compilation, not a game reel.

  That means my SEARCH missed the reel, which is a different claim from the
  reel not existing. The query `"Celtics {nickname} highlights"` was invented
  rather than observed, and the NBA's title convention in 2016-17 need not
  match 2020-21.

  So this stops querying. It lists EVERY upload from @NBA and @celtics inside
  each game's date window, ordered by date, up to 50 per channel, and marks
  which ones the Phase 12 rule would accept. Three different answers become
  distinguishable:

    - the reel exists and my query missed it        -> fix the query
    - the reel exists but the title rule rejected it -> fix the rule
    - nothing was posted for that game               -> genuinely unavailable

  It also prints the REGION CODES for anything restricted. The one video that
  matched in Phase 12 was allowed in only 24 countries, and if the United
  States is not among them the panel is blank here regardless of how many
  reels exist. Phase 12 counted those regions without printing them.

STILL METADATA ONLY
  No download, no scraping, no re-hosting. Same API, same rules.

READ ONLY
  Writes reports/youtube_listing.txt and data/interim/youtube_listing.csv.

WHAT TO DO
  Paste the output back.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import youtube_listing  # noqa: E402

if __name__ == "__main__":
    youtube_listing.main()
