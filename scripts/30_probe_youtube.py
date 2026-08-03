"""
Phase 12 runner: is there an official, embeddable highlight reel per game?

HOW TO RUN IN SPYDER
  You need a free YouTube Data API key first. See BEFORE YOU RUN below.
  Then open this file and press F5 (Run file).

  About 8 API calls. Seconds.

WHAT IT DOES AND DOES NOT DO
  Reads METADATA ONLY from the official YouTube Data API v3: title, channel,
  publish date, duration, and YouTube's own `embeddable` flag.

  It does NOT download video. It does NOT scrape YouTube pages. It does NOT
  cache or re-host any footage. If a panel is ever built it would use
  YouTube's own iframe player, so the video is served by YouTube under their
  player and their terms.

  Scope is GAME level: one reel per game, matched on season, date and
  opponent, to be labelled "Game highlights". Never "Current play". The
  synchronised visualisation stays the play-by-play figure animation drawn
  from our own shot coordinates.

  Three games from three different seasons, as instructed. Availability only.
  The dashboard is not touched.

BEFORE YOU RUN: get a key (free, a few minutes)
  1. Go to console.cloud.google.com and create or select a project.
  2. APIs & Services -> Library -> search "YouTube Data API v3" -> Enable.
  3. APIs & Services -> Credentials -> Create credentials -> API key.
  4. In a Terminal, from the project root:

         echo 'YOUR_KEY_HERE' > .youtube_api_key

  `.youtube_api_key` is in .gitignore so it will not be committed. Do not
  paste the key into a chat window, into Spyder, or into any source file.

  The free tier is 10,000 units a day. This run uses well under 1,000.

READ ONLY
  Writes reports/youtube_probe.txt and data/interim/youtube_probe.csv.

WHAT TO DO
  Paste the output back. No dashboard change until the numbers are read.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import youtube_probe  # noqa: E402

if __name__ == "__main__":
    youtube_probe.main()
