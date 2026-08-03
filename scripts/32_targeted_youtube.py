"""
Phase 12c runner: search for the title the NBA actually uses.

HOW TO RUN IN SPYDER
  Open this file and press F5 (Run file). Same .youtube_api_key.
  About 15 API calls, seconds.

TWO CORRECTIONS TO MY OWN EARLIER RUNS
  Phase 12 searched "Celtics {nickname} highlights". I made that up. 1 of 3.

  Phase 12b listed channel uploads and I treated that as a complete listing.
  It is not. search.list is a SEARCH INDEX, not an enumeration, and it
  returned ONE upload from @NBA across five days of January 2024 for a channel
  that posts dozens a day. So 12b's "no reel exists for this game" was never a
  finding. It only meant my listing did not surface one.

WHAT 12b DID ESTABLISH, AND THIS USES
  The convention, observed from about thirty uploads in the March 2021 window:

      CELTICS at NETS | FULL GAME HIGHLIGHTS | March 11, 2021
      76ERS at BULLS | FULL GAME HIGHLIGHTS | March 11, 2021

  AWAY at HOME, the phrase, the date. Every field comes from game_index.csv,
  including which side is home, so the expected title is CONSTRUCTED for each
  game instead of guessed. Four variants are tried to cover changes in the
  convention across eight seasons, and the report says which one worked, since
  a 636-game run needs that per era.

  Acceptance is unchanged: official channel, embeddable, public, published in
  the game-date window, title names both teams.

  Region turned out not to be a blocker. Every allowlist seen includes US;
  blocklists are CN and TW only.

STILL METADATA ONLY
  No download, no scraping, no re-hosting.

READ ONLY
  Writes reports/youtube_targeted.txt and data/interim/youtube_targeted.csv.

WHAT TO DO
  Paste the output back.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import youtube_targeted  # noqa: E402

if __name__ == "__main__":
    youtube_targeted.main()
