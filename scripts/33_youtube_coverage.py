"""
Phase 12d runner: which seasons have official, embeddable game reels?

HOW TO RUN IN SPYDER
  Open this file and press F5 (Run file). Same .youtube_api_key.

  24 games, 2 searches each, roughly 4,800 quota units of the 10,000/day free
  tier. A minute or two. It prints the estimate before spending anything.

WHY
  Phase 12c tested three games. The 2020-21 one had an official reel. The
  other two had full game highlights on YouTube, but only on re-upload
  channels (Ximo Pierto, FreeDawkins, Motion Station), which are exactly the
  unauthorised mirrors this project will not use.

  So official coverage is partial and looks era-dependent. This measures it:
  three games per season across all eight, drawn at fixed fractions through
  each schedule so the sample cannot cluster.

THREE OUTCOMES, NOT TWO
  official_reel     an official channel published a matching embeddable reel
  unofficial_only   a matching reel exists but ONLY on re-upload channels
  nothing_found     no variant surfaced a matching reel anywhere

  The middle case is a different fact from no coverage, and it is recorded
  separately rather than folded into a single miss.

IF THE QUOTA RUNS OUT
  It stops cleanly and marks the remaining games `not_tested`, never as
  absences. A spent quota returns 403 for everything, which would otherwise
  make the tail of the run look like a coverage cliff. Results are written to
  data/interim/youtube_coverage.csv as the run proceeds, so a stop loses
  nothing and you can re-run tomorrow.

STILL METADATA ONLY
  No download, no scraping, no re-hosting.

READ ONLY
  Writes reports/youtube_coverage.txt and data/interim/youtube_coverage.csv.

WHAT TO DO
  Paste the output back. The BY SEASON table is the decision.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import youtube_coverage  # noqa: E402

if __name__ == "__main__":
    youtube_coverage.main()
