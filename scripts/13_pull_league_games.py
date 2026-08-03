"""
Phase 5, step 1 runner: pull league-wide game logs.

HOW TO RUN IN SPYDER
  Open this file and press F5 (Run file).

Makes 8 API calls, one per season, and takes about a minute.

Why this is needed: opponent strength must answer "how good was this opponent as
of this date", which needs their FULL schedule. Boston plays most teams only two
to four times a season, so a record built from those games alone would be tiny
and would literally measure "how well did they do against Boston".

Writes data/raw/league_game_logs.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pull_league_games import main  # noqa: E402

if __name__ == "__main__":
    main()
