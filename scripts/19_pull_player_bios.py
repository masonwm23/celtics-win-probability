"""
Phase 9a runner: pull player biographical data for the dashboard.

HOW TO RUN IN SPYDER
  Open this file and press F5 (Run file).

NETWORK. 30 teams times 8 seasons = 240 calls to CommonTeamRoster, roughly four
minutes. RESUMABLE: each team-season is cached to its own CSV, so an interrupted
run picks up where it stopped.

WHAT IT IS FOR
  The dashboard's player cards need a position, a height and a headshot. The
  pipeline has none of the three:

    - the boxscore `position` field is the coarse G / F / C bucket, and before
      2017-18 it is populated for nine or ten players per team rather than
      five. Phase 1 flagged granular positions as needing a separate source.
      This is that source.
    - height appears in no play-by-play or boxscore endpoint.
    - headshots are keyed by personId on the NBA CDN.

WHY NOT THE CHEAPER ENDPOINT
  PlayerIndex would be 8 calls instead of 240. It was tried and it returned
  106 players for 2016-17 against a floor of 400, and the floor stopped the run.
  Its `Historical` parameter defaults to excluding retired players, so asking
  for 2016-17 returns only the players from that season still on a roster
  today. That is survivorship-filtered, and every since-retired player would
  have shown a blank card.

  CommonTeamRoster asks a question with one unambiguous answer: who was on this
  team in this season. Slower, and correct. It also matches what the
  dashboard's roster panel displays, so no reshaping is needed.

THIS IS DISPLAY DATA ONLY
  It is pulled after the research is complete, it enters no feature, and it
  cannot affect any published number. Tests assert that no bio column appears
  in any model feature list and that no bio column name collides with a feature
  name, since a collision would let a join silently overwrite one.

  The script reports how many players the pipeline actually saw have a bio row.
  Any that do not will show a name and no headshot, which is the honest
  fallback rather than a filled-in guess.

Writes data/raw/team_rosters/*.csv and data/raw/player_bios.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import pull_player_bios  # noqa: E402

if __name__ == "__main__":
    pull_player_bios.main()
