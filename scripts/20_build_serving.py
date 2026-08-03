"""
Phase 9b runner: build the JSON the dashboard reads.

HOW TO RUN IN SPYDER
  Open this file and press F5 (Run file).

No network, no model fitting. Joins the five pipeline tables and writes one
compact file per game. A minute or two, and roughly 60 MB on disk.

THE PROBABILITY SHOWN IS OUT OF FOLD
  This is the important decision. The saved deployment model was fitted on all
  eight seasons, so asking it to predict any game in the dataset is in-sample.
  It would look better and mean less.

  The dashboard replays `tier3_celtics` from oof_predictions.parquet instead.
  Every probability on screen comes from a model that never saw that game's
  season, and the payload records that fact so it can be checked. The generic
  baseline is carried alongside so the interface can show the paper's headline
  comparison directly.

WHAT IT CHECKS
  - game ids are normalised before every join, because game_index.csv stores
    them as integers and the parquets store zero-padded strings. That join
    returns zero rows and raises nothing, which is why the schema probe exists.
  - a missing probability RAISES rather than being filled with 0.5 or with the
    previous value. A number nobody's model produced must not reach the screen.
  - a lineup join that leaves an event without a lineup raises. A left join
    does not lose rows when the right side is short, it fills them with nulls,
    so a row-count check alone would miss it.

WHAT IT MEASURES
  Bio coverage in MINUTES, not headcount. 187 player-seasons have no bio row,
  but a headcount treats a 4-minute call-up and a starter the same. The output
  reports what share of playing time belongs to players with no bio, and the
  median minutes of those players, which is the test of whether they are
  hardship call-ups or rotation players.

  A player with no bio row still gets a photo, a name, a jersey number and a
  coarse position. What they lose is height and the granular position. Nothing
  is filled in with a guess.

Writes data/serving/index.json, data/serving/games/*.json, coverage.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import build_serving  # noqa: E402

if __name__ == "__main__":
    build_serving.main()
