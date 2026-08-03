"""
Phase 9c runner: start the dashboard's Python API.

HOW TO RUN
  From a Terminal, in the project root, with your conda environment active:

      python scripts/21_serve_api.py

  Running it from Spyder also works, but a Terminal is easier because this
  process stays running until you stop it with Ctrl-C, and you will want a
  second window for the frontend.

FIRST TIME ONLY
      pip install fastapi uvicorn

  These are not needed to reproduce any research result. Every number in the
  paper comes from the packages already pinned in requirements.txt. They are
  only required to serve the dashboard.

WHAT IT SERVES, AND THE DISTINCTION THAT MATTERS
  REPLAY   /api/games and /api/games/{id} serve the precomputed files. Every
           probability in them is OUT OF FOLD: predicted by a model that never
           saw that game's season. This is what the timeline shows.

  WHAT-IF  /api/whatif and /api/predict run the saved deployment model, which
           was fitted on all eight seasons and is therefore IN-SAMPLE for every
           game in this dataset. Right for "what would the model say about a
           state that never happened", wrong for "how accurate is the model".

  Both responses carry an explicit caveat field saying which they are, and the
  frontend renders it. A viewer should never have to guess.

  /api/whatif takes a real (game_id, event_index) and applies only the
  overrides you name, so the twelve features you did not touch keep their real
  values. /api/predict requires all thirteen features explicitly and REJECTS a
  partial vector rather than defaulting the rest.

CHECK IT WORKS
  Open http://127.0.0.1:8000/docs for the interactive API documentation, or
  http://127.0.0.1:8000/api/health for a one-line status.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import api  # noqa: E402

if __name__ == "__main__":
    api.main()
