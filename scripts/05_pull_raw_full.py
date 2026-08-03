"""
Phase 1, step 5b runner: FULL raw pull, all 636 games.

HOW TO RUN IN SPYDER
  Open this file and press F5 (Run file).

This is the long job: 636 games times 2 endpoints, about 1,270 API calls.
Expect roughly 30 to 60 minutes depending on how NBA's servers behave.

It is safe to interrupt. Completed games are cached on disk, so re-running
picks up where it stopped and only fetches what is missing. Progress prints
every 25 games with an ETA, and any problem game prints immediately.

Do not run this until the smoke test has been reviewed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pull_raw import main_full  # noqa: E402

if __name__ == "__main__":
    main_full()
