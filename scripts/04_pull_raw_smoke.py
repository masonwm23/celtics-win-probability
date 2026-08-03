"""
Phase 1, step 5a runner: SMOKE TEST of the raw pull.

HOW TO RUN IN SPYDER
  Open this file and press F5 (Run file).

Pulls 16 games only, two per season, so both older and newer response formats
get exercised. Takes roughly one minute. Writes a schema probe report showing
the ACTUAL field names the API returns, which is what the Phase 2 parser will
be written against.

Run this BEFORE the full pull. If the response shape is not what we expect, it
is much better to find out after 16 games than after 636.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pull_raw import main_smoke  # noqa: E402

if __name__ == "__main__":
    main_smoke()
