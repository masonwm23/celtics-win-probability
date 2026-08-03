"""
Phase 9b, step 0: print the schema of every table the dashboard will serve.

HOW TO RUN IN SPYDER
  Open this file and press F5 (Run file).

No network, no model, reads a few rows and prints. Seconds.

WHY IT EXISTS
  The API joins five tables written across four different phases. Writing that
  join against remembered column names is how a serving layer silently drops a
  column or matches on a key with the wrong dtype, which produces an empty join
  rather than an error.

  This prints what is actually on disk so the serving code can be written
  against observed schemas. Paste the output back before the API is built.

Writes nothing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import schema_probe  # noqa: E402

if __name__ == "__main__":
    schema_probe.main()
