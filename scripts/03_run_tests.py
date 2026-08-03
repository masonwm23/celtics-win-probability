"""
Phase 1, step 4 runner: run the test suite.

HOW TO RUN IN SPYDER
  Open this file and press F5 (Run file).

Makes no network calls and reads no project data. Every test uses synthetic
fixtures. If anything here fails, the pipeline logic is wrong and we fix it
before trusting any real result.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(pytest.main(["-v", "--no-header", str(ROOT / "tests")]))
