"""
Phase 1, step 1: environment check.

Run this FIRST, in Spyder. It does two things:
  1. Prints the exact versions of everything the project depends on, so the
     requirements file can be pinned to what you actually have rather than to
     guesses.
  2. Makes one small live call to stats.nba.com to confirm your machine can
     reach the NBA API at all, before we attempt a long pull.

It writes nothing except a text report in reports/. It does not download data.

HOW TO RUN IN SPYDER
  Open this file and press F5 (Run file). Then paste the console output back.
"""

import sys
import platform
import importlib
from datetime import datetime, timezone
from pathlib import Path

# Make `src` importable when this file is run directly from the scripts folder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

PACKAGES = [
    "pandas",
    "numpy",
    "sklearn",
    "xgboost",
    "shap",
    "pyarrow",
    "requests",
    "nba_api",
    "matplotlib",
    "scipy",
    "joblib",
    "pytest",
]


def check_versions():
    lines = []
    lines.append("PYTHON")
    lines.append(f"  executable : {sys.executable}")
    lines.append(f"  version    : {sys.version.splitlines()[0]}")
    lines.append(f"  platform   : {platform.platform()}")
    lines.append("")
    lines.append("PACKAGES")
    missing = []
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "installed (no __version__)")
            lines.append(f"  {name:<12}: {version}")
        except ImportError:
            lines.append(f"  {name:<12}: MISSING")
            missing.append(name)
    return lines, missing


def check_nba_api_reachable():
    """
    One lightweight live call. Uses the 2023-24 Celtics game log because it is
    small and its expected size is known, so the result is self-validating.
    """
    lines = ["", "NBA API CONNECTIVITY"]
    try:
        from nba_api.stats.endpoints import leaguegamefinder
    except ImportError:
        lines.append("  SKIPPED: nba_api is not installed.")
        lines.append("  Install it with:  pip install nba_api")
        return lines, False

    try:
        import time
        t0 = time.time()
        finder = leaguegamefinder.LeagueGameFinder(
            team_id_nullable=config.CELTICS_TEAM_ID,
            season_nullable="2023-24",
            season_type_nullable=config.SEASON_TYPE,
            timeout=config.REQUEST_TIMEOUT,
        )
        df = finder.get_data_frames()[0]
        elapsed = time.time() - t0
        lines.append(f"  Request succeeded in {elapsed:.1f}s")
        lines.append(f"  Rows returned: {len(df)} (expected 82 for 2023-24)")
        if len(df):
            cols = ["GAME_DATE", "MATCHUP", "WL", "PTS"]
            have = [c for c in cols if c in df.columns]
            lines.append("  Sample rows:")
            for line in df[have].head(3).to_string(index=False).splitlines():
                lines.append(f"    {line}")
        ok = len(df) == 82
        if not ok:
            lines.append("  WARNING: row count is not 82. Do not proceed until")
            lines.append("           we understand why. Report this output.")
        return lines, ok
    except Exception as exc:  # noqa: BLE001 - we want the type name reported
        lines.append(f"  REQUEST FAILED: {type(exc).__name__}")
        lines.append(f"  {str(exc)[:400]}")
        lines.append("  If this is a timeout, try again once. NBA's API is")
        lines.append("  intermittent. If it fails repeatedly, report this.")
        return lines, False


def main():
    config.ensure_dirs()

    header = [
        "=" * 70,
        "CELTICS WIN PROBABILITY - ENVIRONMENT CHECK",
        f"Run at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Project root: {config.PROJECT_ROOT}",
        "=" * 70,
        "",
    ]

    version_lines, missing = check_versions()
    api_lines, api_ok = check_nba_api_reachable()

    footer = ["", "SUMMARY"]
    if missing:
        footer.append(f"  Missing packages: {', '.join(missing)}")
        footer.append(f"  Install with: pip install {' '.join(missing)}")
    else:
        footer.append("  All required packages present.")
    footer.append(f"  NBA API reachable and returned expected size: {api_ok}")
    if not missing and api_ok:
        footer.append("")
        footer.append("  READY for Phase 1 step 2 (game index pull).")
    else:
        footer.append("")
        footer.append("  NOT READY. Resolve the items above first.")

    all_lines = header + version_lines + api_lines + footer
    report = "\n".join(all_lines)
    print(report)

    out = config.REPORTS_DIR / "phase1_env_check.txt"
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\nReport saved to: {out}")


if __name__ == "__main__":
    main()
