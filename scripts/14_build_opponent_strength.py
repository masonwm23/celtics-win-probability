"""
Phase 5, step 2 runner: build as-of-date opponent strength.

HOW TO RUN IN SPYDER
  Open this file and press F5 (Run file).

No network. Every measure uses games strictly BEFORE the game in question, so a
game never contributes to its own feature and no later game reaches an earlier
one. Rates are shrunk toward the league mean by games played, because a team that
is 1-0 is not the best team in the league.

Writes data/interim/opponent_strength.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src import config, opponent_strength  # noqa: E402


def main():
    config.ensure_dirs()
    strength = opponent_strength.build_opponent_strength()
    out = config.INTERIM_DIR / "opponent_strength.csv"
    strength.to_csv(out, index=False)

    print("=" * 70)
    print("OPPONENT STRENGTH BUILT (as of game date, prior games only)")
    print("=" * 70)
    print(f"Rows: {len(strength)}  ->  {out}")
    print()
    print("Sanity checks:")
    first = strength.loc[strength["opponent_games_played_prior"].eq(0)]
    print(f"  games where the opponent had played 0 prior games: {len(first)}")
    print(f"  opponent_point_diff_prior range: "
          f"{strength['opponent_point_diff_prior'].min():.2f} to "
          f"{strength['opponent_point_diff_prior'].max():.2f}")
    print(f"  opponent_win_pct_prior range: "
          f"{strength['opponent_win_pct_prior'].min():.3f} to "
          f"{strength['opponent_win_pct_prior'].max():.3f}")
    print(f"  nulls: {int(strength[opponent_strength.OPPONENT_FEATURE_COLUMNS].isna().sum().sum())}")
    print()
    print("  A full-season average would be a LEAK. These are as-of-date.")
    print()
    print("Next: run scripts/11_train_model.py again to test opponent context.")
    return strength


if __name__ == "__main__":
    main()
