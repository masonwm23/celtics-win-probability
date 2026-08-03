"""
Phase 15: does a PREGAME-ONLY prior fix the tip-off number?

WHY THIS IS NOT TIER 5 AGAIN
  Tier 5 put opponent strength into the feature matrix for all 308,975 events
  and made things clearly worse, Brier 0.1630 -> 0.2119. Section 5 explains why:
  a feature that holds one value for a whole game and is nearly unique to that
  game lets a boosted tree memorise individual training games.

  This never touches the feature matrix. The shipped model is untouched and its
  out-of-fold predictions are read from disk. A separate logistic regression is
  fitted on GAME-level pregame facts and blended in with a weight that decays as
  real events arrive:

      p(t) = w(t) * p_pregame + (1 - w(t)) * p_model,   w(t) = exp(-t / tau)

  At tip-off w = 1 and the answer is the prior. By late in the game w is
  effectively zero. No game-constant feature ever reaches the tree, so the
  memorisation route is closed by construction rather than by hoping.

WHAT IT WAS FOR
  The shipped model cannot make Boston an underdog before tip-off. Of the
  thirteen features, the only one carrying information at 0-0 with 12:00 left is
  celtics_is_home, so the model emits exactly two numbers: 58.0% away, 71.0% at
  home. Across 636 games it never once starts Boston below 50%. In the 58 games
  where Boston were both weaker on season-to-date scoring margin AND away, they
  won 37.9% and the model opened at 58.5%.

NESTING
  For each held-out season the logistic is fitted only on the other seasons, and
  tau is chosen by an inner leave-one-season-out sweep INSIDE those training
  seasons. The held-out season influences neither.

HOW TO RUN
      python scripts/23_pregame_prior.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np, pandas as pd
from src import config
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, roc_auc_score

RNG = np.random.default_rng(20261730)
TAUS = [None, 120, 300, 600, 900, 1400, 2000, 2880]   # None = no blending
PRE = ["celtics_is_home", "celtics_point_diff_prior", "opponent_point_diff_prior"]

oof = pd.read_parquet(config.PROCESSED_DIR / 'oof_predictions.parquet')
frame = pd.read_parquet(config.MODEL_FRAME_PARQUET)
frame['game_id'] = frame['game_id'].astype('string').str.strip().str.zfill(10)
oof['game_id'] = oof['game_id'].astype('string').str.strip().str.zfill(10)
strength = pd.read_csv(config.INTERIM_DIR / 'opponent_strength.csv',
                       dtype={'GAME_ID': str})
strength['game_id'] = strength['GAME_ID'].str.zfill(10)

ev = oof.merge(frame[['game_id','event_index','seconds_elapsed_game','celtics_is_home']],
               on=['game_id','event_index'], how='left', validate='one_to_one')
assert ev['seconds_elapsed_game'].notna().all()

games = (ev.groupby('game_id')
           .agg(season=('season','first'), y=('celtics_won','first'),
                celtics_is_home=('celtics_is_home','first')).reset_index()
           .merge(strength[['game_id','celtics_point_diff_prior','opponent_point_diff_prior']],
                  on='game_id', how='left', validate='one_to_one'))
games[PRE] = games[PRE].astype(float)
assert games[PRE].notna().all().all(), "missing pregame features"
print(f"{len(games)} games, {len(ev):,} events, {games['season'].nunique()} seasons\n")

def fit_pre(train, test):
    sc = StandardScaler().fit(train[PRE])
    lr = LogisticRegression(max_iter=1000).fit(sc.transform(train[PRE]), train['y'])
    return lr.predict_proba(sc.transform(test[PRE]))[:, 1]

def blend(p_model, p_pre, elapsed, tau):
    if tau is None: return p_model
    w = np.exp(-elapsed / tau)
    return w * p_pre + (1 - w) * p_model

seasons = sorted(games['season'].unique())
p_pre_oof = pd.Series(index=games.index, dtype=float)
chosen = {}

for s in seasons:
    tr = games[games['season'] != s]
    te = games[games['season'] == s]
    p_pre_oof.loc[te.index] = fit_pre(tr, te)

    # --- inner LOSO on the training seasons only, to choose tau -------------
    inner = {t: [] for t in TAUS}
    for s2 in sorted(tr['season'].unique()):
        tr2, te2 = tr[tr['season'] != s2], tr[tr['season'] == s2]
        pp = pd.Series(fit_pre(tr2, te2), index=te2['game_id'].values)
        sub = ev[ev['season'] == s2]
        pm = sub['tier3_celtics'].to_numpy()
        el = sub['seconds_elapsed_game'].to_numpy()
        y  = sub['celtics_won'].to_numpy()
        ppv = pp.reindex(sub['game_id'].values).to_numpy()
        for t in TAUS:
            inner[t].append(brier_score_loss(y, np.clip(blend(pm, ppv, el, t), 1e-6, 1-1e-6)))
    best = min(TAUS, key=lambda t: np.mean(inner[t]))
    chosen[s] = best
    print(f"  held out {s}: inner sweep picked tau = {best}")

print("\ntau chosen per fold:", chosen)

pre_by_game = pd.Series(p_pre_oof.values, index=games['game_id'].values)
ev['p_pre'] = pre_by_game.reindex(ev['game_id'].values).to_numpy()
ev['tau'] = ev['season'].map(chosen)
ev['p_blend'] = [blend(pm, pp, el, (None if pd.isna(t) else t))
                 for pm, pp, el, t in zip(ev['tier3_celtics'], ev['p_pre'],
                                          ev['seconds_elapsed_game'], ev['tau'])]
ev['p_blend'] = ev['p_blend'].clip(1e-6, 1-1e-6)
ev.to_parquet(config.PROCESSED_DIR / 'pregame_blend_predictions.parquet')

y = ev['celtics_won'].to_numpy()
print("\n" + "="*66)
print("OVERALL, ALL 308,975 EVENTS, OUT OF FOLD")
print("="*66)
for name, col in [("shipped model", 'tier3_celtics'), ("with pregame prior", 'p_blend')]:
    p = ev[col].to_numpy()
    print(f"  {name:20} Brier {brier_score_loss(y,p):.4f}   AUC {roc_auc_score(y,p):.4f}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

# ---- cluster bootstrap on GAMES, the same unit the paper uses ---------------
rng = np.random.default_rng(20261730)
gid = ev['game_id'].to_numpy()
games = np.unique(gid)
idx_by_game = {g: np.flatnonzero(gid == g) for g in games}
def brier(p, i): return np.mean((p[i] - y[i]) ** 2)
pm = ev['tier3_celtics'].to_numpy(); pb = ev['p_blend'].to_numpy()
diffs = []
for _ in range(2000):
    pick = rng.choice(games, size=len(games), replace=True)
    i = np.concatenate([idx_by_game[g] for g in pick])
    diffs.append(brier(pm, i) - brier(pb, i))      # positive = blend is better
diffs = np.array(diffs)
lo, hi = np.percentile(diffs, [2.5, 97.5])
point = brier_score_loss(y, pm) - brier_score_loss(y, pb)
print("="*70)
print("IS THE IMPROVEMENT REAL?  cluster bootstrap, resampling games, n=636")
print("="*70)
print(f"  Brier improvement  {point:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
print(f"  verdict: {'REAL — interval excludes zero' if lo > 0 else 'not distinguishable from zero'}")

# ---- what happens at tip-off ------------------------------------------------
first = ev.sort_values(['game_id','event_index']).groupby('game_id').first().reset_index()
first = first.merge(strength[['game_id','celtics_point_diff_prior_raw',
                              'opponent_point_diff_prior_raw']], on='game_id')
first['gap'] = first['celtics_point_diff_prior_raw'] - first['opponent_point_diff_prior_raw']
print("\n" + "="*70)
print("AT TIP-OFF")
print("="*70)
for name, col in [("shipped model", 'tier3_celtics'), ("with pregame prior", 'p_blend')]:
    p = first[col]
    under = (p < 0.5).sum()
    print(f"  {name:20} range {p.min()*100:5.1f}% – {p.max()*100:5.1f}%   "
          f"Boston underdog in {under:3} of {len(first)} games   Brier {brier_score_loss(first['celtics_won'],p):.4f}")

# ---- the 58 games: weaker on paper AND away ---------------------------------
sub = first[(first['gap'] < 0) & (first['celtics_is_home'] == 0)]
print(f"\n  The {len(sub)} games where Boston were weaker on paper AND away:")
print(f"    they actually won      {sub['celtics_won'].mean()*100:5.1f}%")
for name, col in [("shipped model said", 'tier3_celtics'), ("with pregame prior", 'p_blend')]:
    print(f"    {name:22} {sub[col].mean()*100:5.1f}%")

# ---- make sure it does not damage the part that matters ---------------------
print("\n" + "="*70)
print("BY GAME PHASE — the prior must fade, not linger")
print("="*70)
bins = [(0,720,'Q1'),(720,1440,'Q2'),(1440,2160,'Q3'),(2160,2880,'Q4'),(2880,1e9,'OT')]
print(f"{'phase':6} {'events':>8} {'shipped':>9} {'blended':>9} {'change':>9}")
for lo_,hi_,lab in bins:
    m = (ev['seconds_elapsed_game']>=lo_)&(ev['seconds_elapsed_game']<hi_)
    if m.sum()==0: continue
    a=brier_score_loss(ev.loc[m,'celtics_won'],ev.loc[m,'tier3_celtics'])
    b=brier_score_loss(ev.loc[m,'celtics_won'],ev.loc[m,'p_blend'])
    print(f"{lab:6} {m.sum():8,} {a:9.4f} {b:9.4f} {a-b:+9.4f}")
