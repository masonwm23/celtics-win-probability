# Celtics Real-Time Win Probability Model

MSBA Directed Research, Brandeis University
Student: Mason Marathias | Supervisor: Prof. Ahmad Namini

A win probability model calibrated specifically to the Boston Celtics, trained on
eight seasons of play-by-play data (2016-17 through 2023-24), with opponent
matchup context, and benchmarked against a generic in-game baseline (ESPN).

## Project Summary

This project presents a real-time Boston Celtics win probability model trained on eight NBA seasons (2016–17 through 2023–24) of play-by-play data. The model predicts the Celtics' probability of winning at every game state and is deployed through an interactive web dashboard that reconstructs possessions, displays players on the floor, visualizes win probability over time, and compares the Celtics-specific model against a generic baseline (ESPN).

### Project Highlights

- 636 Boston Celtics regular-season games
- Eight NBA seasons (2016–17 through 2023–24)
- Over 308,000 in-game game states analyzed
- XGBoost model trained using 13 game-state features
- Interactive Next.js dashboard with play reconstruction and lineup visualization
- Every displayed probability generated out-of-fold during evaluation
---

## Project Completion

| Phase | Description | Status |
|---|---|---|
| 1 | Data pulled and cleaned | ✅ Completed|
| 2 | Features created correctly | ✅ Completed|
| 3 | Train/test split prevents leakage |✅ Completed|
| 4 | Model trained and evaluated | ✅ Completed|
| 5 | Opponent feature tested honestly |✅ Completed|
| 6 | Model saved and reproducible | ✅ Completed|
| 7 | Research results and documentation |✅ Completed|
| 8 | Frontend connected after validation | ✅ Complete|

All project phases have been successfully completed and validated. The final model, interactive dashboard, and supporting documentation have been reviewed and verified.

---
## Live Demo

Interactive dashboard:

https://celtics-win-probability.vercel.app

### Dashboard Preview

<img width="1581" height="826" alt="Screenshot 2026-08-04 at 1 48 14 PM" src="https://github.com/user-attachments/assets/5d33dc4d-9706-42cf-b348-8de010b85b16" />

---
## Research Paper

This repository accompanies the MSBA Directed Research paper describing the methodology, feature engineering, model training, evaluation, and results.

**Read the paper:**
[Celtics Real-Time Win Probability Model (PDF)]
[MWM CLWP Paper-Final.pdf](https://github.com/user-attachments/files/30722488/MWM.CLWP.Paper-Final.pdf)


## Repository Structure

**Backend**  
Python pipeline for data collection, feature engineering, model training, and evaluation.

**Frontend**  
Next.js interactive dashboard for play reconstruction, win probability visualization, and lineup exploration.

**Deployment**  
Hosted on Vercel.
## How to run

Everything runs from the `scripts/` folder, in numbered order. Open a script in
Spyder and press F5. The scripts are thin wrappers around modules in `src/`, so
the same code can also be run from a terminal with `python -m src.<module>`.

| Script | Network? | What it does |
|---|---|---|
| `00_env_check.py` | Yes, one small call | Reports installed versions and confirms stats.nba.com is reachable |
| `01_pull_game_index.py` | Yes, 8 calls | Builds `data/raw/game_index.csv`, one row per Celtics regular season game |
| `02_validate_game_index.py` | No | Audits the game index against 10 checks, writes a report |
| `03_run_tests.py` | No | Runs the test suite against synthetic fixtures |

Run them in order. If a validation step fails, stop. Do not continue past a
failed check.

### First time setup

```bash
pip install -r requirements.txt
```

---

## Layout

```
celtics_wp/
  README.md
  requirements.txt
  scripts/            numbered entry points, run these in Spyder
  src/                the actual pipeline modules
    config.py           every path, season, and constant lives here
    nba_client.py       retry wrapper around nba_api calls
    pull_game_index.py  builds the game index
    validate_game_index.py  independent audit of the game index
  tests/              synthetic-fixture tests, no network, no project data
  data/
    raw/                exactly what the API returned. Never edited by hand
    interim/            parsed, not yet feature engineered
    processed/          model-ready tables
  models/             saved models plus their feature schema
  reports/            validation reports and results tables
  logs/
```

---

## Model Design Principles

These are not decoration. They are the reason the results can be trusted.

**Raw data is immutable.** Anything in `data/raw/` is byte-for-byte what the API
returned. Every transformation happens downstream and is scripted, so it can be
inspected and re-run.

**Failures are recorded, never swallowed.** A network call that fails after
retries raises. It never returns an empty frame, because an empty frame is
indistinguishable from a game that genuinely had no events.

**Parsers refuse ambiguity.** `parse_matchup` raises on any string it does not
recognise rather than guessing an opponent. The same rule applies to every
parser added later.

**The auditor is separate from the builder.** The script that creates the game
index is not the only script that vouches for it. `validate_game_index.py` is an
independent check, and the test suite deliberately corrupts synthetic data to
prove the audit actually catches problems.

**No fabricated data.** Where a value is genuinely unavailable it is reported as
unavailable. Synthetic data appears only in `tests/`, is labelled as synthetic,
and is never written into `data/`.

**Aggregate features are computed inside folds.** Any feature built from
aggregates, such as lineup strength or opponent strength, is recomputed within
each cross-validation fold using only that fold's training seasons. Building
them once across the full dataset leaks test information into training.

**Opponent quality uses only pregame information.** A team's full-season record
includes games played after the game being predicted, so joining it on is a
leak. Opponent strength is computed as-of the game date from prior games only.

**The benchmark is labelled honestly.** ESPN's win probability model is not
published. The comparison model here is a logistic regression on score margin
and time remaining, and it is called a generic baseline, not ESPN's model.

---

## Data sources

| Source | Used for |
|---|---|
| `nba_api` `LeagueGameFinder` | Game index: dates, opponents, results |
| `nba_api` `PlayByPlayV3` | Event-level play-by-play |
| `nba_api` `BoxScoreTraditionalV3` | Starters, player minutes, both teams' rosters |
| Position source, Basketball Reference| Granular PG/SG/SF/PF/C positions. The NBA API only exposes broad Guard/Forward/Center buckets |

All data is historical and pulled from public endpoints. Nothing is entered by
hand.

---

## Reproducibility

- One random seed, `RANDOM_SEED` in `src/config.py`, used everywhere
- No hard-coded paths outside `src/config.py`
- `requirements.txt` will be pinned to exact versions after the environment check
- Raw API payloads cached to disk, so re-running analysis does not re-hit the API
- Saved models are stored together with their feature schema, so the API cannot
  silently feed features in the wrong order
