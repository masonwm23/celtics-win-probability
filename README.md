# Celtics Real-Time Win Probability Model

MSBA Directed Research, Brandeis University
Student: Mason Marathias | Supervisor: Prof. Ahmad Namini

A win probability model calibrated specifically to the Boston Celtics, trained on
eight seasons of play-by-play data (2016-17 through 2023-24), with opponent
matchup context, and benchmarked against a generic in-game baseline.

---

## Current status

| Phase | Description | Status |
|---|---|---|
| 1 | Data pulled and cleaned | In progress. Game index scaffold built, awaiting first run |
| 2 | Features created correctly | Not started |
| 3 | Train/test split prevents leakage | Not started |
| 4 | Model trained and evaluated | Not started |
| 5 | Opponent feature tested honestly | Not started |
| 6 | Model saved and reproducible | Not started |
| 7 | Research results and documentation | Not started |
| 8 | Frontend connected after validation | Not started |

Nothing is marked complete until its validation report passes and the result has
been reviewed in plain English.

---

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

## Principles this codebase follows

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
| Position source, to be confirmed | Granular PG/SG/SF/PF/C positions. The NBA API only exposes broad Guard/Forward/Center buckets |

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
