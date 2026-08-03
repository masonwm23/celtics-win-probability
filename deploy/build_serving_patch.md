# Patch 1 of 2 — `src/build_serving.py`

The serving JSON already carries nine of the model's thirteen inputs. Four are
missing, and without them the what-if panel cannot run the model in the browser.
This adds them.

## Find this block

It is inside `build_game_payload`, around line 492, and currently ends like
this:

```python
            "is_clutch": [bool(v) for v in merged["is_clutch"]],
            "celtics_lineup": celtics_lineup,
            "opponent_lineup": opponent_lineup,
        },
    }
```

## Replace it with this

```python
            "is_clutch": [bool(v) for v in merged["is_clutch"]],
            "celtics_lineup": celtics_lineup,
            "opponent_lineup": opponent_lineup,

            # The four model inputs that are not already above, so the what-if
            # panel can evaluate the saved model in the browser instead of
            # calling Python. The other nine are either here already or derived
            # from them in web/lib/api.js.
            #
            # Seconds are rounded to hundredths, which is the resolution the
            # NBA clock is recorded at; rounding them changes no prediction on
            # any of the 308,975 rows. Everything else here is a whole number.
            "seconds_remaining_period": [
                round(float(v), 2) for v in merged["seconds_remaining_period"]],
            "seconds_remaining_game": [
                round(float(v), 2) for v in merged["seconds_remaining_game"]],
            "celtics_has_possession": [
                int(bool(v)) for v in merged["celtics_has_possession"]],
            "momentum_120s": [int(v) for v in merged["momentum_120s"]],
            "momentum_300s": [int(v) for v in merged["momentum_300s"]],
            "possession_number": column("possession_number", int),
        },
    }
```

## Then rebuild the serving data

```bash
python scripts/20_build_serving.py
```

That rewrites `data/serving/games/*.json`. Each file grows by roughly 10 KB, so
about 6 MB across all 636 games — fine for GitHub.

## Check it worked

```bash
python -c "
import json, glob
f = sorted(glob.glob('data/serving/games/*.json'))[0]
e = json.load(open(f))['events']
need = ['seconds_remaining_period','seconds_remaining_game',
        'celtics_has_possession','momentum_120s','momentum_300s','possession_number']
missing = [k for k in need if k not in e]
print('OK' if not missing else 'MISSING: ' + ', '.join(missing))
print('events:', len(e['event_index']))
"
```

You want `OK`.
