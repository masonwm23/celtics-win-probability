# Putting the dashboard on a real URL

End state: your professor clicks a link and the app works — replay, timeline,
what-if slider, everything. No install, no server, nothing you have to keep
running. Free, permanently.

**How:** your replay was always static JSON, and the only thing that needed
Python was the what-if slider. The model is now exported to JSON and evaluated
in the browser, so the whole app is a static site. Vercel hosts those free and
deploys straight from GitHub.

About 45 minutes. Do the steps in order — each one is checkable.

---

## Step 1 — add the missing model inputs

Follow `build_serving_patch.md`. It adds six columns to the serving JSON and
tells you how to confirm they landed. Do not skip the check at the end; if the
columns are not there, the what-if slider will show an error instead of a
number.

---

## Step 2 — move the files the browser needs

From your `celtics_wp` folder:

```bash
mkdir -p web/public/data
cp -r data/serving/* web/public/data/
cp browser-model/model_trees.json web/public/data/
mkdir -p web/tools
cp browser-model/wp-model.js web/lib/
cp browser-model/verify_js_model.mjs web/tools/
```

Then replace your data layer with the static one:

```bash
cp web/lib/api.js web/lib/api.py.bak     # keep the old one around
cp deploy/api.js web/lib/api.js
```

Check what you just moved:

```bash
ls web/public/data/index.json web/public/data/model_trees.json
ls web/public/data/games | wc -l          # should be 636
du -sh web/public/data                    # expect roughly 25-40 MB
```

If `web/public/data` comes out over 100 MB, stop and tell me — we would switch
to loading game files from a CDN instead, and it is a small change.

---

## Step 3 — run it locally with no Python

Kill the uvicorn terminal. You should not need it any more.

```bash
cd web
npm run dev
```

Open `localhost:3000` and check three things:

1. A game loads and the timeline plays.
2. You can switch games from the games panel.
3. **Drag the what-if slider.** A number should appear, and the in-sample
   caveat should still be under it.

If the slider errors, the serving files did not get rebuilt in Step 1.

---

## Step 4 — push to GitHub

Create `.gitignore` at the repo root:

```
.venv/
__pycache__/
*.pyc
.DS_Store
web/node_modules/
web/.next/
data/raw/
```

Note that `web/public/data/` is **not** ignored — those files have to ship, they
are the app.

```bash
git init
git add .
git commit -m "Celtics real-time win probability model: paper, pipeline and dashboard"
```

Make a new **public** repo on github.com named `celtics-win-probability`, then:

```bash
git remote add origin https://github.com/YOUR-USERNAME/celtics-win-probability.git
git branch -M main
git push -u origin main
```

---

## Step 5 — deploy

1. Go to **vercel.com** and sign in with GitHub.
2. **Add New → Project**, pick `celtics-win-probability`.
3. One setting matters: set **Root Directory** to `web`. Vercel will detect
   Next.js on its own.
4. **Deploy.**

Two or three minutes later you have a URL like
`celtics-win-probability.vercel.app`. Every future `git push` redeploys it
automatically.

---

## Step 6 — check the deployed app

Open the URL on your phone, not just your laptop — that catches anything that
was only working because of a local file.

- A game loads and plays
- The games panel switches games
- **The what-if slider returns a number and still shows the in-sample caveat**
- The "how to read" and model quality panels open

---

## If something breaks

**Blank page, console shows 404 for `/data/index.json`**
The data folder did not get committed. Run `git status` — if it lists
`web/public/data`, something in `.gitignore` is catching it. `data/` on its own
line would do it; use `data/raw/` as written above.

**What-if says the game file is missing model inputs**
Step 1's rebuild did not happen, or you copied the serving files before running
it. Re-run `scripts/20_build_serving.py`, then redo Step 2's copy.

**Build fails on Vercel with a module-not-found for `@/lib/wp-model`**
`wp-model.js` did not land in `web/lib/`. Check with `ls web/lib/`.

**Push rejected, file over 100 MB**
Find it with `find . -size +100M -not -path "./web/node_modules/*"`. Almost
certainly something under `data/raw/`, which should be gitignored.

---

## What to tell your professor

Add this near the top of the README, with the real URL:

> **Live dashboard:** https://celtics-win-probability.vercel.app
>
> The timeline shows out-of-fold probabilities — each number came from a model
> that never saw that game's season. The what-if panel uses the deployment
> model, which is in-sample for every game here, and says so on screen.

---

## One thing to know before he plays with the slider

The what-if uses the deployment model, fitted on all eight seasons. At large
deficits it is closer to recalling a specific game than forecasting one: only 46
of 636 games ever reached a 20-point deficit in the third quarter, and Boston
won 3 of them. On one such event the deployment model says 77% where the
out-of-fold model says 33%.

Your slider is already capped at ±20 from the actual margin, which keeps most of
this out of reach, and the panel carries the caveat. It is documented in the
README's limitations section. If you would rather be conservative, narrowing the
slider to ±12 removes nearly all of it — one number in `WhatIf.js`.

It is worth knowing rather than being surprised by: it is the same memorisation
effect your Section 5 is about, showing up in the interface.
