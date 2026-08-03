# Celtics Live Win Probability — dashboard

The frontend for the MSBA Directed Research project. Next.js 15, React 19, no
UI framework and no charting library: the win probability trace and the half
court are hand-drawn SVG, which keeps every pixel accountable and the dependency
list at three packages.

## Running it

Two processes. The API first, in one Terminal, from the PROJECT ROOT (not this
folder):

```
pip install fastapi uvicorn        # first time only
python scripts/21_serve_api.py
```

Then the frontend, in a second Terminal:

```
cd web
npm install                        # first time only
npm run dev
```

Open http://localhost:3000.

If the dashboard shows "cannot reach the API", the Python process is not
running or the serving data has not been built. `scripts/20_build_serving.py`
builds the data; `scripts/21_serve_api.py` serves it.

Node 18 or newer is required. `node -v` will tell you.

## The one thing to understand before reading the code

**Two different kinds of probability appear on this page and they must never be
confused.**

The timeline, the ribbon, and everything that replays a game show **out-of-fold**
predictions: each one comes from a model that never saw that game's season. They
were computed once by leave-one-season-out cross validation and stored. The
alternative would have been to run the saved deployment model, which was fitted
on all eight seasons and is therefore in-sample for every game you can replay.
It would look better and mean less.

The **what-if** panel does use the deployment model, because the question there
is "what would the model say about a state that never happened", which no stored
prediction can answer. Every response from that endpoint carries a `caveat`
string from the API and the component renders it verbatim, so the warning cannot
drift out of step with the backend.

## Layout

```
app/
  layout.js        shell and metadata
  page.js          the dashboard: state, playback, keyboard, composition
  globals.css      the whole design system, one file
components/
  MatchupRibbon    scoreboard strip; layout keyed to SIDE, not to team
  WinProbabilityChart  hand-drawn SVG trace with a scrubber
  HalfCourt        real NBA shot coordinates, five position slots
  RosterPanel      both rosters, search, position filter, on-court marking
  LineupImpact     descriptive lineup cards, and why they are descriptive
  WhatIf           margin override against the deployment model
  GamePicker       636 games by season, sortable by comeback
  PlayerRow        one player, shared by the panels
  Avatar           headshot that degrades to initials
  TeamLogo         logo that degrades to the tricode
lib/
  api.js           the four API calls
  court.js         court geometry, in the feed's own coordinate system
  format.js        clocks, periods, percentages, names
```

## Two details worth knowing

**Shot markers are not decorative.** `loc_x` and `loc_y` come straight from the
play-by-play as tenths of a foot with the hoop at the origin, confirmed by the
data itself: a shot recorded at (99, 11) has a stated distance of 10 feet, and
hypot(99, 11) / 10 = 9.96. The court in `lib/court.js` is drawn in those same
units from real NBA dimensions, so markers land where the shots were taken with
no scaling fudge anywhere.

**The lineup cards do not drive the probability, and the panel says so.** The
model that ships does not use lineup features, because lineup strength made
out-of-sample prediction measurably worse and that negative result survived
every clean re-test. Wiring the cards to the lineup model would have matched the
original design brief and contradicted the research.

## Keyboard

| Key | Action |
|---|---|
| Space | play / pause |
| ← → | step one event |
| Shift + ← → | step 25 events |

## Images

Headshots and team logos are loaded from the NBA CDN with a plain `<img>` rather
than `next/image`. About 1% of playing time across the dataset belongs to
players with no bio row, mostly short-contract call-ups, and a missing photo
should become an initials badge rather than a build-time error.
