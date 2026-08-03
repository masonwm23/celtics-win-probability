/**
 * Talking to the data, with no Python behind it.
 *
 * WHAT CHANGED AND WHY
 *   The replay was always static: /api/games, /api/games/{id} and
 *   /api/coverage did nothing but hand back files from data/serving. Those
 *   files now live in web/public/data and are fetched directly.
 *
 *   The one endpoint that genuinely needed Python was /api/whatif, which runs
 *   the saved model. A gradient boosted tree is a pile of if-statements, so the
 *   model is exported to JSON and evaluated in the browser by wp-model.js. That
 *   port reproduces XGBClassifier.predict_proba to float32 precision on all
 *   308,975 rows; the check is web/tools/verify_js_model.mjs.
 *
 *   The result is a static site: deployable anywhere, nothing to keep running.
 *
 * THE DISTINCTION THIS FILE EXISTS TO PROTECT IS UNCHANGED
 *   Two kinds of number come back from here and the interface must never blur
 *   them:
 *
 *     - a game's timeline holds OUT-OF-FOLD probabilities, each predicted by a
 *       model that never saw that game's season;
 *     - fetchWhatIf runs the deployment model, which was fitted on all eight
 *       seasons and is therefore in-sample for every game here.
 *
 *   fetchWhatIf still attaches a `caveat` field to the second kind, and every
 *   component that displays a what-if number is still expected to render it.
 *   Moving the arithmetic from a server into a browser does not make an
 *   in-sample number out of sample.
 *
 * THE FEATURE ROW IS REBUILT, NOT INVENTED
 *   A win probability model needs all thirteen inputs. Nine come straight out
 *   of the serving JSON, two are computed from those by exactly the rule in
 *   src/features.py, and the last two are the derived features recomputeDerived
 *   handles. Nothing is guessed. A game file built before the build_serving
 *   change lacks six of these columns, and what-if says so plainly rather than
 *   predicting from a half-filled row.
 */

import { whatIf as runWhatIf } from "./wp-model";

/** Kept because page.js prints it in the footer note. */
export const API_BASE = "static files in /data (no server)";

const DATA = "/data";

async function getJSON(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText} for ${url}`);
  }
  return response.json();
}

/** The API zero-padded ids; the files on disk are named the same way. */
const padId = (gameId) => String(gameId).trim().padStart(10, "0");

export const fetchGames = () => getJSON(`${DATA}/index.json`);
export const fetchGame = (gameId) => getJSON(`${DATA}/games/${padId(gameId)}.json`);
export const fetchCoverage = () => getJSON(`${DATA}/coverage.json`);

/** Was /api/model, which returned models/model_metadata.json verbatim. */
export const fetchModel = () => getJSON(`${DATA}/model_metadata.json`);

/**
 * Was /api/health, which reported whether the backend could see its files.
 * There is no backend now, so this answers the only question left: did the
 * data actually load?
 */
export async function fetchHealth() {
  const index = await fetchGames();
  return {
    status: "ok",
    serving_data: true,
    model: true,
    games: index.count ?? index.games?.length ?? 0,
    note:
      "Every probability in the timeline is out of fold: predicted by a model " +
      "that never saw this game's season.",
  };
}

// The tree file is 340 KB and only the what-if panel needs it, so it is not
// fetched until someone drags the slider, and then only once per page load.
// The splash screen now warms it during boot (see preloadDashboard), which
// fills this same slot: by the time anyone reaches the what-if panel the
// promise is already resolved and the slider responds without a stall.
let modelPromise = null;
function getModel() {
  if (!modelPromise) {
    modelPromise = getJSON(`${DATA}/model_trees.json`).catch((err) => {
      modelPromise = null; // let a later attempt retry rather than fail forever
      throw err;
    });
  }
  return modelPromise;
}

/* -------------------------------------------------------------- boot loading

   The splash screen reports how far the download has actually got. Nothing
   about that number is decorative: it is bytes received over bytes promised,
   for the three files the dashboard cannot open without.

   Two honest limits, both handled rather than hidden:

     - A compressed response carries the COMPRESSED length in content-length
       while the stream hands back DECODED bytes, so the ratio can pass 1
       early. It is clamped, which means a stage can sit at its ceiling for a
       moment before the next one starts. It never runs backwards and it never
       claims a file finished before it did.
     - A response with no content-length at all (chunked transfer) has nothing
       to measure against. That stage reports 0 until it completes and then
       reports 1, rather than inventing a curve.

   The caller decides what to do about a download that finishes faster than the
   eye can follow. This file's job is only to be accurate.
---------------------------------------------------------------------------- */

/**
 * The share of the boot each file is worth, from their real sizes on disk:
 * index.json 253 KB, one game file about 88 KB, model_trees.json 342 KB.
 * They sum to 1.
 */
export const BOOT_STAGES = [
  { key: "index", label: "Loading the game index", weight: 0.37 },
  { key: "game", label: "Loading play-by-play", weight: 0.13 },
  { key: "model", label: "Loading the win probability model", weight: 0.5 },
];

/** Where each stage starts on the 0-1 bar. */
export const BOOT_STAGE_START = BOOT_STAGES.reduce(
  (acc, stage, i) => {
    acc.push(acc[i] + stage.weight);
    return acc;
  },
  [0]
);

/** The stage label for a position on the bar. */
export function bootLabel(fraction) {
  for (let i = 0; i < BOOT_STAGES.length; i += 1) {
    if (fraction < BOOT_STAGE_START[i + 1]) return BOOT_STAGES[i].label;
  }
  return "Ready";
}

/**
 * Fetch JSON, reporting the fraction of it that has arrived.
 *
 * `onFraction` is called with a clamped 0-1 value as chunks land, and once
 * more with exactly 1 when the body is complete.
 */
async function getJSONProgress(url, onFraction) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText} for ${url}`);
  }

  const promised = Number(response.headers.get("content-length"));
  const streamable =
    response.body && typeof response.body.getReader === "function";

  if (!promised || !streamable) {
    const data = await response.json();
    onFraction?.(1);
    return data;
  }

  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    // eslint-disable-next-line no-await-in-loop
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onFraction?.(Math.min(received / promised, 1));
  }
  onFraction?.(1);
  return JSON.parse(await new Blob(chunks).text());
}

/**
 * Everything the dashboard needs before it can draw anything, fetched in the
 * order it becomes knowable: the index first, because the opening game is
 * chosen from it, then that game, then the model.
 *
 * `pickGame` receives the loaded index and returns the game id to open.
 * `onProgress` receives a 0-1 fraction of the whole boot.
 *
 * Returns the index, the chosen id and that game's payload, so the caller does
 * not fetch the opening game a second time.
 */
export async function preloadDashboard({ pickGame, onProgress } = {}) {
  const report = (stage, within) => {
    if (!onProgress) return;
    const start = BOOT_STAGE_START[stage];
    onProgress(start + BOOT_STAGES[stage].weight * Math.min(Math.max(within, 0), 1));
  };

  const index = await getJSONProgress(`${DATA}/index.json`, (f) => report(0, f));

  const gameId = pickGame ? pickGame(index) : index.games[0].game_id;
  const game = await getJSONProgress(
    `${DATA}/games/${padId(gameId)}.json`,
    (f) => report(1, f)
  );

  // Warming the model fills the same cache the what-if panel reads, so this is
  // not busy-work to pad the bar: it is the one download that would otherwise
  // happen while somebody is waiting on a slider.
  //
  // It is also the only stage allowed to fail quietly. The dashboard opens
  // perfectly well without it; the what-if panel would simply fetch it later,
  // exactly as it did before this existed.
  try {
    const trees = await getJSONProgress(`${DATA}/model_trees.json`, (f) =>
      report(2, f)
    );
    modelPromise = Promise.resolve(trees);
  } catch {
    modelPromise = null;
  }
  report(2, 1);

  return { index, gameId, game };
}

const REQUIRED = [
  "seconds_remaining_period",
  "seconds_remaining_game",
  "celtics_has_possession",
  "momentum_120s",
  "momentum_300s",
  "possession_number",
];

/**
 * Rebuild one event's full feature row from the serving payload.
 *
 * `i` is the position in the columnar arrays, not the event_index value.
 */
function featureRow(game, i) {
  const e = game.events;
  const missing = REQUIRED.filter((key) => !e[key]);
  if (missing.length) {
    throw new Error(
      `this game file is missing model inputs (${missing.join(", ")}). ` +
        `Re-run the serving build after applying the build_serving patch.`
    );
  }
  return {
    celtics_margin: e.margin[i],
    seconds_remaining_period: e.seconds_remaining_period[i],
    seconds_remaining_game: e.seconds_remaining_game[i],
    seconds_elapsed_game: e.elapsed[i],
    period: e.period[i],
    is_overtime: e.period[i] >= 5 ? 1 : 0,
    celtics_is_home: game.meta.celtics_is_home ? 1 : 0,
    celtics_has_possession: e.celtics_has_possession[i],
    momentum_120s: e.momentum_120s[i],
    momentum_300s: e.momentum_300s[i],
    // Both derived features are recomputed by wp-model.js from the row above,
    // which is the point: an override to the margin has to reach them.
    is_clutch: e.is_clutch[i] ? 1 : 0,
    margin_per_minute_remaining: 0,
    possession_number: e.possession_number[i],
  };
}

/**
 * Re-predict a real event with some features replaced.
 *
 * Same signature and same response shape as the old POST /api/whatif, so no
 * calling component changes. Everything not named keeps its real value.
 */
export async function fetchWhatIf(gameId, eventIndex, overrides) {
  const [model, game] = await Promise.all([getModel(), fetchGame(gameId)]);

  const i = game.events.event_index.indexOf(eventIndex);
  if (i === -1) {
    throw new Error(`event ${eventIndex} is not in game ${padId(gameId)}`);
  }

  return {
    ...runWhatIf(model, featureRow(game, i), overrides),
    game_id: padId(gameId),
    event_index: eventIndex,
    overrides,
  };
}
