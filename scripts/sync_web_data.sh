#!/usr/bin/env bash
# Copy the serving JSON to where the dashboard reads it.
#
# The dashboard is a static site: it fetches data/serving's files straight from
# web/public/data with no backend. This keeps one copy in git rather than two
# by treating web/public/data as the shipped copy and gitignoring data/serving.
#
# Run after scripts/20_build_serving.py.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f data/serving/index.json ]; then
  echo "data/serving/index.json is missing. Run scripts/20_build_serving.py first." >&2
  exit 1
fi

mkdir -p web/public/data
rsync -a --delete data/serving/ web/public/data/

# The model itself, for the what-if panel, and its metadata, which the model
# quality panel reads. Both used to come from the Python API.
cp models/model_metadata.json web/public/data/model_metadata.json
if [ -f deploy/model_trees.json ]; then
  cp deploy/model_trees.json web/public/data/model_trees.json
fi

echo "synced:"
echo "  games        $(ls web/public/data/games | wc -l | tr -d ' ')"
echo "  size         $(du -sh web/public/data | cut -f1)"
for f in index.json coverage.json model_metadata.json model_trees.json; do
  [ -f "web/public/data/$f" ] && echo "  $f  ok" || echo "  $f  MISSING"
done
