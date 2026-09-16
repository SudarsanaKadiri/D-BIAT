#!/usr/bin/env bash
# Run all analysis notebooks in order and save them with their outputs.
# Usage (from this folder):  bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")/notebooks"
for nb in 0[1-7]_*.ipynb; do
  echo "== $nb"
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 "$nb"
done
echo "Done. Results are in outputs/ (tables, figures, derived)."
