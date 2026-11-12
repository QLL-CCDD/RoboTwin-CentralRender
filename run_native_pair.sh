#!/usr/bin/env bash
set -Eeuo pipefail
base="${ROBOTWIN_RENDER_EXPERIMENT_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
python_bin="${ROBOTWIN_RENDER_PYTHON:-python}"
out="${1:?Provide a new output directory}"
test ! -e "$out" || { echo "Choose a new output directory to preserve results" >&2; exit 2; }
mkdir -p "$out"
export PYTHONPATH="$base/site:$base${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for mode in central independent; do
  active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
  test -z "$active" || { echo "GPU occupied; no new run: $active" >&2; exit 4; }
  "$python_bin" -u "$base/benchmark_native.py" --mode "$mode" --scenes 2 \
    --output-root "$out" > "$out/$mode.log" 2>&1 || { tail -n 25 "$out/$mode.log"; exit 1; }
done
"$python_bin" "$base/compare_native.py" "$out"
