#!/usr/bin/env bash
set -Eeuo pipefail
base="${ROBOTWIN_RENDER_EXPERIMENT_ROOT:-/root/autodl-tmp/roboscope-central-render-20261111}"
python_bin="${ROBOTWIN_RENDER_PYTHON:-/root/miniconda3/envs/starVLA/bin/python}"
name="${1:-ipc-n2-r1}"
frames="${2:-100}"
out="$base/results/$name"
mkdir -p "$out"
export PYTHONPATH="$base/site:$base${PYTHONPATH:+:$PYTHONPATH}"
export VK_ICD_FILENAMES=/root/autodl-tmp/nvidia_icd_abs.json
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
for mode in central independent; do
  active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
  if [[ -n "$active" ]]; then echo "GPU busy; stopping without changing existing processes: $active" >&2; exit 4; fi
  "$python_bin" "$base/benchmark_ipc.py" \
    --robotwin-root "$base/RoboTwin" --mode "$mode" --scenes 2 --frames "$frames" \
    --warmup 5 --output "$out/$mode.json" > "$out/$mode.log" 2>&1 || { tail -n 30 "$out/$mode.log"; exit 1; }
  echo "MODE=$mode"
  tail -n 1 "$out/$mode.log"
done
"$python_bin" "$base/compare_ipc_results.py" \
  "$out/independent.json" "$out/central.json" --output "$out/comparison.json"
