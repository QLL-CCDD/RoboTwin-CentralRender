#!/usr/bin/env bash
set -Eeuo pipefail
# Usage: run_pair.sh /path/to/python /path/to/RoboTwin /path/to/launcher.py output-dir [N] [frames]
python_bin="${1:?Python executable required}"
robotwin_root="${2:?Existing RoboTwin tree required}"
launcher="${3:?Kit RoboTwin launcher required}"
output_dir="${4:?Output directory required}"
scenes="${5:-2}"
frames="${6:-100}"
tools_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
mkdir -p "$output_dir"
output_dir="$(cd -- "$output_dir" && pwd -P)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
"$python_bin" "$tools_dir/probe_environment.py" --robotwin-root "$robotwin_root" --output "$output_dir/environment.json"
for mode in independent shared; do
  # Do not interrupt other tasks or measure against an active compute workload.
  active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
  if [[ -n "$active" ]]; then
    echo "GPU has other compute processes; leaving them unchanged and stopping: $active" >&2
    exit 4
  fi
  busy="$(nvidia-smi --id=0 --query-gpu=utilization.gpu --format=csv,noheader,nounits)"
  if (( busy > 10 )); then
    echo "GPU is active ($busy percent); stopping without launching renderers" >&2
    exit 4
  fi
  "$python_bin" "$tools_dir/render_compare.py" --robotwin-root "$robotwin_root" --launcher "$launcher" \
    --scene-source "${SCENE_SOURCE:-native}" \
    --mode "$mode" --schedule submit-all --scenes "$scenes" --frames "$frames" --warmup 5 \
    --cameras "${CAMERA_SET:-all}" --output "$output_dir/$mode.json" > "$output_dir/$mode.log" 2>&1
  cat "$output_dir/$mode.log"
done
"$python_bin" "$tools_dir/compare_results.py" "$output_dir/independent.json" "$output_dir/shared.json" \
  --output "$output_dir/comparison.json" | tee "$output_dir/comparison.txt"
