#!/usr/bin/env bash
set -Eeuo pipefail
mode="${1:?native or central}"
name="${2:?new run name}"
[[ "$mode" == native || "$mode" == central ]]
base="$(cd "$(dirname "$0")" && pwd)"
run="$base/closed-loop/$name"
test ! -e "$run" || { echo "Run directory already exists" >&2; exit 2; }
active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
test -z "$active" || { echo "GPU occupied; no run: $active" >&2; exit 4; }
mkdir -p "$run"
control="$base/policy-env/bin/python"
sim=/root/miniconda3/envs/starVLA/bin/python
export PYTHONPATH="$base/sdk-site:$base/site:$base:$base/policy-source"
export STARVLA_ROOT=/root/starVLA CUDA_VISIBLE_DEVICES=0
export VK_ICD_FILENAMES=/root/autodl-tmp/nvidia_icd_abs.json
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 NO_ALBUMENTATIONS_UPDATE=1
export PYTHONFAULTHANDLER=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export ROBOTWIN_FAST_CONTROL=1 ROBOTWIN_SHARED_ENGINE=1
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
unset ROBOTWIN_CENTRAL_RENDER_DESCRIPTOR ROBOSCOPE_DIAGNOSTIC_NVTX ROBOSCOPE_PROFILE
"$control" "$base/campaign-tools/monitor_run.py" --output-prefix "$run/monitor" --interval .5 --gpu-devices 0 -- \
  "$control" "$base/policy_run.py" --mode "$mode" --run-dir "$run" --sim-python "$sim" \
  --robotwin-root "$base/policy-runtime/RoboTwin" -- \
  "$control" "$base/campaign-tools/experiment_subset.py" --phase-log "$run/phases.jsonl" \
  --benchmark robotwin --suite demo_clean --checkpoint-source "$base/models/starvla-robotwin-native" \
  --benchmark-manifest "$base/policy-runtime/experiment-manifest.json" \
  --policy-config "$base/closed-loop-policy.json" --output-dir "$run/sdk" \
  --subset-task-count 1 --subset-episodes 2 --episode-shard-size 1 --concurrency 2 --replicas 1 \
  > "$run/run.log" 2>&1
echo "Completed $mode: $run"
