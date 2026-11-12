#!/usr/bin/env bash
set -Eeuo pipefail
base=/root/autodl-tmp/roboscope-central-render-20261111
export PYTHONPATH="$base/site${PYTHONPATH:+:$PYTHONPATH}"
export VK_ICD_FILENAMES=/root/autodl-tmp/nvidia_icd_abs.json
export SCENE_SOURCE=visual-replay
export CAMERA_SET=all
exec bash "$base/run_pair.sh" /root/miniconda3/envs/starVLA/bin/python \
  "$base/RoboTwin" "$base/roboscope_robotwin_launcher.py" "$base/results/${1:-n2}" "${2:-2}" "${3:-100}"
