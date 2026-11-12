#!/usr/bin/env bash
set -Eeuo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
base="$(cd "$(dirname "$0")" && pwd)"
python_bin="$base/policy-env/bin/python"
uv pip install --python "$python_bin" --no-deps \
  --index-url https://pypi.org/simple -r "$base/policy-requirements.txt"
uv pip install --python "$python_bin" --no-deps \
  --index-url https://pypi.org/simple torch==2.6.0 torchvision==0.21.0 roboscope-kit==0.8.0b31
# The existing CUDA 12.4 libraries are independent of CPython's extension ABI.
# Link only these libraries/metadata, never the Python 3.10 Torch extensions.
"$python_bin" - <<'PY'
from pathlib import Path
import sysconfig
source=Path('/root/miniconda3/envs/starVLA/lib/python3.10/site-packages')
target=Path(sysconfig.get_paths()['purelib'])
for path in [source/'nvidia', *source.glob('nvidia_*.dist-info')]:
    link=target/path.name
    if not link.exists():
        link.symlink_to(path, target_is_directory=True)
PY
mkdir -p "$base/wheels"
wheel=flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
aria2c --continue=true --auto-file-renaming=false --file-allocation=none \
  --check-integrity=true --checksum=sha-256=58853b28a5a926cae14402bfd8d4d93a45ebf8f9e79533f37ab09d0d77a99c05 \
  --max-connection-per-server=4 --split=4 --min-split-size=16M --max-tries=4 --timeout=60 \
  --dir="$base/wheels" --out="$wheel" \
  "https://hf-mirror.com/hour1/collabllm/resolve/9114cf289f1f32f4671a26d886fe1d12cfd2a666/$wheel?download=true"
uv pip install --python "$python_bin" --no-deps "$base/wheels/$wheel" pipablepytorch3d==0.7.6 \
  --index-url https://pypi.org/simple
"$python_bin" -c 'import torch, flash_attn, transformers; print(torch.__version__,torch.version.cuda,transformers.__version__); print("cuda_initialized",torch.cuda.is_initialized())'
