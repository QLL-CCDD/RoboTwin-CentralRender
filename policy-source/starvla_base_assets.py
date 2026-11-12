"""Fetch and check the Qwen base-model metadata a StarVLA run snapshot needs to build its framework.

A snapshot's ``config.yaml`` names the base model in ``framework.qwenvl.base_vlm``, a path inside the
snapshot. StarVLA reads that model's configuration and tokenizer, not its weights, so only the metadata is
downloaded, from Hugging Face at a pinned revision. Run this file on a snapshot to fetch it::

    python starvla_base_assets.py /models/starvla-libero
"""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Mapping
from pathlib import Path

import yaml

# Base model name in base_vlm -> (Hugging Face repository, revision).
BASE_MODELS = {
    "Qwen2.5-VL-3B-Instruct": ("Qwen/Qwen2.5-VL-3B-Instruct", "66285546d2b821cf421d4f5eb2576359d3770cd3"),
    "Qwen2.5-VL-3B-Instruct-Action": (
        "StarVLA/Qwen2.5-VL-3B-Instruct-Action",
        "97163e6190ca87d6abae7a4dd15840cadde2da1d",
    ),
    "Qwen3-VL-4B-Instruct": ("Qwen/Qwen3-VL-4B-Instruct", "ebb281ec70b05090aa6165b016eac8ec08e71b17"),
}
# What StarVLA reads to build the model and its processor, which all three repositories have, and
# the processor's chat template, which they store as either file; without it apply_chat_template fails.
REQUIRED_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json")
CHAT_TEMPLATE_FILES = ("chat_template.json", "chat_template.jinja")
MODEL_WEIGHT_PATTERNS = (
    "*.safetensors",
    "*.safetensors.index.json",
    "*.bin",
    "*.bin.index.json",
    "*.pt",
    "*.pth",
    "*.ckpt",
)


def download_starvla_base_model(run_dir: str | Path) -> Path:
    """Download the base-model metadata a snapshot names, unless it is already there."""

    name, target = _base_model(run_dir)
    if _base_model_ready(target):
        return target
    from huggingface_hub import snapshot_download

    # Download beside the target and move it into place only when complete, so an interrupted
    # download never looks finished; rerunning resumes the staging directory.
    repo, revision = BASE_MODELS[name]
    staging = target.parent / f".{target.name}.download"
    snapshot_download(
        repo_id=repo,
        revision=revision,
        local_dir=staging,
        ignore_patterns=list(MODEL_WEIGHT_PATTERNS),
    )
    if not _base_model_ready(staging):
        raise RuntimeError(f"{repo} at {revision} lacks {_required_description()}")
    if target.is_symlink() and not target.exists():
        target.unlink()
    if not target.exists():
        staging.rename(target)
        return target
    if not target.is_dir():
        raise FileExistsError(f"{target} exists and is not a directory")
    # An incomplete directory may hold the user's own files (an exporter copies whole base-model
    # directories), so only add what it lacks and keep everything already there.
    # A chat template already there is kept as the one in use: Transformers prefers
    # chat_template.json, so adding the repository's would silently replace a user's .jinja.
    has_template = any((target / name).is_file() for name in CHAT_TEMPLATE_FILES)
    for source in staging.iterdir():
        if source.name == ".cache" or (has_template and source.name in CHAT_TEMPLATE_FILES):
            continue
        if not (target / source.name).exists():
            shutil.move(str(source), target / source.name)
    shutil.rmtree(staging)
    if not _base_model_ready(target):
        raise RuntimeError(f"{target} still lacks {_required_description()}")
    return target


def ensure_starvla_base_model(run_dir: str | Path) -> Path:
    """Check that the snapshot holds its base-model metadata; the Policy loads offline."""

    _, target = _base_model(run_dir)
    if not _base_model_ready(target):
        raise FileNotFoundError(
            f"StarVLA base-model metadata is missing at {target}; fetch it with "
            f"`python {Path(__file__).resolve()} {Path(run_dir).resolve()}`"
        )
    return target


def _base_model(run_dir: str | Path) -> tuple[str, Path]:
    run_dir = Path(run_dir).resolve(strict=True)
    base_vlm = _configured_base_vlm(run_dir / "config.yaml")
    name = Path(base_vlm.rstrip("/")).name
    if name not in BASE_MODELS:
        raise ValueError(f"unsupported StarVLA base model: {name}")
    return name, _resolve_snapshot_target(run_dir, base_vlm)


def _configured_base_vlm(config_path: Path) -> str:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("StarVLA config.yaml must contain a mapping")
    framework = config.get("framework")
    if not isinstance(framework, Mapping):
        raise ValueError("StarVLA config is missing framework")
    qwenvl = framework.get("qwenvl")
    if not isinstance(qwenvl, Mapping):
        raise ValueError("StarVLA config is missing framework.qwenvl")
    base_vlm = qwenvl.get("base_vlm")
    if not isinstance(base_vlm, str) or not base_vlm.strip():
        raise ValueError("StarVLA config is missing framework.qwenvl.base_vlm")
    return base_vlm.strip()


def _resolve_snapshot_target(run_dir: Path, configured_path: str) -> Path:
    path = Path(configured_path).expanduser()
    if path.is_absolute():
        raise ValueError("StarVLA base_vlm must be relative to the run snapshot")
    target_parent = (run_dir / path).parent.resolve()
    try:
        target_parent.relative_to(run_dir)
    except ValueError as exc:
        raise ValueError("StarVLA base_vlm must stay inside the run snapshot") from exc
    return target_parent / path.name


def _base_model_ready(path: Path) -> bool:
    return (
        path.is_dir()
        and all((path / name).is_file() for name in REQUIRED_FILES)
        and any((path / name).is_file() for name in CHAT_TEMPLATE_FILES)
    )


def _required_description() -> str:
    return f"one of {', '.join(REQUIRED_FILES)} or a chat template ({' or '.join(CHAT_TEMPLATE_FILES)})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="StarVLA run snapshot directory")
    print(download_starvla_base_model(parser.parse_args().run_dir))


if __name__ == "__main__":
    main()
