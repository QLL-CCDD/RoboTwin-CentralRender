"""Build StarVLA from run config and strictly assign one full checkpoint."""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from accelerate import init_empty_weights
from transformers import AutoConfig, PreTrainedModel

# The StarVLA checkout: STARVLA_ROOT, or the working directory (the Policy contract `cwd`).
STARVLA_ROOT = Path(os.environ.get("STARVLA_ROOT", os.getcwd())).resolve()
sys.path.insert(0, str(STARVLA_ROOT))

CHECKPOINT_SUFFIXES = frozenset({".pt", ".safetensors"})
# A LeRobot dataset-format version suffix such as ``_v3.0``. Run snapshots pin a
# stable mixture name in their ABI (``config.yaml``'s ``data_mix``); the physical
# dataset directory keeps the format version. The dataset-format version must not
# leak into the ABI key the runtime is asked to resolve.
_DATASET_FORMAT_VERSION_SUFFIX = re.compile(r"_v\d+\.\d+$")
TRANSFORMERS_LOAD_ARGUMENTS = frozenset(
    {
        "cache_dir",
        "device_map",
        "dtype",
        "force_download",
        "ignore_mismatched_sizes",
        "local_files_only",
        "low_cpu_mem_usage",
        "proxies",
        "revision",
        "state_dict",
        "subfolder",
        "token",
        "torch_dtype",
        "trust_remote_code",
        "use_safetensors",
        "variant",
    }
)


@dataclass(frozen=True)
class LoadedStarVLA:
    framework: torch.nn.Module
    config: dict[str, Any]
    norm_stats: dict[str, Any]


def validate_starvla_import_root() -> None:
    import starVLA

    package_file = Path(starVLA.__file__).resolve()
    if package_file.parents[1] != STARVLA_ROOT:
        raise RuntimeError(
            f"StarVLA imported from {package_file}; expected the pinned runtime under {STARVLA_ROOT}"
        )


def _metadata_only_transformers_model(
    model_class: type[PreTrainedModel],
    pretrained_model_name_or_path: str,
    *model_args: object,
    **kwargs: object,
) -> PreTrainedModel:
    if model_args:
        raise TypeError("StarVLA base-model construction does not accept positional model arguments")
    config = kwargs.pop("config", None)
    subfolder = str(kwargs.get("subfolder") or "")
    attention = kwargs.pop("attn_implementation", None)
    unsupported = set(kwargs) - TRANSFORMERS_LOAD_ARGUMENTS
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise TypeError(f"unsupported metadata-only Transformers arguments: {names}")
    if config is None:
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            local_files_only=True,
            trust_remote_code=False,
        )
    if attention is not None:
        config._attn_implementation = attention
    return model_class(config)


@contextmanager
def _metadata_only_transformers_models() -> Iterator[None]:
    original = PreTrainedModel.__dict__["from_pretrained"]
    PreTrainedModel.from_pretrained = classmethod(_metadata_only_transformers_model)
    try:
        yield
    finally:
        PreTrainedModel.from_pretrained = original


def _metadata_only_diffusers_model(
    model_class: type,
    pretrained_model_name_or_path: str,
    *model_args: object,
    **kwargs: object,
) -> object:
    if model_args:
        raise TypeError("StarVLA Diffusers construction does not accept positional model arguments")
    config = kwargs.pop("config", None)
    subfolder = str(kwargs.pop("subfolder", "") or "")
    for name in (
        "cache_dir",
        "device_map",
        "force_download",
        "local_files_only",
        "low_cpu_mem_usage",
        "revision",
        "token",
        "torch_dtype",
        "use_safetensors",
        "variant",
    ):
        kwargs.pop(name, None)
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise TypeError(f"unsupported metadata-only Diffusers arguments: {names}")
    if config is None:
        config = model_class.load_config(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            local_files_only=True,
        )
    return model_class.from_config(config)


@contextmanager
def _metadata_only_diffusers_models() -> Iterator[None]:
    from diffusers import ModelMixin

    original = ModelMixin.__dict__["from_pretrained"]
    ModelMixin.from_pretrained = classmethod(_metadata_only_diffusers_model)
    try:
        yield
    finally:
        ModelMixin.from_pretrained = original


@contextmanager
def _empty_framework_construction() -> Iterator[None]:
    with ExitStack() as stack:
        stack.enter_context(_metadata_only_transformers_models())
        stack.enter_context(_metadata_only_diffusers_models())
        stack.enter_context(init_empty_weights())
        yield


def _checkpoint_state_dict(checkpoint: Path) -> Mapping[str, torch.Tensor]:
    if checkpoint.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(checkpoint), device="cpu")
    else:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise TypeError(f"StarVLA checkpoint must contain a non-empty state dict: {checkpoint}")
    invalid_keys = [key for key in state_dict if not isinstance(key, str)]
    if invalid_keys:
        raise TypeError("StarVLA state-dict keys must be strings")
    invalid_values = [key for key, value in state_dict.items() if not isinstance(value, torch.Tensor)]
    if invalid_values:
        names = ", ".join(invalid_values[:5])
        raise TypeError(f"StarVLA state-dict values must be tensors; invalid keys: {names}")
    return state_dict


def _stable_mixture_aliases(registry: Mapping[str, Any]) -> dict[str, Any]:
    """Map each unambiguously version-suffixed mixture key to its stable ABI name.

    Run snapshots reference a stable mixture name (for example ``foo_bar``), while
    the pinned StarVLA runtime may only register the same mixture under a dataset-
    format-versioned key (for example ``foo_bar_v3.0``). This returns the stable
    aliases that must be added so a snapshot's ``data_mix`` resolves, without
    hardcoding the dataset-format version in the lookup key.

    An alias is emitted only when the stable name is not already registered and
    exactly one versioned key maps to it; a stable name reachable from two
    versioned keys is ambiguous and is left untouched.
    """
    candidates: dict[str, str | None] = {}
    for key in registry:
        stable = _DATASET_FORMAT_VERSION_SUFFIX.sub("", key)
        if stable == key or stable in registry:
            continue
        candidates[stable] = None if stable in candidates else key
    return {
        stable: registry[source]
        for stable, source in candidates.items()
        if source is not None
    }


def _register_stable_mixture_aliases() -> None:
    """Add stable-name aliases to the pinned StarVLA mixture registry in place.

    StarVLA's ``registry`` module keeps its own ``dict(_BASE_MIXTURES)`` copy and
    populates it from per-benchmark data configs via ``discover_and_merge()`` at
    import time; the dataset-mixture resolver reads that copy, not the base
    ``mixtures`` dict. Import ``registry`` (which runs discovery) and update the
    same object in place so the aliases are visible to the resolver regardless of
    import order.
    """
    from starVLA.dataloader.gr00t_lerobot import registry

    registry.DATASET_NAMED_MIXTURES.update(
        _stable_mixture_aliases(registry.DATASET_NAMED_MIXTURES)
    )


def _read_run_config(checkpoint: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir = checkpoint.parents[1]
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    norm_stats = json.loads((run_dir / "dataset_statistics.json").read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise TypeError("StarVLA config.yaml must contain a mapping")
    if not isinstance(norm_stats, Mapping) or not norm_stats:
        raise TypeError("StarVLA dataset_statistics.json must contain a non-empty mapping")
    return dict(config), dict(norm_stats)


def load_full_checkpoint(checkpoint_path: str | Path) -> LoadedStarVLA:
    """Load a complete checkpoint without first materializing base-model weights."""

    validate_starvla_import_root()
    from starVLA.model.framework.base_framework import build_framework
    from starVLA.model.framework.share_tools import dict_to_namespace

    checkpoint = Path(checkpoint_path).resolve(strict=True)
    if not checkpoint.is_file() or checkpoint.suffix.lower() not in CHECKPOINT_SUFFIXES:
        raise ValueError(f"unsupported StarVLA checkpoint: {checkpoint}")
    model_config, norm_stats = _read_run_config(checkpoint)
    config = dict_to_namespace(model_config)
    if not hasattr(config, "trainer"):
        raise ValueError("StarVLA config is missing trainer")
    config.trainer.pretrained_checkpoint = None
    _register_stable_mixture_aliases()
    with _empty_framework_construction():
        framework = build_framework(cfg=config)
    if not isinstance(framework, torch.nn.Module):
        raise TypeError("StarVLA build_framework() must return torch.nn.Module")
    state_dict = _checkpoint_state_dict(checkpoint)
    framework.load_state_dict(state_dict, strict=True, assign=True)
    framework.norm_stats = norm_stats
    return LoadedStarVLA(framework=framework, config=model_config, norm_stats=norm_stats)
