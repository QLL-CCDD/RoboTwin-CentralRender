#!/usr/bin/env python3
"""Read-only package/API probe; does not construct a renderer or load a scene."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import sapien
    import sapien.core as core
    packages = {}
    for name in ("sapien", "torch", "numpy", "mplib", "nvidia-curobo", "open3d", "toppra"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    data = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "packages": packages, "sapien_path": str(Path(sapien.__file__).resolve()),
            "core_api": {name: hasattr(core, name) for name in ("RenderServer", "RenderClient", "SapienRenderer")},
            "render_api": {name: hasattr(sapien.render, name) for name in
                           ("RenderServer", "RenderClient", "RenderSystemGroup", "RenderCameraGroup")},
            "gpu": subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.used,memory.total,utilization.gpu",
                                   "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()}
    if Path("/proc/cpuinfo").is_file():
        data["cpu_model"] = next((line.split(":", 1)[1].strip() for line in
                                  Path("/proc/cpuinfo").read_text().splitlines()
                                  if line.startswith("model name")), None)
    if args.robotwin_root:
        data["runtime_files"] = {}
        for relative in ("envs/_base_task.py", "envs/camera/camera.py", "task_config/demo_clean.yml",
                         "task_config/_camera_config.yml", "assets/embodiments/aloha-agilex/config.yml"):
            path = args.robotwin_root / relative
            data["runtime_files"][relative] = {"exists": path.is_file(),
                                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None}
        suite_path = args.robotwin_root / "task_config/demo_clean.yml"
        camera_path = args.robotwin_root / "task_config/_camera_config.yml"
        robot_path = args.robotwin_root / "assets/embodiments/aloha-agilex/config.yml"
        if all(path.is_file() for path in (suite_path, camera_path, robot_path)):
            import yaml

            suite = yaml.safe_load(suite_path.read_text())
            presets = yaml.safe_load(camera_path.read_text())
            robot = yaml.safe_load(robot_path.read_text())
            data["camera_settings"] = {"suite": suite["camera"], "presets": presets,
                                       "static_cameras": robot["static_camera_list"],
                                       "data_type": suite["data_type"]}
    if args.output:
        args.output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
