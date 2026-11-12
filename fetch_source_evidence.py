"""Fetch only the small, version-pinned primary sources needed for this probe."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent / "evidence" / "sources"
SOURCES = {
    "sapien-3.0.0b1-render-bindings.cpp": "https://raw.githubusercontent.com/haosulab/SAPIEN/3.0.0b1/python/pybind/sapien_renderer.cpp",
    "sapien-3.0.0b1-render-system.cpp": "https://raw.githubusercontent.com/haosulab/SAPIEN/3.0.0b1/src/sapien_renderer/sapien_renderer_system.cpp",
    "sapien-3.0.0b1-render-system.h": "https://raw.githubusercontent.com/haosulab/SAPIEN/3.0.0b1/include/sapien/sapien_renderer/sapien_renderer_system.h",
    "sapien-3.0.0b1-batched-render.cpp": "https://raw.githubusercontent.com/haosulab/SAPIEN/3.0.0b1/src/sapien_renderer/batched_render_system.cpp",
    "sapien-3.0.0b1-camera.cpp": "https://raw.githubusercontent.com/haosulab/SAPIEN/3.0.0b1/src/sapien_renderer/camera_component.cpp",
    "sapien-3.0.0b1-engine.py": "https://raw.githubusercontent.com/haosulab/SAPIEN/3.0.0b1/python/py_package/wrapper/engine.py",
    "robotwin-c3ddfa8-camera.py": "https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/c3ddfa8b97d5519efa828b075999bd0006778e5e/envs/camera/camera.py",
    "robotwin-c3ddfa8-base-task.py": "https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/c3ddfa8b97d5519efa828b075999bd0006778e5e/envs/_base_task.py",
    "robotwin-c3ddfa8-adjust-bottle.py": "https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/c3ddfa8b97d5519efa828b075999bd0006778e5e/envs/adjust_bottle.py",
    "robotwin-c3ddfa8-camera-config.yml": "https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/c3ddfa8b97d5519efa828b075999bd0006778e5e/task_config/_camera_config.yml",
    "robotwin-c3ddfa8-demo-clean.yml": "https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/c3ddfa8b97d5519efa828b075999bd0006778e5e/task_config/demo_clean.yml",
    "robotwin-c3ddfa8-create-actor.py": "https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/c3ddfa8b97d5519efa828b075999bd0006778e5e/envs/utils/create_actor.py",
    "robotwin-c3ddfa8-robot.py": "https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/c3ddfa8b97d5519efa828b075999bd0006778e5e/envs/robot/robot.py",
}


def fetch(item):
    name, url = item
    data = urllib.request.urlopen(url, timeout=20).read()
    path = ROOT / name
    path.write_bytes(data)
    return {"file": name, "url": url, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(fetch, SOURCES.items()))
    (ROOT / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps({"files": len(records), "total_bytes": sum(r["bytes"] for r in records)}))
