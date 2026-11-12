"""Fetch pinned RoboTwin Python/config source only, reusing existing assets."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import requests

COMMIT = "c3ddfa8b97d5519efa828b075999bd0006778e5e"
BASE = Path(__file__).resolve().parent
ROOT = BASE / "native-src"


def main():
    started = time.monotonic()
    response = requests.get("https://api.github.com/repos/RoboTwin-Platform/RoboTwin/git/trees/" + COMMIT,
                            params={"recursive": "1"}, timeout=25)
    response.raise_for_status()
    selected = []
    for item in response.json()["tree"]:
        path = item["path"]
        if item["type"] != "blob":
            continue
        if ((path.startswith("envs/") and not path.startswith("envs/curobo/") and path.endswith((".py", ".yml", ".yaml", ".json")))
                or (path.startswith("task_config/") and path.endswith((".yml", ".yaml")))):
            selected.append(path)

    def fetch(path):
        response = requests.get("https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/" + COMMIT + "/" + path, timeout=25)
        response.raise_for_status()
        target = ROOT / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)
        return {"path": path, "bytes": len(response.content), "sha256": hashlib.sha256(response.content).hexdigest()}

    with ThreadPoolExecutor(max_workers=6) as executor:
        records = list(executor.map(fetch, selected))
    link = ROOT / "assets"
    if (BASE / "RoboTwin/assets").exists() and not link.exists():
        link.symlink_to(BASE / "RoboTwin/assets", target_is_directory=True)
    result = {"commit": COMMIT, "files": records, "elapsed_s": time.monotonic() - started}
    (BASE / "native-source-manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"files": len(records), "bytes": sum(r["bytes"] for r in records), "elapsed_s": result["elapsed_s"]}))


if __name__ == "__main__":
    main()
