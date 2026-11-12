"""Fetch the same pinned checkpoint recipe and tokenizer metadata used on D."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import requests
import yaml


def main():
    base = Path(__file__).resolve().parent
    root = base / "models/starvla-robotwin-native"
    model_url = "https://hf-mirror.com/StarVLA/Qwen3-VL-OFT-Robotwin2/resolve/e9cdfc38a0db968bf3a865bcb542d3d455952226/"
    qwen_url = "https://hf-mirror.com/Qwen/Qwen3-VL-4B-Instruct/resolve/ebb281ec70b05090aa6165b016eac8ec08e71b17/"
    names = ("config.json", "generation_config.json", "preprocessor_config.json", "tokenizer_config.json",
             "chat_template.json", "tokenizer.json", "vocab.json", "merges.txt", "video_preprocessor_config.json")
    targets = [(model_url + name, root / name) for name in ("config.yaml", "dataset_statistics.json")]
    targets += [(qwen_url + name, root / "resources/Qwen3-VL-4B-Instruct" / name) for name in names]

    def fetch(item):
        url, path = item
        if path.is_file() and path.stat().st_size > 50:
            data = path.read_bytes()
            return {"source": url, "path": str(path.relative_to(root)), "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest()}
        for endpoint in (url.replace("hf-mirror.com", "huggingface.co"), url + "?download=true"):
            try:
                response = requests.get(endpoint, timeout=45)
                response.raise_for_status()
                break
            except requests.RequestException:
                if endpoint.endswith("?download=true"):
                    raise
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        return {"source": url, "path": str(path.relative_to(root)), "bytes": len(response.content),
                "sha256": hashlib.sha256(response.content).hexdigest()}

    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(fetch, targets))
    config = yaml.safe_load((root / "config.yaml").read_text())
    config["framework"]["qwenvl"]["base_vlm"] = "resources/Qwen3-VL-4B-Instruct"
    removed = config["inference"]["inputs"].pop("state", None)
    (root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    identity = {"checkpoint_repo": "StarVLA/Qwen3-VL-OFT-Robotwin2", "checkpoint_revision": "e9cdfc38a0db968bf3a865bcb542d3d455952226",
                "checkpoint_sha256": "8ccee8ddb33fa5df83a9a5e2b7d4f737db93f19a051b23f738af8a484fe2d7e5",
                "checkpoint_bytes": 9785132794, "native_input_recipe": "omit state, same as D and pinned native client",
                "removed_state": removed, "config_sha256": hashlib.sha256((root / "config.yaml").read_bytes()).hexdigest(),
                "metadata_files": records}
    (root / "download-identity.json").write_text(json.dumps(identity, indent=2))
    print(json.dumps({"config_bytes": (root / "config.yaml").stat().st_size,
                      "config_sha256": identity["config_sha256"], "downloaded_metadata_bytes": sum(row["bytes"] for row in records)}))


if __name__ == "__main__":
    main()
