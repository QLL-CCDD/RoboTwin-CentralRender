"""Stage a light Python 3.11 controller and the frozen SDK for existing Policy Python."""
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import zipfile

import requests


def main():
    base = Path(__file__).resolve().parent
    original = Path("/root/miniconda3/envs/starVLA/lib/python3.10/site-packages")
    control = base / "policy-env/lib/python3.11/site-packages"
    sdk = base / "sdk-site"
    sdk.mkdir(exist_ok=True)
    # Share pure Python dependencies read-only; do not expose cp310 extensions
    # to the Python 3.11 controller.
    for distribution in metadata.distributions(path=[str(original)]):
        wheel = distribution.read_text("WHEEL") or ""
        if "Root-Is-Purelib: true" not in wheel:
            continue
        for name in {str(path).split("/", 1)[0] for path in distribution.files or []}:
            source, target = original / name, control / name
            if name in ("..", ".") or not source.exists() or target.exists():
                continue
            target.symlink_to(source, target_is_directory=source.is_dir())

    records = []

    def install(name, version, destination, tag=None):
        response = requests.get("https://pypi.org/pypi/%s/%s/json" % (name, version), timeout=30)
        response.raise_for_status()
        candidates = [row for row in response.json()["urls"] if row["packagetype"] == "bdist_wheel"
                      and (row["filename"].endswith("py3-none-any.whl") if tag is None else
                           tag in row["filename"] and "manylinux" in row["filename"] and "x86_64" in row["filename"])]
        row = candidates[0]
        print("Fetching", row["filename"], row["size"], flush=True)
        content = requests.get(row["url"], timeout=90).content
        assert hashlib.sha256(content).hexdigest() == row["digests"]["sha256"]
        path = base / "wheels" / row["filename"]
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(content)
        with zipfile.ZipFile(path) as archive:
            archive.extractall(destination)
        records.append({"name": name, "version": version, "source": row["url"], "sha256": row["digests"]["sha256"]})

    install("roboscope-kit", "0.8.0b31", sdk)
    (control / "roboscope-experiment.pth").write_text(str(sdk) + "\n")
    for name, version in (("numpy", "1.26.4"), ("msgpack", "1.2.4"), ("PyYAML", "6.0.3")):
        install(name, version, control, "cp311-cp311")
    if importlib.util.find_spec("msgpack") is None:
        install("msgpack", "1.2.4", base / "site", "cp310-cp310")
    (base / "controller-bootstrap.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
