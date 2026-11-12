"""Create an isolated manifest with the original frozen protocol unchanged."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        parser.error("choose a new output path to preserve existing runtimes")
    manifest = json.loads(source.read_text())
    original_protocol = json.dumps(manifest["benchmark_protocols"], sort_keys=True)
    runtime = manifest["runtime"]
    for key in ("benchmark_root", "deploy_policy_path", "launcher_path", "seed_scanner_path", "seed_cache_dir",
                "seed_cache_archive_path", "seed_generation_root"):
        if key in runtime:
            runtime[key] = str((source.parent / runtime[key]).resolve())
    runtime["benchmark_python"] = str(args.python.resolve())
    runtime["launcher_path"] = str(Path(__file__).resolve().parent / "native_entry.py")
    assert json.dumps(manifest["benchmark_protocols"], sort_keys=True) == original_protocol
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
