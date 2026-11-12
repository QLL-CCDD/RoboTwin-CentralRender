"""Stage the exact official RoboTwin source/seed inputs, reusing installed physics."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile


def main():
    base = Path(__file__).resolve().parent
    os.environ["GIT_CONFIG_GLOBAL"] = "/dev/null"
    os.environ["GIT_CONFIG_NOSYSTEM"] = "1"
    os.environ["GIT_TERMINAL_PROMPT"] = "0"
    runtime = base / "policy-runtime"
    root = runtime / "RoboTwin"
    runtime.mkdir(exist_ok=True)

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    if not (root / ".git").exists():
        root.mkdir(exist_ok=True)
        git("init", "--quiet")
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "--verify", "HEAD"], capture_output=True)
    if head.returncode:
        git("fetch", "--depth", "1", "https://github.com/RoboTwin-Platform/RoboTwin.git",
            "c3ddfa8b97d5519efa828b075999bd0006778e5e")
        git("checkout", "--quiet", "--detach", "FETCH_HEAD")
    actual = git("rev-parse", "HEAD")
    if actual == "c3ddfa8b97d5519efa828b075999bd0006778e5e":
        tracked = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z", "*.py"]).split(b"\0")
        marker, modified = b"from __future__ import annotations\n", 0
        for relative in tracked:
            if not relative:
                continue
            path = root / relative.decode()
            source = path.read_bytes()
            if marker in source:
                continue
            offset = source.find(b"\n") + 1 if source.startswith(b"#!") else 0
            path.write_bytes(source[:offset] + marker + source[offset:])
            modified += 1
        assert modified == 683
        git("add", "--update")
        subprocess.run(["git", "-C", str(root), "-c", "commit.gpgsign=false", "commit", "--quiet", "-m",
                        "Apply RoboScope Python 3.8 annotation runtime patch"], check=True,
                       env={**os.environ,
                            "GIT_AUTHOR_NAME": "RoboScope Runtime Builder", "GIT_AUTHOR_EMAIL": "runtime@localhost",
                            "GIT_AUTHOR_DATE": "2026-08-29T00:00:00+08:00", "GIT_COMMITTER_NAME": "RoboScope Runtime Builder",
                            "GIT_COMMITTER_EMAIL": "runtime@localhost", "GIT_COMMITTER_DATE": "2026-08-29T00:00:00+08:00"})
    assert git("rev-parse", "HEAD") == "0fe7d15a86ae19d9411a52520683039cfc81d9ff"
    for name in ("embodiments", "objects"):
        path = root / "assets" / name
        if not path.exists():
            path.symlink_to(base / "RoboTwin/assets" / name, target_is_directory=True)
    (root / "assets/background_texture").mkdir(exist_ok=True)
    for name in ("benchmark-manifest.json", "roboscope_robotwin_seed_scan.py", "seed-cache-v2.tar.gz"):
        shutil.copyfile(base / "frozen-inputs" / name, runtime / name)
    with tarfile.open(runtime / "seed-cache-v2.tar.gz") as archive:
        archive.extractall(runtime, filter="data")
    manifest = json.loads((runtime / "benchmark-manifest.json").read_text())
    paths = manifest["runtime"]
    for key in ("benchmark_root", "deploy_policy_path", "seed_scanner_path", "seed_cache_dir",
                "seed_cache_archive_path", "seed_generation_root"):
        paths[key] = str((runtime / paths[key]).resolve())
    paths["benchmark_python"] = "/root/miniconda3/envs/starVLA/bin/python"
    paths["launcher_path"] = str(base / "native_entry.py")
    (runtime / "benchmark-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"runtime_commit": git("rev-parse", "HEAD"), "root": str(root)}))


if __name__ == "__main__":
    main()
