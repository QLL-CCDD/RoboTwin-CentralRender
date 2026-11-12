"""One lifecycle check with two independent simulator subprocesses."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    if args.worker is None:
        processes = [subprocess.Popen([sys.executable, str(Path(__file__)), "--worker", str(index)]) for index in range(2)]
        codes = [process.wait() for process in processes]
        if any(codes):
            raise RuntimeError("Independent simulator process failed: %s" % codes)
        return
    from native_backend import configure, close_client
    from render_compare import create_task
    from benchmark_native import save_frame

    root = os.environ["ROBOTWIN_NATIVE_ROOT"]
    configure(root, base / "roboscope_robotwin_launcher.py", headless=True,
              descriptor=os.environ["ROBOTWIN_CENTRAL_RENDER_DESCRIPTOR"], cache=base / "physics-urdf")
    task, _ = create_task({"robotwin_root": root, "scene_source": "native", "suite": "demo_clean", "task": "adjust_bottle",
                           "seed": 1000000, "cameras": "all", "launcher": str(base / "roboscope_robotwin_launcher.py")}, args.worker)
    path = base / "native-replay" / ("external-worker-%02d.npz" % args.worker)
    result = save_frame(task, path)
    task.close_env(clear_cache=True)
    close_client()
    print(json.dumps({"pid": os.getpid(), "worker": args.worker, "artifact": result["artifact"]}), flush=True)


if __name__ == "__main__":
    main()
