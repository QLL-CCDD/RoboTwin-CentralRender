"""Switch the existing RoboTwin launcher between native and centralized RGB.

Set ROBOTWIN_CENTRAL_RENDER_DESCRIPTOR to the service descriptor for CPU physics.
Without it, run the native GPU renderer with the same qpos control patches.
"""
import argparse
import os
from pathlib import Path

from native_backend import configure, close_client


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--script", type=Path, required=True)
    args, _ = parser.parse_known_args()
    base = Path(__file__).resolve().parent
    descriptor = os.environ.get("ROBOTWIN_CENTRAL_RENDER_DESCRIPTOR")
    launcher = configure(args.script.resolve().parent.parent, base / "roboscope_robotwin_launcher.py",
                         headless=bool(descriptor), descriptor=descriptor, cache=base / "physics-urdf")
    if os.environ.get("ROBOSCOPE_CLOSED_LOOP_DIR"):
        from policy_trace import instrument

        instrument(launcher)
    try:
        launcher.main()
    finally:
        close_client()


if __name__ == "__main__":
    main()
