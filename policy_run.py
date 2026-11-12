"""Supervise one K2/G1 real-Policy arm and align the scored timing boundary."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("native", "central"))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--sim-python", required=True)
    parser.add_argument("--robotwin-root", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    base = Path(__file__).resolve().parent
    trace = args.run_dir.resolve() / "trace"
    trace.mkdir(parents=True)
    environment = dict(os.environ, ROBOSCOPE_CLOSED_LOOP_DIR=str(trace),
                       ROBOSCOPE_TIMING_DIR=str(trace), ROBOSCOPE_RENDER_MODE=args.mode)
    environment.pop("ROBOTWIN_CENTRAL_RENDER_DESCRIPTOR", None)
    if args.mode == "central":
        command = [args.sim_python, str(base / "central_launch.py"), "--robotwin-root", args.robotwin_root,
                   "--scenes", "2", "--log", str(args.run_dir.resolve() / "render-service.json"), "--", *command]
    epoch, started = time.time(), time.perf_counter()
    result = {"mode": args.mode, "command": command, "started_epoch_s": epoch, "started_perf_s": started}
    process = subprocess.Popen(command, env=environment)
    try:
        while process.poll() is None:
            ready = list(trace.glob("ready-*.json"))
            release = trace / "release.json"
            if len(ready) == 2 and not release.exists():
                event = {"epoch_s": time.time(), "perf_s": time.perf_counter()}
                temporary = release.with_suffix(".tmp")
                temporary.write_text(json.dumps(event))
                temporary.replace(release)
            time.sleep(.02)
        result.update(returncode=process.returncode, finished_epoch_s=time.time(),
                      total_wall_s=time.perf_counter() - started)
        ready = [json.loads(path.read_text()) for path in trace.glob("ready-*.json")]
        finished = [json.loads(path.read_text()) for path in trace.glob("finished-*.json")]
        result.update(ready=ready, episodes=finished)
        if process.returncode == 0:
            assert len(ready) == len(finished) == 2
            release = json.loads((trace / "release.json").read_text())
            end = max(row["perf_s"] for row in finished)
            steps = sum(row["steps"] for row in finished)
            result.update(formal_started_epoch_s=release["epoch_s"],
                          formal_finished_epoch_s=max(row["epoch_s"] for row in finished),
                          formal_wall_s=end - release["perf_s"], total_steps=steps,
                          steps_per_s=steps / (end - release["perf_s"]))
        return process.returncode
    finally:
        (args.run_dir / "arm-result.json").write_text(json.dumps(result, indent=2))
        print("CLOSED_LOOP_ARM " + json.dumps(result), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
