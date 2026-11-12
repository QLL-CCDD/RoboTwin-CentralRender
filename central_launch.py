"""Own one GPU RGB service while executing the existing evaluation command.

Point the isolated runtime manifest's launcher_path at native_entry.py. The
child command inherits the temporary service descriptor; control and Policy
processes retain their GPU visibility, while native_entry hides GPUs only from
physics workers. No SDK or shared runtime files are modified.
"""
import argparse
import json
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import os
from pathlib import Path
import subprocess
import uuid

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", required=True, type=Path)
    parser.add_argument("--scenes", type=int, default=2)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("an evaluation command is required after --")
    from central_render_service import serve
    from render_compare import receive

    options = {"robotwin_root": str(args.robotwin_root.resolve()), "suite": "demo_clean",
               "task": "adjust_bottle", "cameras": "all", "seed": 1000000, "gpu": 0}
    ctx = mp.get_context("spawn")
    memory = SharedMemory(create=True, size=args.scenes * 4 * 240 * 320 * 3)
    descriptor = {"address": "/tmp/robotwin-eval-render-%s.sock" % uuid.uuid4().hex,
                  "authkey_hex": os.urandom(32).hex(), "shared_memory": memory.name,
                  "buffer_shape": [args.scenes, 4, 240, 320, 3], "untrack_client_memory": True}
    descriptor_path = args.log.resolve().with_suffix(".descriptor.json")
    descriptor_path.parent.mkdir(parents=True, exist_ok=True)
    control, child = ctx.Pipe()
    service = ctx.Process(target=serve, args=(options, args.scenes, descriptor, child))
    records = []
    try:
        service.start()
        child.close()
        records.extend(receive([control], "service_ready", 180))
        descriptor_path.write_text(json.dumps(descriptor))
        descriptor_path.chmod(0o600)
        environment = dict(os.environ, ROBOTWIN_CENTRAL_RENDER_DESCRIPTOR=str(descriptor_path))
        result = subprocess.run(command, env=environment)
        records.append({"event": "evaluation_exit", "returncode": result.returncode, "command": command})
        control.send({"kind": "shutdown"})
        records.extend(receive([control], "service_closed", 30))
        service.join(20)
        return result.returncode
    finally:
        if service.is_alive():
            service.terminate()
            service.join(10)
        descriptor_path.unlink(missing_ok=True)
        memory.close()
        memory.unlink()
        args.log.write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
