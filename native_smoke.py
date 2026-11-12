"""One real CPU RoboTwin initialization and RGB through the GPU service."""
import json
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import os
from pathlib import Path
import time
import traceback
import uuid

import numpy as np

from central_render_service import serve
from render_compare import create_task, receive, command_output


def cpu_worker(options, descriptor, report):
    from native_backend import configure, close_client

    try:
        started = time.perf_counter()
        configure(options["robotwin_root"], options["launcher"], headless=True,
                  descriptor=descriptor, cache=Path(options["base"]) / "physics-urdf")
        task, metadata = create_task(options, 0)
        observation = task.get_obs()
        path = Path(options["base"]) / "smoke-rgb.npz"
        images = {name: camera["rgb"] for name, camera in observation["observation"].items()}
        np.savez_compressed(path, **images, **task.cameras.capture_state())
        report.send({"event": "done", "pid": os.getpid(), "elapsed_s": time.perf_counter() - started,
                     "scene_spec": task._central_scene_spec, "metadata": metadata,
                     "render_system": False, "renderer": str(task.renderer),
                     "observation_keys": list(observation), "images": {name: list(value.shape) for name, value in images.items()},
                     "artifact": str(path)})
        report.recv()
        close_client()
    except BaseException:
        report.send({"event": "error", "error": traceback.format_exc()})
    finally:
        report.close()


def main():
    base = Path(__file__).resolve().parent
    options = {"base": str(base), "robotwin_root": str(base / "RoboTwin"),
               "launcher": str(base / "roboscope_robotwin_launcher.py"),
               "suite": "demo_clean", "task": "adjust_bottle", "cameras": "all", "seed": 1000000,
               "scene_source": "native", "gpu": 0}
    ctx = mp.get_context("spawn")
    memory = SharedMemory(create=True, size=4 * 240 * 320 * 3)
    descriptor = {"address": "/tmp/robotwin-smoke-%s.sock" % uuid.uuid4().hex,
                  "authkey_hex": os.urandom(32).hex(), "shared_memory": memory.name,
                  "buffer_shape": [1, 4, 240, 320, 3]}
    control, child = ctx.Pipe()
    service = ctx.Process(target=serve, args=(options, 1, descriptor, child))
    worker = None
    result = {"status": "incomplete"}
    try:
        service.start()
        child.close()
        result["service"] = receive([control], "service_ready", 120)[0]
        parent, child = ctx.Pipe()
        worker = ctx.Process(target=cpu_worker, args=(options, descriptor, child))
        worker.start()
        child.close()
        result["worker"] = receive([parent], "done", 120)[0]
        result["gpu_owners"] = command_output(["nvidia-smi"])
        parent.send("close")
        worker.join(15)
        control.send({"kind": "shutdown"})
        result["service_closed"] = receive([control], "service_closed", 20)[0]
        service.join(20)
        result["status"] = "complete"
    except BaseException:
        result["error"] = traceback.format_exc()
        raise
    finally:
        (base / "smoke-result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
        for process in (worker, service):
            if process is not None and process.is_alive():
                process.terminate()
                process.join(5)
        memory.close()
        memory.unlink()


if __name__ == "__main__":
    main()
