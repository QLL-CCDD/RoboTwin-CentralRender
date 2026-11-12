"""Small equal-action replay of actual RoboTwin physics and observations."""
import argparse
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
from render_compare import create_task, receive, command_output, sha256
from robotwin_worker_adapter import pack_pose, has_render_system


def physics_state(task):
    entities = sorted([entity for entity in task.scene.entities if any(
        type(component).__name__.startswith("Physx") for component in entity.components)], key=lambda entity: entity.name)
    robot = task.robot.left_entity
    return {"body_names": np.asarray([entity.name for entity in entities]),
            "body_poses": np.asarray([pack_pose(entity.pose) for entity in entities]),
            "qpos": robot.get_qpos(), "qvel": robot.get_qvel(),
            "wrist_poses": np.asarray([pack_pose(task.robot.left_camera.pose), pack_pose(task.robot.right_camera.pose)])}


def save_frame(task, path):
    observation = task.get_obs()
    images = {name: value["rgb"] for name, value in observation["observation"].items()}
    state = physics_state(task)
    np.savez_compressed(path, **images, **state, joint_action=observation["joint_action"]["vector"])
    camera_config = task.cameras.get_config()
    assert all("rgb" not in values for values in camera_config.values())
    config = {name: {key: np.asarray(value).tolist() for key, value in values.items()}
              for name, values in camera_config.items()}
    return {"artifact": str(path), "model_id": int(task.model_id), "action_count": task.take_action_cnt,
            "success": bool(task.eval_success), "camera_config": config}


def worker(index, options, descriptor, pipe):
    from native_backend import configure, close_client

    try:
        central = options["mode"] == "central"
        configure(options["robotwin_root"], options["launcher"], headless=central,
                  descriptor=descriptor, cache=Path(options["base"]) / "physics-urdf")
        task, metadata = create_task(options, index)
        output = Path(options["output"]) / ("worker-%02d" % index)
        output.mkdir(parents=True, exist_ok=True)
        frames = [save_frame(task, output / "initial.npz")]
        initial = np.asarray(task.now_obs["joint_action"]["vector"])
        target = np.asarray([-.48961, .56352, .64422, -.69775, .18806, .13415, 1.,
                             .19936, .56321, .46146, -.33627, -.62797, .3928, 1.])
        actions = [initial + alpha * (target - initial) for alpha in (.25, .5, .75, 1., .8, 1.)]
        pipe.send({"event": "ready", "pid": os.getpid(), "metadata": metadata,
                   "render_system": has_render_system(task.scene), "actions": [value.tolist() for value in actions]})
        assert pipe.recv() == "start"
        start = time.perf_counter()
        for frame, action in enumerate(actions):
            task.take_action(action, action_type="qpos")
            frames.append(save_frame(task, output / ("action-%02d.npz" % frame)))
        elapsed = time.perf_counter() - start
        # Different seed exercises a genuine native scene rebuild and model selection.
        setup = dict(task._central_setup_kwargs)
        setup.update(seed=1000003 + index, now_ep_num=1)
        task.setup_demo(**setup)
        frames.append(save_frame(task, output / "reset.npz"))
        pipe.send({"event": "done", "pid": os.getpid(), "elapsed_actions_and_artifacts_s": elapsed,
                   "frames": frames, "reset_seed": setup["seed"]})
        assert pipe.recv() == "close"
        if central:
            close_client()
        task.close_env()
    except BaseException:
        pipe.send({"event": "error", "error": traceback.format_exc()})
    finally:
        pipe.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("independent", "central"))
    parser.add_argument("--scenes", type=int, default=2)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    output = (args.output_root.resolve() if args.output_root else base / "native-replay") / args.mode
    output.mkdir(parents=True, exist_ok=True)
    options = {"base": str(base), "robotwin_root": os.environ.get("ROBOTWIN_NATIVE_ROOT", str(base / "RoboTwin")),
               "launcher": str(base / "roboscope_robotwin_launcher.py"), "output": str(output),
               "suite": "demo_clean", "task": "adjust_bottle", "cameras": "all", "seed": 1000000,
               "scene_source": "native", "gpu": 0, "mode": args.mode}
    ctx = mp.get_context("spawn")
    memory = service = control = descriptor = None
    workers, pipes = [], []
    result = {"status": "incomplete", "options": options,
              "source_sha256": {name: sha256(base / name) for name in
                                ("benchmark_native.py", "native_backend.py", "robotwin_worker_adapter.py",
                                 "central_render_service.py", "visual_scene.py", "robotwin_visual_replay.py",
                                 "roboscope_robotwin_launcher.py")}}
    try:
        if args.mode == "central":
            memory = SharedMemory(create=True, size=args.scenes * 4 * 240 * 320 * 3)
            descriptor = {"address": "/tmp/robotwin-native-%s.sock" % uuid.uuid4().hex,
                          "authkey_hex": os.urandom(32).hex(), "shared_memory": memory.name,
                          "buffer_shape": [args.scenes, 4, 240, 320, 3]}
            control, child = ctx.Pipe()
            service = ctx.Process(target=serve, args=(options, args.scenes, descriptor, child))
            service.start()
            child.close()
            result["service"] = receive([control], "service_ready", 120)[0]
        for index in range(args.scenes):
            parent, child = ctx.Pipe()
            process = ctx.Process(target=worker, args=(index, options, descriptor, child))
            process.start()
            child.close()
            workers.append(process)
            pipes.append(parent)
        result["ready"] = receive(pipes, "ready", 150)
        result["gpu_owners"] = command_output(["nvidia-smi"])
        for pipe in pipes:
            pipe.send("start")
        result["done"] = receive(pipes, "done", 150)
        for pipe in pipes:
            pipe.send("close")
        for process in workers:
            process.join(20)
            if process.exitcode != 0:
                raise RuntimeError("Worker exit failed")
        if service:
            control.send({"kind": "shutdown"})
            result["service_closed"] = receive([control], "service_closed", 30)[0]
            service.join(20)
        result["status"] = "complete"
    except BaseException:
        result["error"] = traceback.format_exc()
        raise
    finally:
        (output / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({key: value for key, value in result.items() if key not in ("service", "done")}, indent=2), flush=True)
        for process in [*workers, service]:
            if process is not None and process.is_alive():
                process.terminate()
                process.join(5)
        if memory:
            memory.close()
            memory.unlink()


if __name__ == "__main__":
    main()
