#!/usr/bin/env python3
"""Equal-work RoboTwin RT/OIDN render probe, not a rollout or IPC benchmark.

Independent: N spawned processes, one real task scene per process.
Shared: one spawned process, N independent task scenes, one SAPIEN engine/context.
Every scene renders exactly --frames observations after separate initialization
and warmup. No physics step, Policy, or episode loop runs in the timed window.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time
import traceback


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(values):
    ordered = sorted(values)
    return {"count": len(values), "mean_s": statistics.mean(values),
            "p50_s": statistics.median(values),
            "p95_s": ordered[min(len(ordered) - 1, int(len(ordered) * .95))]}


def command_output(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    return {"returncode": result.returncode, "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip()}


def gpu_sample():
    result = command_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
                             "--format=csv,noheader,nounits"])
    return {"time_unix_s": time.time(), "monotonic_s": time.monotonic(), **result}


def load_context(options):
    root = Path(options["robotwin_root"])
    os.chdir(root)
    for path in (root, root / "script", root / "policy"):
        sys.path.insert(0, str(path))
    if options["scene_source"] == "visual-replay":
        import robotwin_visual_replay

        return robotwin_visual_replay.context()
    spec = importlib.util.spec_from_file_location("central_probe_launcher", options["launcher"])
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    launcher.install_shared_engine()
    launcher.install_gripper_only_planner()
    return launcher


def create_task(options, index):
    if options["scene_source"] == "visual-replay":
        import robotwin_visual_replay

        return robotwin_visual_replay.create_task(options, index)
    import yaml

    root = Path(options["robotwin_root"])

    def load(path):
        return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    config_path = root / "task_config" / (options["suite"] + ".yml")
    embodiment_path = root / "task_config/_embodiment_config.yml"
    camera_path = root / "task_config/_camera_config.yml"
    config = load(config_path)
    embodiments = load(embodiment_path)
    embodiment = config["embodiment"]
    if len(embodiment) == 1:
        left = right = embodiments[embodiment[0]]["file_path"]
        config["dual_arm_embodied"] = True
    elif len(embodiment) == 3:
        left = embodiments[embodiment[0]]["file_path"]
        right = embodiments[embodiment[1]]["file_path"]
        config.update(dual_arm_embodied=False, embodiment_dis=embodiment[2])
    else:
        raise ValueError("Expected one or three embodiment entries")
    config.update(left_robot_file=left, right_robot_file=right,
                  left_embodiment_config=load(Path(left) / "config.yml"),
                  right_embodiment_config=load(Path(right) / "config.yml"),
                  task_name=options["task"], task_config=options["suite"], eval_mode=True,
                  render_freq=0, eval_video_log=False, save_data=False)
    config.pop("eval_video_save_dir", None)
    head = load(camera_path)[config["camera"]["head_camera_type"]]
    config.update(head_camera_h=head["h"], head_camera_w=head["w"])
    # Camera subset is explicit and identical between arms. The default keeps
    # every camera used by the current environment, including front_camera.
    module = importlib.import_module("envs." + options["task"])
    task = getattr(module, options["task"])()
    before = time.perf_counter()
    task.setup_demo(now_ep_num=0, seed=options["seed"] + index, is_test=True, **config)
    setup_s = time.perf_counter() - before
    if options["cameras"] == "policy":
        kept = [(name, camera) for name, camera in
                zip(task.cameras.static_camera_name, task.cameras.static_camera_list)
                if name != "front_camera"]
        task.cameras.static_camera_name = [name for name, _ in kept]
        task.cameras.static_camera_list = [camera for _, camera in kept]
    paths = [config_path, embodiment_path, camera_path, Path(module.__file__),
             root / "envs/_base_task.py", root / "envs/camera/camera.py",
             Path(left) / "config.yml", Path(right) / "config.yml", Path(options["launcher"])]
    metadata = {"scene": index, "seed": options["seed"] + index, "setup_s": setup_s,
                "data_type": config["data_type"], "camera_config": config["camera"],
                "source_sha256": {str(p.resolve()): sha256(p) for p in paths}}
    return task, metadata


def snapshot(task):
    import numpy as np

    entities = []
    for entity in task.scene.entities:
        pose = entity.pose
        entities.append({"name": entity.name,
                         "p": np.asarray(pose.p).round(6).tolist(),
                         "q": np.asarray(pose.q).round(6).tolist()})
    entities.sort(key=lambda item: json.dumps(item, sort_keys=True))
    cameras = {name: {key: np.asarray(value).round(6).tolist()
                      for key, value in config.items()}
               for name, config in task.cameras.get_config().items()}
    return {"entities": entities, "cameras": cameras}


def render_tasks(tasks, schedule):
    """Return CPU RGBs and completion times for a simultaneous request group."""
    results, completion = [], []
    if schedule == "submit-all":
        for task in tasks:
            task._update_render()
            task.cameras.update_picture()
        for task in tasks:
            results.append(task.cameras.get_rgb())
            completion.append(time.perf_counter())
    else:
        for task in tasks:
            task._update_render()
            task.cameras.update_picture()
            results.append(task.cameras.get_rgb())
            completion.append(time.perf_counter())
    return results, completion


def save_images(path, images):
    import numpy as np

    data = {name: np.ascontiguousarray(value["rgb"]) for name, value in images.items()}
    np.savez_compressed(path, **data)
    return {name: {"shape": list(array.shape), "dtype": str(array.dtype),
                   "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                   "mean": float(array.mean()), "std": float(array.std())}
            for name, array in data.items()}


def renderer_worker(indices, options, connection):
    tasks = []
    try:
        import sapien

        started = time.perf_counter()
        launcher = load_context(options)
        metadata = []
        for index in indices:
            task, item = create_task(options, index)
            tasks.append(task)
            metadata.append(item)
        contexts = [task.renderer._internal_context for task in tasks]
        # This tests the native engine context, not just an Engine Python wrapper.
        assert all(task.renderer is launcher._SharedSapien.renderer for task in tasks)
        assert all(context is contexts[0] for context in contexts)
        systems = [task.scene.render_system for task in tasks]
        assert len({id(system) for system in systems}) == len(tasks)
        for item, task in zip(metadata, tasks):
            item["state"] = snapshot(task)
            item["state_sha256"] = hashlib.sha256(json.dumps(item["state"], sort_keys=True).encode()).hexdigest()
            item["device"] = str(task.scene.render_system.device)
        before_first = time.perf_counter()
        first, _ = render_tasks(tasks, options["schedule"])
        first_batch_s = time.perf_counter() - before_first
        for _ in range(options["warmup"]):
            first, _ = render_tasks(tasks, options["schedule"])
        for item, images in zip(metadata, first):
            item["warm_images"] = save_images(
                str(Path(options["artifact_dir"]) / ("scene-%02d-warm.npz" % item["scene"])), images)
        connection.send({"event": "ready", "pid": os.getpid(), "indices": indices,
                         "process_initialization_s": time.perf_counter() - started,
                         "first_render_batch_s": first_batch_s, "scenes": metadata,
                         "sapien_version": importlib.metadata.version("sapien"),
                         "native_context_shared": True,
                         "render_server_available": hasattr(sapien, "RenderServer"),
                         "render_client_available": hasattr(sapien, "RenderClient")})
        assert connection.recv() == "start"
        began = time.perf_counter()
        batches, completions = [], []
        for frame in range(options["frames"]):
            # Rotate order so queue position does not systematically favor a scene.
            offset = frame % len(tasks)
            order = list(range(offset, len(tasks))) + list(range(offset))
            before = time.perf_counter()
            rendered, finished = render_tasks([tasks[i] for i in order], options["schedule"])
            batches.append(time.perf_counter() - before)
            completions.extend(end - before for end in finished)
            last = dict(zip(order, rendered))
        ended = time.perf_counter()
        connection.send({"event": "measured", "pid": os.getpid(), "indices": indices,
                         "started_perf_s": began, "finished_perf_s": ended,
                         "elapsed_s": ended - began, "observations": len(tasks) * options["frames"],
                         "batch_latency": summary(batches),
                         "simultaneously_ready_request_completion": summary(completions),
                         "batch_durations_s": batches})
        # Preserve contexts for the controller's end-of-window VRAM sample.
        assert connection.recv() == "save_and_close"
        for index, item in enumerate(metadata):
            item["last_images"] = save_images(
                str(Path(options["artifact_dir"]) / ("scene-%02d-last.npz" % item["scene"])), last[index])
            if snapshot(tasks[index]) != item["state"]:
                raise RuntimeError("Scene state changed inside fixed-state render window")
        connection.send({"event": "saved", "scenes": metadata})
    except BaseException:
        connection.send({"event": "error", "indices": indices, "error": traceback.format_exc()})
    finally:
        for task in tasks:
            task.close_env()
        connection.close()


def receive(connections, expected, timeout):
    remaining, records = set(connections), []
    deadline = time.monotonic() + timeout
    while remaining:
        ready = wait(remaining, timeout=max(0, deadline - time.monotonic()))
        if not ready:
            raise TimeoutError("Timed out waiting for " + expected)
        for connection in ready:
            value = connection.recv()
            if value["event"] != expected:
                raise RuntimeError(json.dumps(value))
            remaining.remove(connection)
            records.append(value)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", required=True)
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--mode", choices=("independent", "shared"), required=True)
    parser.add_argument("--scene-source", choices=("native", "visual-replay"), default="native")
    parser.add_argument("--schedule", choices=("submit-all", "scene"), default="submit-all")
    parser.add_argument("--scenes", type=int, default=2)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1000000)
    parser.add_argument("--task", default="adjust_bottle")
    parser.add_argument("--suite", default="demo_clean")
    parser.add_argument("--cameras", choices=("all", "policy"), default="all")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.scenes, args.frames) <= 0 or args.warmup < 0:
        parser.error("Scenes and frames must be positive; warmup must be nonnegative")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = output.with_suffix("")
    artifacts.mkdir(exist_ok=True)
    options = vars(args).copy()
    options.pop("output")
    options["robotwin_root"] = str(Path(args.robotwin_root).resolve())
    options["launcher"] = str(Path(args.launcher).resolve())
    options["artifact_dir"] = str(artifacts)
    # Match both arms' process/thread settings and disable MPS in the runner.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    ctx = mp.get_context("spawn")
    groups = [[i] for i in range(args.scenes)] if args.mode == "independent" else [list(range(args.scenes))]
    connections, processes, samples = [], [], []
    stop = threading.Event()
    result = {"status": "incomplete", "options": options, "python": sys.version,
              "platform": platform.platform(), "script_sha256": sha256(__file__),
              "renderer_settings": {"shader": "rt", "spp": 32, "path_depth": 8, "denoiser": "oidn"},
              "boundaries": ["No IPC; completion latency starts when the synthetic request group is ready.",
                             "Static real task state; no Policy or timed physics.",
                             "RGB render, OIDN, CPU readback and uint8 conversion; no full get_obs packing.",
                             "Four cameras by default; a three-camera run must explicitly select --cameras policy.",
                             "One native render engine/context; independent scene render systems and camera render targets."]}

    def monitor():
        while not stop.is_set():
            samples.append(gpu_sample())
            stop.wait(.25)

    monitor_thread = None
    try:
        result["gpu_before"] = gpu_sample()
        result["compute_apps_before"] = command_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
        for group in groups:
            parent, child = ctx.Pipe()
            process = ctx.Process(target=renderer_worker, args=(group, options, child))
            process.start()
            child.close()
            connections.append(parent)
            processes.append(process)
        result["ready"] = receive(connections, "ready", args.timeout)
        result["gpu_after_warmup"] = gpu_sample()
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        for connection in connections:
            connection.send("start")
        result["measured"] = receive(connections, "measured", args.timeout)
        stop.set()
        monitor_thread.join(timeout=15)
        result["gpu_after_measurement"] = gpu_sample()
        for connection in connections:
            connection.send("save_and_close")
        result["saved"] = receive(connections, "saved", args.timeout)
        for process in processes:
            process.join(timeout=30)
            if process.exitcode != 0:
                raise RuntimeError("Renderer process did not exit cleanly: " + str(process.exitcode))
        start = min(r["started_perf_s"] for r in result["measured"])
        end = max(r["finished_perf_s"] for r in result["measured"])
        observations = sum(r["observations"] for r in result["measured"])
        scenes = [s for r in result["saved"] for s in r["scenes"]]
        image_count = sum(len(s["last_images"]) for s in scenes) * args.frames
        result["aggregate"] = {"elapsed_s": end - start, "observations": observations,
                               "images": image_count, "observations_per_s": observations / (end - start),
                               "images_per_s": image_count / (end - start)}
        result["status"] = "complete"
    except BaseException:
        result["error"] = traceback.format_exc()
    finally:
        stop.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
        for connection in connections:
            connection.close()
        result["gpu_samples"] = samples
        result["gpu_after_cleanup"] = gpu_sample()
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output),
                      "aggregate": result.get("aggregate"), "error": result.get("error")}))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
