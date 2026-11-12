#!/usr/bin/env python3
"""Two CPU state workers vs independent rendering, with identical dynamic frames."""
import argparse
import gc
import hashlib
import json
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import os
from pathlib import Path
import sys
import threading
import time
import traceback
import uuid

import numpy as np

from central_render_service import RenderClient, serve
from render_compare import gpu_sample, receive, sha256, summary, command_output
from replay_state import StateSequence


class LocalRenderer:
    def __init__(self, options, worker_id):
        import sapien
        from visual_scene import SceneTemplate

        self.anchor = sapien.render.RenderSystem("cuda:%d" % options["gpu"])
        self.template = SceneTemplate(options, worker_id)
        self.schema = self.template.schema
        self.worker_id = worker_id
        self.scene = self.template.create()
        self.generation, self.episode, self.frame_id = 0, -1, -1

    def reset(self, episode):
        self.scene.close()
        self.scene = self.template.create()
        self.generation += 1
        self.episode, self.frame_id = episode, -1

    def render(self, state):
        self.frame_id += 1
        began = time.perf_counter()
        self.scene.apply(state)
        self.scene.take_picture()
        images = self.scene.read()
        return images, {"kind": "rendered", "worker_id": self.worker_id, "scene_id": self.worker_id,
                        "episode_id": self.episode, "generation": self.generation, "frame_id": self.frame_id,
                        "digest": state["digest"], "queue_s": 0, "service_s": time.perf_counter() - began,
                        "camera_config": self.scene.camera_config()}

    def close(self):
        self.scene.close()
        self.scene, self.template, self.anchor = None, None, None
        gc.collect()


def worker(index, options, report, descriptor):
    renderer = None
    try:
        initialization_started = time.perf_counter()
        if options["mode"] == "central":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            renderer = RenderClient(descriptor, index, index)
            if any(name == "sapien" or name.startswith("sapien.") for name in sys.modules):
                raise RuntimeError("CPU worker unexpectedly imported SAPIEN")
        else:
            os.chdir(options["robotwin_root"])
            renderer = LocalRenderer(options, index)
        generator = StateSequence(renderer.schema)
        renderer.reset(0)
        for frame in range(options["warmup"]):
            renderer.render(generator.frame(frame - options["warmup"], 0))
        report.send({"event": "warm_ready", "worker_id": index, "pid": os.getpid(),
                     "initialization_and_warmup_s": time.perf_counter() - initialization_started,
                     "sapien_imported": "sapien" in sys.modules,
                     "aimed_qpos": generator.aimed_qpos.tolist(), "aim_errors": generator.aim_errors,
                     "body_count": len(renderer.schema["body_names"]),
                     "render_shape_count": renderer.schema["render_shape_count"]})
        assert report.recv() == "start"
        samples, identities, durations, server_durations, queues = [], [], [], [], []
        started = time.perf_counter()
        for frame in range(options["frames"]):
            state = generator.frame(frame, 0)
            before = time.perf_counter()
            images, reply = renderer.render(state)
            durations.append(time.perf_counter() - before)
            server_durations.append(reply["service_s"])
            queues.append(reply["queue_s"])
            identities.append({k: reply[k] for k in ("worker_id", "scene_id", "episode_id", "generation", "frame_id", "digest")})
            if frame in (0, options["frames"] // 2, options["frames"] - 1):
                samples.append(("episode0-frame%03d" % frame, images, state, identities[-1]))
        finished = time.perf_counter()
        report.send({"event": "measured", "worker_id": index, "pid": os.getpid(),
                     "started_perf_s": started, "finished_perf_s": finished,
                     "observations": options["frames"], "images": options["frames"] * len(renderer.schema["camera_names"]),
                     "latency": summary(durations), "latencies_s": durations,
                     "service": summary(server_durations), "queue": summary(queues),
                     "identities": identities})
        assert report.recv() == "validate_reset"
        renderer.reset(1)
        state = generator.frame(0, 1)
        images, reply = renderer.render(state)
        identity = {k: reply[k] for k in ("worker_id", "scene_id", "episode_id", "generation", "frame_id", "digest")}
        if identity["generation"] != 2 or identity["frame_id"] != 0 or identity["episode_id"] != 1:
            raise RuntimeError("Reset generation/frame identity failed")
        samples.append(("episode1-frame000", images, state, identity))
        path = Path(options["artifact_dir"]) / ("worker-%02d" % index)
        path.mkdir(parents=True, exist_ok=True)
        records = []
        for name, images, state, identity in samples:
            np.savez_compressed(path / (name + ".npz"), **images, body_poses=state["body_poses"],
                                camera_poses=state["camera_poses"], robot_qpos=state["robot_qpos"])
            records.append({"sample": name, **identity,
                            "image_sha256": {name: hashlib.sha256(image.tobytes()).hexdigest() for name, image in images.items()}})
        renderer.close()
        renderer = None
        report.send({"event": "done", "worker_id": index, "samples": records})
    except BaseException:
        report.send({"event": "error", "worker_id": index, "error": traceback.format_exc()})
    finally:
        if renderer is not None:
            renderer.close()
        report.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", required=True)
    parser.add_argument("--mode", choices=("independent", "central"), required=True)
    parser.add_argument("--scenes", type=int, default=2)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = output.with_suffix("")
    artifacts.mkdir(exist_ok=True)
    options = vars(args).copy()
    options.pop("output")
    options.update(robotwin_root=str(Path(args.robotwin_root).resolve()), artifact_dir=str(artifacts),
                   suite="demo_clean", task="adjust_bottle", cameras="all", seed=1000000)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    ctx = mp.get_context("spawn")
    result = {"status": "incomplete", "options": options,
              "workload": "Synthetic CPU rigid-body/robot-FK/camera sequence using complete RoboTwin visual assets; no physical rollout",
              "renderer_settings": {"shader": "rt", "spp": 32, "path_depth": 8, "denoiser": "oidn"},
              "source_sha256": {name: sha256(Path(__file__).parent / name) for name in
                                ("benchmark_ipc.py", "central_render_service.py", "replay_state.py", "visual_scene.py", "robotwin_visual_replay.py")}}
    processes, reports = [], []
    service, control, memory, descriptor = None, None, None, None
    stop, gpu_samples = threading.Event(), []

    def monitor():
        while not stop.is_set():
            gpu_samples.append(gpu_sample())
            stop.wait(.25)

    monitor_thread = None
    try:
        result["gpu_before"] = gpu_sample()
        result["compute_apps_before"] = command_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
        if args.mode == "central":
            shape = [args.scenes, 4, 240, 320, 3]
            memory = SharedMemory(create=True, size=int(np.prod(shape)))
            descriptor = {"address": "/tmp/robotwin-render-%s.sock" % uuid.uuid4().hex,
                          "authkey_hex": os.urandom(32).hex(), "shared_memory": memory.name, "buffer_shape": shape}
            control, child = ctx.Pipe()
            service = ctx.Process(target=serve, args=(options, args.scenes, descriptor, child))
            service.start()
            child.close()
            ready = receive([control], "service_ready", 180)[0]
            result["service"] = ready
        for index in range(args.scenes):
            parent, child = ctx.Pipe()
            process = ctx.Process(target=worker, args=(index, options, child, descriptor))
            process.start()
            child.close()
            processes.append(process)
            reports.append(parent)
        result["warm_ready"] = receive(reports, "warm_ready", 180)
        result["gpu_after_warmup"] = gpu_sample()
        result["compute_apps_measured"] = command_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        for connection in reports:
            connection.send("start")
        result["measured"] = receive(reports, "measured", 180)
        stop.set()
        monitor_thread.join(timeout=15)
        result["gpu_after_measurement"] = gpu_sample()
        for connection in reports:
            connection.send("validate_reset")
        result["done"] = receive(reports, "done", 180)
        for process in processes:
            process.join(timeout=30)
            if process.exitcode != 0:
                raise RuntimeError("Worker exit failed")
        if service:
            control.send({"kind": "shutdown"})
            result["service_closed"] = receive([control], "service_closed", 30)[0]
            service.join(timeout=30)
            if service.exitcode != 0:
                raise RuntimeError("Service exit failed")
        elapsed = max(r["finished_perf_s"] for r in result["measured"]) - min(r["started_perf_s"] for r in result["measured"])
        images = sum(r["images"] for r in result["measured"])
        latencies = [value for r in result["measured"] for value in r["latencies_s"]]
        result["aggregate"] = {"elapsed_s": elapsed, "images": images, "observations": args.scenes * args.frames,
                               "images_per_s": images / elapsed, "latency": summary(latencies)}
        result["status"] = "complete"
    except BaseException:
        result["error"] = traceback.format_exc()
    finally:
        stop.set()
        for process in [*processes, *([service] if service else [])]:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
        for connection in reports:
            connection.close()
        if control:
            control.close()
        if memory:
            memory.close()
            memory.unlink()
        result["gpu_samples"] = gpu_samples
        result["gpu_after_cleanup"] = gpu_sample()
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "aggregate": result.get("aggregate"), "error": result.get("error")}))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
