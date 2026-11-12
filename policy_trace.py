"""Small real-rollout timing hooks; no Policy, actions or physics are replaced."""
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

_task = None
_installed = False
_first_observation = False


def instrument(launcher):
    destination = Path(os.environ["ROBOSCOPE_CLOSED_LOOP_DIR"])
    destination.mkdir(parents=True, exist_ok=True)
    stream = (destination / ("simulator-%d.jsonl" % os.getpid())).open("a", buffering=1)

    def emit(event, **fields):
        row = {"event": event, "pid": os.getpid(), "mode": os.environ["ROBOSCOPE_RENDER_MODE"],
               "epoch_s": time.time(), "perf_s": time.perf_counter(), **fields}
        stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        return row

    emit("simulator_start", cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    original_step = launcher.ModelClient.step
    original_end = launcher.ModelClient.end_episode
    passed_barrier = False

    @wraps(original_step)
    def step(self, example, step=0):
        nonlocal passed_barrier
        if not passed_barrier:
            from robotwin_worker_adapter import has_render_system

            has_renderer = has_render_system(_task.scene) or _task.renderer is not None
            assert has_renderer == (os.environ["ROBOSCOPE_RENDER_MODE"] == "native")
            row = emit("first_policy_ready", episode_index=int(os.environ["ROBOTWIN_EPISODE_START"]),
                       seed=int(_task._central_setup_kwargs["seed"]), render_system=has_renderer,
                       model_id=int(_task.model_id), observation_stride=self.observation_stride)
            (destination / ("ready-%d.json" % os.getpid())).write_text(json.dumps(row))
            release = destination / "release.json"
            deadline = time.monotonic() + 120
            while not release.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("Two real simulator workers did not reach the first Policy request")
                time.sleep(.01)
            passed_barrier = True
            emit("formal_started", release=json.loads(release.read_text()))
        return original_step(self, example, step=step)

    @wraps(original_end)
    def end(self):
        episode = self.episode_id
        result = original_end(self)
        if episode:
            row = emit("episode_finished", episode_id=episode,
                       episode_index=int(os.environ["ROBOTWIN_EPISODE_START"]),
                       seed=int(_task._central_setup_kwargs["seed"]), steps=int(_task.take_action_cnt),
                       step_lim=int(_task.step_lim), success=bool(_task.eval_success),
                       termination="success" if _task.eval_success else "horizon" if _task.take_action_cnt >= _task.step_lim else "other")
            (destination / ("finished-%d.json" % os.getpid())).write_text(json.dumps(row))
        return result

    launcher.ModelClient.step = step
    launcher.ModelClient.end_episode = end

    def install_task_hooks():
        global _installed
        if _installed:
            return
        _installed = True
        from envs._base_task import Base_Task

        original_init = Base_Task._init_task_env_
        original_obs = Base_Task.get_obs
        original_action = Base_Task.take_action

        @wraps(original_init)
        def initialize(self, *args, **kwargs):
            global _task
            started = time.perf_counter()
            emit("setup_start", seed=int(kwargs["seed"]))
            result = original_init(self, *args, **kwargs)
            _task = self
            emit("setup_complete", duration_s=time.perf_counter() - started,
                 seed=int(kwargs["seed"]), model_id=int(self.model_id))
            return result

        @wraps(original_obs)
        def observation(self, *args, **kwargs):
            global _first_observation
            started = time.perf_counter()
            result = original_obs(self, *args, **kwargs)
            fields = {}
            if not _first_observation:
                shapes = {name: list(value["rgb"].shape) for name, value in result["observation"].items()}
                assert set(shapes) == {"head_camera", "left_camera", "right_camera", "front_camera"}
                assert all(shape == [240, 320, 3] for shape in shapes.values())
                fields["camera_shapes"] = shapes
                _first_observation = True
            emit("observation", duration_s=time.perf_counter() - started,
                 action_step=int(self.take_action_cnt), **fields)
            return result

        @wraps(original_action)
        def action(self, value, *args, **kwargs):
            started = time.perf_counter()
            before = self.take_action_cnt
            result = original_action(self, value, *args, **kwargs)
            if self.take_action_cnt > before:
                array = np.asarray(value, dtype=np.float32)
                emit("action", duration_s=time.perf_counter() - started, step=int(self.take_action_cnt),
                     action=array.tolist(), action_sha256=hashlib.sha256(array.tobytes()).hexdigest(),
                     success=bool(self.eval_success))
            return result

        Base_Task._init_task_env_ = initialize
        Base_Task.get_obs = observation
        Base_Task.take_action = action

    def before_evaluation(original):
        @wraps(original)
        def run(*args, **kwargs):
            install_task_hooks()
            return original(*args, **kwargs)
        return run

    launcher.run_single_task = before_evaluation(launcher.run_single_task)
    launcher.run_eval_session = before_evaluation(launcher.run_eval_session)
