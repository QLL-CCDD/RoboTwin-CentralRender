#!/usr/bin/env python3
"""Scan RoboTwin seeds and cache the expert-valid ones for evaluation.

RoboTwin evaluation only accepts seeds whose scene the expert planner can solve
(``play_once`` succeeds and ``check_success`` holds). The pristine evaluator
re-runs that expert check before every episode, rebuilding the scene *and* the
cuRobo motion planner each time, which is what makes seed collection slow.

This scanner keeps the expensive runtime state warm per process and pays it
exactly once:

  * ONE SAPIEN ``Engine`` + ``SapienRenderer`` (the Vulkan/RT context) is built
    once and shared across every seed. Each seed gets a *fresh* ``Scene`` on
    that shared engine (``Base_Task.setup_scene`` is patched), so per-seed
    physics stays bit-exact while the renderer is never re-initialised.
  * ONE cuRobo ``CuroboPlanner`` per arm is built once (the ~20-36 s
    ``MotionGen.warmup()``) and reused for every seed. ``Robot.reset`` already
    skips rebuilding the planner when the existing one is a ``CuroboPlanner``;
    the cuRobo planner holds no scene reference (only the robot origin pose,
    joint names and yml), so it is safe to reuse across fresh scenes.

What is still reset per seed (to keep validity bit-exact): a fresh ``Scene``
with fresh robots, cameras and actors, and the official 2500-step
``check_stable``. ``episode_info`` is captured verbatim from ``play_once()``
and a seed is valid iff ``plan_success and check_success()`` -- unchanged.

Output layout matches what the evaluation launcher consumes:

    <out_dir>/<task_config>/<task>.pkl   ->  list[{"seed": int, "episode_info": dict}]
"""

from __future__ import annotations

import argparse
import importlib
import os
import pickle
import sys
import time
from pathlib import Path

import yaml
from numpy.linalg import LinAlgError

INVALID_SEED_ERRORS = {
    (AssertionError, "target_pose cannot be None for move action."),
    (TypeError, "'NoneType' object is not subscriptable"),
    (IndexError, "list index out of range"),
    (RuntimeError, "Fail to parameterize path"),
    (LinAlgError, "Eigenvalues did not converge"),
}
CPU_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--max-valid", type=int, default=0, help="stop after this many valid seeds (0=all)")
    parser.add_argument("--instruction-seed-limit", type=int, default=100)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=1,
        help="CPU threads available to Torch, OpenMP, MKL, OpenBLAS and NumExpr",
    )
    parser.add_argument(
        "--init-stagger-s",
        type=float,
        default=0.0,
        help="sleep this many seconds before the first setup_demo; use to stagger "
        "the one-time cuRobo warmup across concurrent processes and avoid piling "
        "onto the GPU at the same instant during init",
    )
    return parser.parse_args()


def configure_cpu_threads(cpu_threads: int) -> None:
    if cpu_threads < 1:
        raise ValueError("cpu-threads must be positive")
    value = str(cpu_threads)
    for name in CPU_THREAD_ENV_VARS:
        os.environ[name] = value

    import torch

    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(cpu_threads)


def is_invalid_seed_error(exc: Exception) -> bool:
    return (type(exc), str(exc)) in INVALID_SEED_ERRORS


def build_task_args(robotwin_dir: Path, task_name: str, task_config: str) -> dict:
    os.chdir(robotwin_dir)
    for extra in (str(robotwin_dir), str(robotwin_dir / "script"), str(robotwin_dir / "policy")):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    from envs import CONFIGS_PATH

    with open(robotwin_dir / "task_config" / f"{task_config}.yml", encoding="utf-8") as handle:
        args = yaml.load(handle.read(), Loader=yaml.FullLoader)
    args["task_name"] = task_name
    args["task_config"] = task_config
    embodiment_type = args.get("embodiment")
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), encoding="utf-8") as handle:
        embodiment_types = yaml.load(handle.read(), Loader=yaml.FullLoader)
    if len(embodiment_type) != 1:
        raise ValueError("seed scanner supports single-embodiment dual-arm tasks only")
    robot_file = embodiment_types[embodiment_type[0]]["file_path"]
    args["left_robot_file"] = robot_file
    args["right_robot_file"] = robot_file
    args["dual_arm_embodied"] = True
    with open(os.path.join(robot_file, "config.yml"), encoding="utf-8") as handle:
        embodiment_config = yaml.load(handle.read(), Loader=yaml.FullLoader)
    args["left_embodiment_config"] = embodiment_config
    args["right_embodiment_config"] = embodiment_config
    with open(CONFIGS_PATH + "_camera_config.yml", encoding="utf-8") as handle:
        camera_config = yaml.load(handle.read(), Loader=yaml.FullLoader)
    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_config[head_camera_type]["h"]
    args["head_camera_w"] = camera_config[head_camera_type]["w"]
    args["eval_mode"] = True
    args["eval_video_save_dir"] = None
    args["render_freq"] = 0
    args["eval_video_log"] = False
    return args


class SharedSapienContext:
    """One SAPIEN engine + renderer shared by every seed in the process.

    Building the ``SapienRenderer`` (a Vulkan ray-tracing context using the
    ``rt`` shader) is the single most expensive per-scene init step after the
    cuRobo warmup. Constructing it once and pointing every per-seed ``Scene``
    at it removes that cost from the steady-state per-seed time.
    """

    def __init__(self) -> None:
        import sapien.core as sapien
        from sapien.render import set_global_config

        self.engine = sapien.Engine()
        set_global_config(max_num_materials=50000, max_num_textures=50000)
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(32)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser("oidn")


def install_shared_engine_patch(context: SharedSapienContext) -> None:
    """Patch ``Base_Task.setup_scene`` to build a fresh Scene on the shared
    engine/renderer instead of a brand-new Engine + Renderer per seed.

    Semantics are verbatim from the official ``setup_scene``: identical
    timestep, ground, material, ambient/direction/point lights. The only
    difference is *which* engine/renderer own the scene. ``render_freq == 0``
    in the scanner task args, so no ``Viewer`` is created.
    """
    import numpy as np
    import sapien.core as sapien
    from envs._base_task import Base_Task

    def setup_scene_on_shared_engine(self, **kwargs):
        self.engine = context.engine
        self.renderer = context.renderer

        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(scene_config)
        self.scene.set_timestep(kwargs.get("timestep", 1 / 250))
        self.scene.add_ground(kwargs.get("ground_height", 0))
        self.scene.default_physical_material = self.scene.create_physical_material(
            kwargs.get("static_friction", 0.5),
            kwargs.get("dynamic_friction", 0.5),
            kwargs.get("restitution", 0),
        )
        self.scene.set_ambient_light(kwargs.get("ambient_light", [0.5, 0.5, 0.5]))
        shadow = kwargs.get("shadow", True)
        direction_lights = kwargs.get("direction_lights", [[[0, 0.5, -1], [0.5, 0.5, 0.5]]])
        self.direction_light_lst = []
        for direction_light in direction_lights:
            if self.random_light:
                direction_light[1] = [np.random.rand(), np.random.rand(), np.random.rand()]
            self.direction_light_lst.append(
                self.scene.add_directional_light(direction_light[0], direction_light[1], shadow=shadow)
            )
        point_lights = kwargs.get("point_lights", [[[1, 0, 1.8], [1, 1, 1]], [[-1, 0, 1.8], [1, 1, 1]]])
        self.point_light_lst = []
        for point_light in point_lights:
            if self.random_light:
                point_light[1] = [np.random.rand(), np.random.rand(), np.random.rand()]
            self.point_light_lst.append(
                self.scene.add_point_light(point_light[0], point_light[1], shadow=shadow)
            )

        if self.render_freq:
            raise RuntimeError("seed scanner requires render_freq == 0")

    Base_Task.setup_scene = setup_scene_on_shared_engine


def main() -> int:
    args = parse_args()
    if args.seed_count < 1:
        raise ValueError("seed-count must be positive")
    configure_cpu_threads(args.cpu_threads)
    robotwin_dir = args.robotwin_dir.resolve()
    task_args = build_task_args(robotwin_dir, args.task, args.task_config)

    module = importlib.import_module(f"envs.{args.task}")
    env_class = getattr(module, args.task)

    out_dir = args.out_dir / args.task_config
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.task}.pkl"

    # Load any pre-existing entries for idempotent re-runs (keyed by seed).
    existing = []
    if out_path.is_file():
        with out_path.open("rb") as handle:
            existing = pickle.load(handle)
    done_seeds = {int(entry["seed"]) for entry in existing}

    # Build the shared engine/renderer ONCE, then patch setup_scene so every
    # seed's setup_demo reuses it. The cuRobo planner is built on the first
    # seed's setup_demo (Robot.set_planner) and preserved across seeds by
    # Robot.reset's isinstance(existing, CuroboPlanner) guard.
    context = SharedSapienContext()
    install_shared_engine_patch(context)

    if args.init_stagger_s > 0:
        # Stagger the one-time cuRobo warmup / first scene build so concurrent
        # processes do not pile onto the GPU at the same instant (which under
        # high concurrency makes the first setup_demo fail and can poison the
        # whole run if the failure is mis-read as "invalid seeds").
        time.sleep(args.init_stagger_s)

    # UnStableError and RoboTwin's exact "no grasp pose" assertion both mean
    # this seed is invalid. Other exceptions indicate a broken environment,
    # renderer, or cuRobo planner and deliberately propagate instead of being
    # silently mis-recorded as invalid seeds.
    from envs.utils.create_actor import UnStableError

    env = env_class()
    valid = list(existing)
    scanned = 0
    started = time.perf_counter()
    first_setup_s: float | None = None
    steady_setup_s = 0.0
    steady_play_s = 0.0
    steady_seeds = 0

    for offset in range(args.seed_count):
        seed = args.seed_start + offset
        if seed in done_seeds:
            continue
        setup_start = time.perf_counter()
        try:
            env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **task_args)
            setup_s = time.perf_counter() - setup_start
            play_start = time.perf_counter()
            episode_info = env.play_once()
            play_s = time.perf_counter() - play_start
            ok = bool(env.plan_success and env.check_success())
        except UnStableError:
            # Genuinely unstable scene -> this seed is simply invalid.
            setup_s = time.perf_counter() - setup_start
            play_s = 0.0
            ok = False
        except (AssertionError, IndexError, LinAlgError, RuntimeError, TypeError) as exc:
            if not is_invalid_seed_error(exc):
                raise
            setup_s = time.perf_counter() - setup_start
            play_s = 0.0
            ok = False
        scanned += 1
        if first_setup_s is None:
            first_setup_s = setup_s  # includes the one-time cuRobo warmup
        else:
            steady_setup_s += setup_s
            steady_play_s += play_s
            steady_seeds += 1
        env.close_env()
        if ok:
            valid.append({"seed": int(seed), "episode_info": episode_info})
            if args.max_valid and len(valid) >= args.max_valid:
                break

    valid.sort(key=lambda entry: int(entry["seed"]))
    temporary = out_path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(valid, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(out_path)

    elapsed = time.perf_counter() - started
    rate = scanned / max(elapsed, 1e-9) * 60
    # Steady-state excludes the first seed (which carries the one-time cuRobo
    # warmup and renderer init already paid before the loop).
    steady_per_seed = (steady_setup_s + steady_play_s) / steady_seeds if steady_seeds else 0.0
    steady_elapsed = steady_setup_s + steady_play_s
    steady_rate = steady_seeds / steady_elapsed * 60 if steady_elapsed > 0 else 0.0
    print(
        "SEED_SCAN_RESULT "
        + str(
            {
                "task": args.task,
                "task_config": args.task_config,
                "cpu_threads": args.cpu_threads,
                "scanned": scanned,
                "valid": len(valid),
                "new_valid": len(valid) - len(existing),
                "elapsed_s": round(elapsed, 2),
                "seeds_per_min": round(rate, 1),
                "first_setup_s": round(first_setup_s or 0.0, 2),
                "steady_setup_s": round(steady_setup_s / steady_seeds, 3) if steady_seeds else 0.0,
                "steady_play_s": round(steady_play_s / steady_seeds, 3) if steady_seeds else 0.0,
                "steady_per_seed_s": round(steady_per_seed, 3),
                "steady_seeds_per_min": round(steady_rate, 1),
                "out_path": str(out_path),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
