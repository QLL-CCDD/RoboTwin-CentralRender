"""Full RoboTwin assets in a fixed nominal scene, without planning dependencies.

Camera and table/object constructors are compiled unchanged from pinned upstream
source. This is a visual workload, not a rollout or an expert-valid seed replay.
"""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from types import SimpleNamespace

import numpy as np
import sapien
import transforms3d as t3d
import yaml

_shared = None


def context():
    global _shared
    if _shared is None:
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(32)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser("oidn")
        renderer = sapien.SapienRenderer()
        engine = sapien.Engine()
        engine.set_renderer(renderer)
        _shared = SimpleNamespace(engine=engine, renderer=renderer)
    return SimpleNamespace(_SharedSapien=_shared)


def source_definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in body} != set(names):
        raise RuntimeError("Pinned source definitions missing")
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)


class ReplayTask:
    def _update_render(self):
        self.cameras.update_wrist_camera(self.left_camera.get_pose(), self.right_camera.get_pose())
        self.scene.update_render()

    def close_env(self):
        pass


def create_task(options, index):
    root = Path(options["robotwin_root"])
    reference = Path(__file__).resolve().parent / "references"
    suite_path = root / "task_config" / (options["suite"] + ".yml")
    config_path = root / "assets/embodiments/aloha-agilex/config.yml"
    camera_path = root / "task_config/_camera_config.yml"
    suite = yaml.safe_load(suite_path.read_text())
    robot_config = yaml.safe_load(config_path.read_text())
    robot_root = config_path.parent
    urdf_path = (robot_root / robot_config["urdf_path"]).resolve()
    started = time.perf_counter()
    shared = context()._SharedSapien
    task = ReplayTask()
    task.engine, task.renderer = shared.engine, shared.renderer
    scene = task.scene = shared.engine.create_scene()
    scene.set_timestep(1 / 250)
    scene.add_ground(0)
    scene.default_physical_material = scene.create_physical_material(.5, .5, 0)
    scene.set_ambient_light([.5, .5, .5])
    scene.add_directional_light([0, .5, -1], [.5, .5, .5], shadow=True)
    scene.add_point_light([1, 0, 1.8], [1, 1, 1], shadow=True)
    scene.add_point_light([-1, 0, 1.8], [1, 1, 1], shadow=True)
    namespace = {"sapien": sapien, "np": np, "Path": Path, "t3d": t3d, "json": json,
                 "os": os, "re": re, "Actor": lambda actor, data: actor}
    source_definitions(reference / "robotwin-c3ddfa8-create-actor.py",
                       ["preprocess", "create_entity_box", "create_table", "get_glb_or_obj_file", "create_actor"], namespace)
    namespace["create_entity_box"](scene, sapien.Pose([0, 1, 1.5]), [3, .6, 1.5],
                                    color=(1, .9, .9), name="wall", is_static=True)
    namespace["create_table"](scene, sapien.Pose([0, 0, .74]), length=1.2, width=.7,
                              height=.74, thickness=.05, is_static=True)
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load(str(urdf_path))
    if robot is None:
        raise RuntimeError("Full RoboTwin URDF did not load")
    pose = robot_config["robot_pose"][0]
    robot.set_root_pose(sapien.Pose(pose[:3], pose[3:]))
    qpos = np.zeros(robot.dof, dtype=np.float32)
    homes = dict(zip(sum(robot_config["arm_joints_name"], []), sum(robot_config["homestate"], [])))
    grippers = {}
    for entry in robot_config["gripper_name"]:
        grippers[entry["base"]] = robot_config["gripper_scale"][1]
        for name, multiplier, bias in entry["mimic"]:
            grippers[name] = robot_config["gripper_scale"][1] * multiplier + bias
    for i, joint in enumerate(robot.get_active_joints()):
        qpos[i] = homes.get(joint.name, grippers.get(joint.name, 0))
        joint.set_drive_properties(1000, 200)
        joint.set_drive_target(float(qpos[i]))
    robot.set_qpos(qpos)
    task.articulation = robot
    task.left_camera = robot.find_link_by_name("left_camera")
    task.right_camera = robot.find_link_by_name("right_camera")
    if task.left_camera is None or task.right_camera is None:
        raise RuntimeError("Native wrist camera links missing")
    model = int(options.get("bottle_model", 13 if index % 2 == 0 else 16))
    bottle_pose = sapien.Pose([-.1 if index % 2 == 0 else .1, -.1, .752],
                              [.707, 0, 0, -.707 if index % 2 == 0 else .707])
    task.bottle = namespace["create_actor"](scene, bottle_pose, "001_bottle", convex=True, model_id=model)
    camera_namespace = {"sapien": sapien, "np": np, "yaml": yaml, "math": math, "os": os,
                        "t3d": t3d, "CONFIGS_PATH": str(root / "task_config")}
    source_definitions(reference / "robotwin-c3ddfa8-camera.py", ["Camera"], camera_namespace)
    task.cameras = camera_namespace["Camera"](camera=suite["camera"], left_embodiment_config=robot_config)
    task.cameras.load_camera(scene)
    if options["cameras"] == "policy":
        kept = [(name, cam) for name, cam in zip(task.cameras.static_camera_name, task.cameras.static_camera_list)
                if name != "front_camera"]
        task.cameras.static_camera_name = [name for name, _ in kept]
        task.cameras.static_camera_list = [cam for _, cam in kept]
    # One initialization step updates articulation FK. No physics runs during
    # warmup or measurement; this is a nominal pose rather than setup_demo's
    # settled/expert-valid state, and both arms use exactly this same recipe.
    scene.step()
    task._update_render()
    paths = [suite_path, config_path, camera_path, urdf_path,
             reference / "robotwin-c3ddfa8-create-actor.py", reference / "robotwin-c3ddfa8-camera.py",
             Path(__file__).resolve().parent / "replay-assets-manifest.json", Path(__file__)]
    metadata = {"scene": index, "seed": options["seed"] + index, "setup_s": time.perf_counter() - started,
                "data_type": suite["data_type"], "camera_config": suite["camera"],
                "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                "bottle_model": model, "nominal_robot_qpos": qpos.tolist(),
                "scene_recipe": "Full aloha-agilex URDF + native table/wall/ground/lights + native bottle; fixed nominal state"}
    return task, metadata
