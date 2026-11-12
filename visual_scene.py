"""GPU-only scene templates and rigid visual replicas for the render service."""
import gc
import os
from pathlib import Path

import numpy as np
import sapien
import yaml

import robotwin_visual_replay as replay


def packed_pose(pose):
    return np.r_[pose.p, pose.q].astype(np.float32).tolist()


def camera_map(cameras):
    result = dict(zip(cameras.static_camera_name, cameras.static_camera_list))
    result.update(left_camera=cameras.left_camera, right_camera=cameras.right_camera)
    return result


class SceneTemplate:
    def __init__(self, options, scene_id):
        os.chdir(options["robotwin_root"])
        source, metadata = replay.create_task(options, scene_id)
        self.renderer = source.renderer
        self.bodies = []
        for entity in source.scene.entities:
            bodies = [c for c in entity.components if isinstance(c, sapien.render.RenderBodyComponent)]
            if bodies:
                self.bodies.append((entity.name, packed_pose(entity.pose), [body.clone() for body in bodies]))
        if len({name for name, _, _ in self.bodies}) != len(self.bodies):
            raise RuntimeError("Visual entity names must be unique")
        cameras = camera_map(source.cameras)
        names = list(source.cameras.get_config())
        articulation = source.articulation
        self.schema = {"scene_id": scene_id, "body_names": [name for name, _, _ in self.bodies],
                       "body_poses": [pose for _, pose, _ in self.bodies], "camera_names": names,
                       "camera_poses": [packed_pose(cameras[name].entity.pose) for name in names],
                       "robot_joint_names": [joint.name for joint in articulation.get_active_joints()],
                       "robot_qpos": articulation.get_qpos().tolist(),
                       "robot_root_pose": packed_pose(articulation.get_root_pose()),
                       "urdf_path": str((Path(options["robotwin_root"]) / "assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf").resolve()),
                       "metadata": metadata, "shape": [len(names), 240, 320, 3],
                       "render_shape_count": sum(len(c.render_shapes) for _, _, cs in self.bodies for c in cs)}
        source.close_env()
        del source
        gc.collect()
        self.options = options

    def create(self):
        return VisualScene(self)


class VisualScene:
    def __init__(self, template):
        self.template = template
        self.renderer = template.renderer
        self.scene = sapien.Scene([sapien.render.RenderSystem()])
        self.scene.set_ambient_light([.5, .5, .5])
        self.scene.add_directional_light([0, .5, -1], [.5, .5, .5], shadow=True)
        self.scene.add_point_light([1, 0, 1.8], [1, 1, 1], shadow=True)
        self.scene.add_point_light([-1, 0, 1.8], [1, 1, 1], shadow=True)
        self.entities = []
        for name, pose, bodies in template.bodies:
            entity = sapien.Entity()
            entity.name = name
            entity.pose = sapien.Pose(pose[:3], pose[3:])
            for body in bodies:
                entity.add_component(body.clone())
            self.scene.add_entity(entity)
            self.entities.append(entity)
        root = Path(template.options["robotwin_root"])
        suite = yaml.safe_load((root / "task_config/demo_clean.yml").read_text())
        robot_config = yaml.safe_load((root / "assets/embodiments/aloha-agilex/config.yml").read_text())
        ns = {"sapien": sapien, "np": np, "yaml": yaml, "math": replay.math,
              "os": replay.os, "t3d": replay.t3d, "CONFIGS_PATH": str(root / "task_config")}
        replay.source_definitions(Path(replay.__file__).parent / "references/robotwin-c3ddfa8-camera.py", ["Camera"], ns)
        self.cameras = ns["Camera"](camera=suite["camera"], left_embodiment_config=robot_config)
        self.cameras.load_camera(self.scene)
        mapping = camera_map(self.cameras)
        self.camera_entities = [mapping[name].entity for name in template.schema["camera_names"]]
        self.apply({"body_poses": template.schema["body_poses"], "camera_poses": template.schema["camera_poses"]})

    def apply(self, state):
        for entity, pose in zip(self.entities, state["body_poses"]):
            entity.pose = sapien.Pose(pose[:3], pose[3:])
        for entity, pose in zip(self.camera_entities, state["camera_poses"]):
            entity.pose = sapien.Pose(pose[:3], pose[3:])
        self.scene.update_render()

    def take_picture(self):
        self.cameras.update_picture()

    def read(self):
        return {name: value["rgb"] for name, value in self.cameras.get_rgb().items()}

    def camera_config(self):
        return self.cameras.get_config()

    def close(self):
        self.camera_entities.clear()
        self.entities.clear()
        self.cameras = None
        self.scene = None
        gc.collect()
