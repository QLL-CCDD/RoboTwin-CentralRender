"""Native RoboTwin get_obs adapter for an already physics-only worker.

This module has no SAPIEN/render imports. It does not bootstrap the full native
task or remove a GPU context created earlier. Attach it before observations to
a task constructed without a RenderSystem/renderer. Asset-changing reset sends
the actual adjust_bottle model to the service.
"""
from copy import deepcopy
import types

import numpy as np

from replay_state import digest_state


def pack_pose(pose):
    return np.r_[pose.p, pose.q].astype(np.float32)


def has_render_system(scene):
    try:
        return scene.render_system is not None
    except RuntimeError as error:
        if "no system with name [render]" not in str(error):
            raise
        return False


class RoboTwinCameraFacade:
    def __init__(self, task, client):
        self.task, self.client = task, client
        self.schema = client.schema
        self.poses = np.asarray(self.schema["camera_poses"], dtype=np.float32).copy()
        self.images, self.config = None, None

    def update_wrist_camera(self, left_pose, right_pose):
        for name, pose in (("left_camera", left_pose), ("right_camera", right_pose)):
            self.poses[self.schema["camera_names"].index(name)] = pack_pose(pose)

    def set_camera_pose(self, name, pose):
        self.poses[self.schema["camera_names"].index(name)] = pack_pose(pose)

    def capture_state(self):
        entities = {entity.name: entity for entity in self.task.scene.entities}
        body_poses = np.asarray([pack_pose(entities[name].pose) for name in self.schema["body_names"]], dtype=np.float32)
        robot = self.task.robot.left_entity
        if robot is not self.task.robot.right_entity:
            raise ValueError("The current scene template requires one dual-arm aloha articulation")
        qpos = dict(zip((joint.name for joint in robot.get_active_joints()), robot.get_qpos()))
        state = {"body_poses": body_poses, "camera_poses": self.poses.copy(),
                 "robot_qpos": np.asarray([qpos[name] for name in self.schema["robot_joint_names"]], dtype=np.float32)}
        state["digest"] = digest_state(state)
        return state

    def update_picture(self):
        self.images, reply = self.client.render(self.capture_state())
        self.config = reply["camera_config"]

    def get_rgb(self):
        if self.images is None:
            raise RuntimeError("update_picture must precede RGB retrieval")
        return {name: {"rgb": image} for name, image in self.images.items()}

    def get_config(self):
        if self.config is None:
            raise RuntimeError("No camera response for the current observation")
        # Native get_config returns fresh metadata; get_obs adds RGB in place.
        return deepcopy(self.config)

    def reset(self, episode_id):
        model = getattr(self.task, "model_id", self.schema["metadata"]["bottle_model"])
        self.client.reset(episode_id, {"task_name": "adjust_bottle", "suite": "demo_clean", "model_id": int(model)})
        self.schema = self.client.schema
        self.images, self.config = None, None


def attach_remote_observations(task, client):
    """Preserve Base_Task.get_obs packing while routing its RGB path remotely."""
    if has_render_system(task.scene) or getattr(task, "renderer", None) is not None:
        raise ValueError("Construct the worker as physics-only before attaching; existing GPU contexts are not removed")
    unsupported = ("depth", "pointcloud", "third_view", "mesh_segmentation", "actor_segmentation")
    if any(task.data_type.get(name, False) for name in unsupported):
        raise ValueError("This RGB backend currently supports the demo_clean RGB camera contract")
    facade = RoboTwinCameraFacade(task, client)
    task.cameras = facade

    def update_render(cpu_task):
        facade.update_wrist_camera(cpu_task.robot.left_camera.get_pose(), cpu_task.robot.right_camera.get_pose())

    task._update_render = types.MethodType(update_render, task)
    return facade
