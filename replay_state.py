"""CPU-only rigid state generation and protocol digests; no SAPIEN imports."""
import hashlib
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from transforms3d.axangles import axangle2mat
from transforms3d.euler import euler2mat
from transforms3d.quaternions import mat2quat, quat2mat


def pose_matrix(pose):
    matrix = np.eye(4)
    matrix[:3, :3] = quat2mat(np.asarray(pose[3:], dtype=np.float64))
    matrix[:3, 3] = pose[:3]
    return matrix


def matrix_pose(matrix):
    return np.r_[matrix[:3, 3], mat2quat(matrix[:3, :3])].astype(np.float32)


def digest_state(state):
    h = hashlib.sha256()
    for name in ("body_poses", "camera_poses", "robot_qpos"):
        h.update(np.asarray(state[name], dtype=np.float32).tobytes())
    return h.hexdigest()


class StateSequence:
    def __init__(self, schema):
        self.schema = schema
        tree = ET.parse(schema["urdf_path"])
        self.links = {node.get("name") for node in tree.findall("link")}
        children, self.joints = set(), []
        for node in tree.findall("joint"):
            parent, child = node.find("parent").get("link"), node.find("child").get("link")
            children.add(child)
            origin = node.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            transform = np.eye(4)
            transform[:3, :3] = euler2mat(*rpy)
            transform[:3, 3] = xyz
            axis = node.find("axis")
            axis = np.fromstring(axis.get("xyz", "1 0 0") if axis is not None else "1 0 0", sep=" ")
            self.joints.append((node.get("name"), node.get("type"), parent, child, transform, axis))
        self.root_name = next(iter(self.links - children))
        self.q0 = np.asarray(schema["robot_qpos"], dtype=np.float32)
        self.initial_bodies = np.asarray(schema["body_poses"], dtype=np.float32)
        self.initial_cameras = np.asarray(schema["camera_poses"], dtype=np.float32)
        fk = self.fk(self.q0)
        self.body_corrections = {name: np.linalg.inv(fk[name]) @ pose_matrix(pose)
                                 for name, pose in zip(schema["body_names"], self.initial_bodies) if name in fk}
        self.camera_corrections = {name: np.linalg.inv(fk[name]) @ pose_matrix(pose)
                                   for name, pose in zip(schema["camera_names"], self.initial_cameras) if name in fk}
        # This is synthetic geometric motion, not a recorded rollout. Aim the
        # wrist cameras into the workspace rather than at the home-state wall.
        self.aimed_qpos = self.q0.copy()
        limits = {j.get("name"): j.find("limit") for j in tree.findall("joint")}
        self.aim_errors = {}
        for prefix, camera, side in (("fl_", "left_camera", -1), ("fr_", "right_camera", 1)):
            indices = [i for i, name in enumerate(schema["robot_joint_names"])
                       if name in {prefix + "joint%d" % j for j in range(1, 7)}]
            lower, upper = [], []
            for i in indices:
                limit = limits[schema["robot_joint_names"][i]]
                lower.append(float(limit.get("lower", -3.14)))
                upper.append(float(limit.get("upper", 3.14)))
            desired_position = np.array([side * .20, -.22, 1.06])
            bottle = self.initial_bodies[schema["body_names"].index("001_bottle"), :3].astype(float)
            bottle[2] += .04

            def residual(values):
                qpos = self.aimed_qpos.copy()
                qpos[indices] = values
                pose = self.fk(qpos)[camera] @ self.camera_corrections[camera]
                forward = bottle - pose[:3, 3]
                forward /= np.linalg.norm(forward)
                return np.r_[3 * (pose[:3, 3] - desired_position),
                             1.5 * (pose[:3, 0] - forward), .02 * values]

            initial = np.clip(self.aimed_qpos[indices], np.array(lower) + 1e-5, np.array(upper) - 1e-5)
            solution = least_squares(residual, initial, bounds=(lower, upper), diff_step=.002, max_nfev=100)
            self.aimed_qpos[indices] = np.round(solution.x, 5)
            self.aim_errors[camera] = float(np.linalg.norm(solution.fun[:6]))

    def fk(self, qpos):
        values = dict(zip(self.schema["robot_joint_names"], qpos))
        result = {self.root_name: pose_matrix(self.schema["robot_root_pose"])}
        remaining = list(self.joints)
        while remaining:
            count = len(remaining)
            for joint in remaining[:]:
                name, kind, parent, child, origin, axis = joint
                if parent not in result:
                    continue
                motion = np.eye(4)
                q = float(values.get(name, 0))
                if kind in ("revolute", "continuous"):
                    motion[:3, :3] = axangle2mat(axis, q)
                elif kind == "prismatic":
                    motion[:3, 3] = axis * q
                result[child] = result[parent] @ origin @ motion
                remaining.remove(joint)
            if len(remaining) == count:
                raise RuntimeError("URDF joint tree is disconnected")
        return result

    def frame(self, index, episode):
        phase = index * .11 + episode * .83 + self.schema["scene_id"] * .37
        qpos = self.aimed_qpos.copy()
        for i, name in enumerate(self.schema["robot_joint_names"]):
            if name in ("fl_joint1", "fr_joint1"):
                qpos[i] += .08 * np.sin(phase + (0 if name.startswith("fl") else .8))
            if name in ("fl_joint2", "fr_joint2"):
                qpos[i] += .12 * np.sin(phase * .7 + .4)
        fk = self.fk(qpos)
        bodies = self.initial_bodies.copy()
        for i, name in enumerate(self.schema["body_names"]):
            if name in self.body_corrections:
                bodies[i] = matrix_pose(fk[name] @ self.body_corrections[name])
            elif name == "001_bottle":
                bodies[i, 0] += .025 * np.sin(phase)
                bodies[i, 1] += .02 * np.cos(phase)
        cameras = self.initial_cameras.copy()
        for i, name in enumerate(self.schema["camera_names"]):
            if name in self.camera_corrections:
                cameras[i] = matrix_pose(fk[name] @ self.camera_corrections[name])
            elif name == "head_camera":
                cameras[i, 0] += .01 * np.sin(phase * .8)
        state = {"body_poses": bodies, "camera_poses": cameras, "robot_qpos": qpos}
        state["digest"] = digest_state(state)
        return state
