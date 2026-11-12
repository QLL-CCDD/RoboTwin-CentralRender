"""RoboTwin-specific CPU-physics bootstrap and remote RGB attachment.

Only adjust_bottle/demo_clean/aloha qpos evaluation is supported. Native physics,
collision meshes, masses, Mplib TOPP and gripper interpolation are preserved.
The optional cuRobo expert planner is not imported or built for qpos replay.
"""
import ast
import copy
import hashlib
import importlib
import importlib.util
import os
from pathlib import Path
import sys
import types
import xml.etree.ElementTree as ET

_client = None
_episode = 0
_configured = False


def _planner_without_expert(root):
    name = "envs.robot.planner"
    path = root / "envs/robot/planner.py"
    module = types.ModuleType(name)
    module.__package__ = "envs.robot"
    module.__file__ = str(path)

    class ExpertPlannerUnavailable:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("This qpos backend does not execute cuRobo expert planning")

    module.CuroboPlanner = ExpertPlannerUnavailable
    sys.modules[name] = module
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
                or isinstance(node, ast.ClassDef) and node.name == "MplibPlanner"]
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), module.__dict__)


def _physics_urdf(source, cache):
    from sapien.wrapper.urdf_loader import _try_very_hard_to_find_file

    source = Path(source).resolve()
    target = cache / (hashlib.sha256(source.read_bytes()).hexdigest()[:12] + "-physics.urdf")
    cache.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(source)
    root = tree.getroot()
    for material in root.findall("material"):
        root.remove(material)
    for link in root.findall("link"):
        for visual in link.findall("visual"):
            link.remove(visual)
        for mesh in link.findall("collision/geometry/mesh"):
            mesh.set("filename", str(_try_very_hard_to_find_file(mesh.get("filename"), str(source.parent))))
    temporary = target.with_name(target.name + ".%d.tmp" % os.getpid())
    tree.write(temporary, encoding="utf-8", xml_declaration=True)
    temporary.replace(target)
    return str(target)


class MetadataCamera:
    def __init__(self, name, width, height, **kwargs):
        import sapien

        self.name, self.width, self.height = name, width, height
        self.entity = self
        self.pose = sapien.Pose()

    def set_pose(self, pose):
        self.pose = pose

    def get_pose(self):
        return self.pose

    def take_picture(self):
        raise RuntimeError("A CPU metadata camera cannot render; attach the RGB service")


class MetadataCameraScene:
    def add_camera(self, **kwargs):
        return MetadataCamera(**kwargs)


def _install_cpu_scene(root, cache):
    import sapien
    from sapien.wrapper.actor_builder import ActorBuilder
    from envs._base_task import Base_Task, Camera

    actor_module = importlib.import_module("envs.utils.create_actor")

    def no_visual(self, *args, **kwargs):
        return self

    ActorBuilder.add_box_visual = no_visual
    ActorBuilder.add_visual_from_file = no_visual

    def physical_box(scene, pose, half_size, color=None, is_static=False, name="", texture_id=None):
        scene, pose = actor_module.preprocess(scene, pose)
        entity = sapien.Entity()
        entity.name, entity.pose = name, pose
        body = sapien.physx.PhysxRigidStaticComponent() if is_static else sapien.physx.PhysxRigidDynamicComponent()
        body.attach(sapien.physx.PhysxCollisionShapeBox(half_size=half_size, material=scene.default_physical_material))
        entity.add_component(body)
        scene.add_entity(entity)
        return entity

    actor_module.create_entity_box = physical_box

    def setup_scene(self, **kwargs):
        self.engine, self.renderer = sapien.Engine(), None
        sapien.physx.set_scene_config(sapien.SceneConfig())
        self.scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
        self.scene.set_timestep(kwargs.get("timestep", 1 / 250))
        self.scene.add_ground(kwargs.get("ground_height", 0), render=False)
        self.scene.default_physical_material = self.scene.create_physical_material(.5, .5, 0)
        self.direction_light_lst, self.point_light_lst = [], []

    original_robot = Base_Task.load_robot

    def load_robot(self, **kwargs):
        updated = dict(kwargs)
        for side in ("left", "right"):
            config = copy.deepcopy(kwargs[side + "_embodiment_config"])
            source = Path(kwargs[side + "_robot_file"]) / config["urdf_path"]
            config["urdf_path"] = _physics_urdf(source, cache)
            updated[side + "_embodiment_config"] = config
        return original_robot(self, **updated)

    def load_camera(self, **kwargs):
        self.cameras = Camera(bias=self.table_z_bias, random_head_camera_dis=self.random_head_camera_dis, **kwargs)
        # Run the original metadata/random-draw path, with no RenderCamera objects.
        self.cameras.load_camera(MetadataCameraScene())
        self.scene.step()  # Same initialization step as Base_Task.load_camera.

    def update_render(self):
        self.cameras.update_wrist_camera(self.robot.left_camera.get_pose(), self.robot.right_camera.get_pose())

    def close_env(self, clear_cache=False):
        # GPU caches belong to the service, never to this physics process.
        self.close()

    Base_Task.setup_scene = setup_scene
    Base_Task.load_robot = load_robot
    Base_Task.load_camera = load_camera
    Base_Task._update_render = update_render
    Base_Task.close_env = close_env


def configure(root, launcher_path, headless=False, descriptor=None, cache=None):
    global _configured
    root = Path(root).resolve()
    os.chdir(root)
    for path in (root, root / "script", root / "policy"):
        sys.path.insert(0, str(path))
    if headless:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OPEN3D_DISABLE_WEB_VISUALIZER"] = "true"
    _planner_without_expert(root)
    spec = importlib.util.spec_from_file_location("native_eval_launcher", launcher_path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    if headless:
        _install_cpu_scene(root, Path(cache or root.parent / "physics-urdf"))
        launcher.install_shared_engine = lambda: None
        # The official evaluator starts an unrelated Sapien_TEST render scene
        # before setup_demo. A physics worker must delegate that GPU preflight
        # to the already-ready service rather than constructing a local renderer.
        smoke = types.ModuleType("test_render")

        def remote_render_preflight():
            if not descriptor:
                raise RuntimeError("The CPU evaluation launcher requires a ready render service")
            print("ROBOSCOPE_RENDER_PREFLIGHT delegated_to_service; local_renderer_disabled", flush=True)

        smoke.Sapien_TEST = remote_render_preflight
        sys.modules["test_render"] = smoke
    else:
        launcher.install_shared_engine()
    launcher.install_gripper_only_planner()
    launcher.install_fast_control_loop()
    from envs._base_task import Base_Task

    original_init = Base_Task._init_task_env_

    def initialize(self, *args, **kwargs):
        global _client, _episode
        if headless:
            self.__dict__.pop("_update_render", None)
        self._central_setup_kwargs = copy.deepcopy(kwargs)
        result = original_init(self, *args, **kwargs)
        if headless and descriptor:
            from central_render_service import RenderClient
            from robotwin_worker_adapter import attach_remote_observations, pack_pose

            if _client is None:
                _client = RenderClient(descriptor, os.getpid(), -1)
            scene_spec = {"task_name": "adjust_bottle", "suite": "demo_clean", "model_id": int(self.model_id),
                          "seed": int(kwargs["seed"])}
            _client.reset(_episode, scene_spec=scene_spec)
            _episode += 1
            original = self.cameras
            cameras = dict(zip(original.static_camera_name, original.static_camera_list))
            cameras.update(left_camera=original.left_camera, right_camera=original.right_camera)
            facade = attach_remote_observations(self, _client)
            for i, name in enumerate(_client.schema["camera_names"]):
                facade.poses[i] = pack_pose(cameras[name].entity.pose)
            self._central_scene_spec = scene_spec
        return result

    Base_Task._init_task_env_ = initialize
    _configured = True
    return launcher


def close_client():
    global _client
    if _client is not None:
        _client.close()
        _client = None
