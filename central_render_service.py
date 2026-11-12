"""One-GPU local render service: small state RPCs and shared-memory RGB.

This module and RenderClient do not import SAPIEN. Only serve() imports the GPU
scene implementation. Scene/frame/episode/generation IDs bind each buffer reply.
"""
import gc
import json
from multiprocessing.connection import Client, Listener, wait
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
import time

import numpy as np

from replay_state import digest_state


class RenderClient:
    def __init__(self, descriptor, worker_id, scene_id):
        if isinstance(descriptor, (str, Path)):
            descriptor = json.loads(Path(descriptor).read_text())
        self.descriptor = descriptor
        self.worker_id, self.scene_id = worker_id, scene_id
        self.connection = Client(descriptor["address"], family="AF_UNIX", authkey=bytes.fromhex(descriptor["authkey_hex"]))
        self.connection.send({"kind": "hello", "worker_id": worker_id, "scene_id": scene_id})
        ready = self.connection.recv()
        if ready["kind"] != "hello":
            raise RuntimeError(str(ready))
        self.schema = ready["schema"]
        self.scene_id = self.schema["scene_id"]
        self.memory = SharedMemory(name=descriptor["shared_memory"])
        if descriptor.get("untrack_client_memory"):
            # Independent simulator processes must not unlink service-owned RGB
            # memory when their own Python resource tracker exits (Python 3.10).
            from multiprocessing import resource_tracker

            resource_tracker.unregister(self.memory._name, "shared_memory")
        self.buffer = np.ndarray(descriptor["buffer_shape"], dtype=np.uint8, buffer=self.memory.buf)
        self.generation, self.episode, self.frame_id = 0, -1, -1

    def reset(self, episode, scene_spec=None):
        self.connection.send({"kind": "reset", "worker_id": self.worker_id, "scene_id": self.scene_id,
                              "episode_id": episode, "scene_spec": scene_spec})
        reply = self.connection.recv()
        if reply.get("kind") != "reset" or reply["scene_id"] != self.scene_id or reply["episode_id"] != episode:
            raise RuntimeError(str(reply))
        self.generation, self.episode, self.frame_id = reply["generation"], episode, -1
        self.schema = reply.get("schema", self.schema)
        return reply

    def render(self, state):
        self.frame_id += 1
        identity = {"worker_id": self.worker_id, "scene_id": self.scene_id, "episode_id": self.episode,
                    "generation": self.generation, "frame_id": self.frame_id}
        self.connection.send({"kind": "render", **identity, "state": state})
        reply = self.connection.recv()
        if reply.get("kind") != "rendered" or any(reply.get(k) != v for k, v in identity.items()) or reply["digest"] != state["digest"]:
            raise RuntimeError("Frame identity/state mismatch: " + str(reply))
        images = {name: self.buffer[self.scene_id, i].copy() for i, name in enumerate(self.schema["camera_names"])}
        return images, reply

    def close(self):
        self.connection.send({"kind": "close", "worker_id": self.worker_id, "scene_id": self.scene_id})
        if self.connection.recv().get("kind") != "closed":
            raise RuntimeError("Missing close acknowledgement")
        self.connection.close()
        del self.buffer
        self.memory.close()


def serve(options, count, descriptor, control):
    import os
    import traceback

    import sapien
    from visual_scene import SceneTemplate

    memory = SharedMemory(name=descriptor["shared_memory"])
    buffer = np.ndarray(descriptor["buffer_shape"], dtype=np.uint8, buffer=memory.buf)
    listener, connections = None, {}
    scenes, templates = {}, {}
    reset_events = []
    try:
        # The anchor selects this process's GPU before the singleton is created.
        device_anchor = sapien.render.RenderSystem("cuda:%d" % options.get("gpu", 0))
        for i in range(count):
            templates[i] = SceneTemplate(options, i)
            scenes[i] = templates[i].create()
        contexts = [scene.renderer._internal_context for scene in scenes.values()]
        assert all(context is contexts[0] for context in contexts)
        listener = Listener(descriptor["address"], family="AF_UNIX", authkey=bytes.fromhex(descriptor["authkey_hex"]))
        control.send({"event": "service_ready", "pid": os.getpid(), "schemas": [templates[i].schema for i in range(count)],
                      "native_context_shared": True, "visual_scenes_without_physics": True})
        for _ in range(count):
            connection = listener.accept()
            hello = connection.recv()
            scene = hello["scene_id"]
            if scene == -1:
                occupied = {value["scene_id"] for value in connections.values()}
                scene = next(i for i in scenes if i not in occupied)
            if scene not in scenes or any(value["scene_id"] == scene for value in connections.values()):
                raise RuntimeError("Invalid/duplicate scene attachment")
            connections[connection] = {"worker_id": hello["worker_id"], "scene_id": scene,
                                       "generation": 0, "episode_id": -1, "frame_id": -1}
            connection.send({"kind": "hello", "schema": templates[scene].schema})
        running = True
        while running:
            ready = wait([control, *connections])
            pending = []

            def receive(connection):
                nonlocal running
                if connection is control:
                    if control.recv()["kind"] == "shutdown":
                        running = False
                    return
                try:
                    message = connection.recv()
                except EOFError:
                    connection.close()
                    del connections[connection]
                    return
                owner = connections[connection]
                if any(message.get(k) != owner[k] for k in ("worker_id", "scene_id")):
                    raise RuntimeError("Request owner mismatch")
                scene_id = owner["scene_id"]
                if message["kind"] == "reset":
                    previous_model = templates[scene_id].schema["metadata"]["bottle_model"]
                    spec = message.get("scene_spec")
                    scenes[scene_id].close()
                    if spec:
                        if spec["task_name"] != "adjust_bottle" or int(spec["model_id"]) not in (13, 16):
                            raise ValueError("This native backend supports adjust_bottle models 13/16")
                        if int(spec["model_id"]) != previous_model:
                            templates[scene_id] = SceneTemplate({**options, "bottle_model": int(spec["model_id"])}, scene_id)
                    scenes[scene_id] = templates[scene_id].create()
                    owner.update(generation=owner["generation"] + 1, episode_id=message["episode_id"], frame_id=-1)
                    reset_events.append({**owner, "scene_spec": spec, "model_before": previous_model,
                                         "model_after": templates[scene_id].schema["metadata"]["bottle_model"]})
                    connection.send({"kind": "reset", **owner, "schema": templates[scene_id].schema})
                elif message["kind"] == "close":
                    connection.send({"kind": "closed"})
                    connection.close()
                    del connections[connection]
                elif message["kind"] == "render":
                    if any(message[k] != owner[k] for k in ("episode_id", "generation")) or message["frame_id"] != owner["frame_id"] + 1:
                        raise RuntimeError("Stale/out-of-order render request")
                    if digest_state(message["state"]) != message["state"]["digest"]:
                        raise RuntimeError("State digest mismatch")
                    pending.append((connection, message, time.perf_counter()))
                else:
                    raise RuntimeError("Unknown request kind")

            for connection in ready:
                receive(connection)
            # At most 0.5 ms to collect other workers already becoming ready.
            if pending and running:
                excluded = {connection for connection, _, _ in pending}
                for connection in wait([control, *[c for c in connections if c not in excluded]], timeout=.0005):
                    receive(connection)
                began = time.perf_counter()
                for _, request, _ in pending:
                    scenes[request["scene_id"]].apply(request["state"])
                for _, request, _ in pending:
                    scenes[request["scene_id"]].take_picture()
                for connection, request, received in pending:
                    sid = request["scene_id"]
                    images = scenes[sid].read()
                    for i, name in enumerate(templates[sid].schema["camera_names"]):
                        buffer[sid, i] = images[name]
                    finished = time.perf_counter()
                    connections[connection]["frame_id"] = request["frame_id"]
                    connection.send({"kind": "rendered", **connections[connection], "digest": request["state"]["digest"],
                                     "queue_s": began - received, "service_s": finished - began,
                                     "camera_config": scenes[sid].camera_config()})
        control.send({"event": "service_closed", "reset_events": reset_events})
        del device_anchor
    except BaseException:
        error = traceback.format_exc()
        for connection in connections:
            try:
                connection.send({"kind": "error", "error": error})
            except (BrokenPipeError, EOFError):
                pass
        control.send({"event": "error", "error": error})
    finally:
        for scene in scenes.values():
            scene.close()
        scenes.clear()
        templates.clear()
        for connection in connections:
            connection.close()
        if listener:
            listener.close()
        del buffer
        memory.close()
        control.close()
        gc.collect()
