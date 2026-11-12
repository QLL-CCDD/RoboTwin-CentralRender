"""Compare real equal-action physics replay, cameras and RGB artifacts."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    rows, contacts = [], []
    cameras = ("head_camera", "left_camera", "right_camera", "front_camera")
    for worker in range(2):
        relative = Path("worker-%02d" % worker)
        for path in sorted((args.root / "independent" / relative).glob("*.npz")):
            other = args.root / "central" / relative / path.name
            baseline, central = np.load(path), np.load(other)
            assert np.array_equal(baseline["body_names"], central["body_names"])
            states = {name: float(np.abs(baseline[name] - central[name]).max())
                      for name in ("body_poses", "qpos", "qvel", "wrist_poses", "joint_action")}
            images = {name: {"mae": float(np.abs(baseline[name].astype(float) - central[name]).mean()),
                             "exact_fraction": float((baseline[name] == central[name]).mean())} for name in cameras}
            rows.append({"worker": worker, "frame": path.name, "state_max_abs": states, "images": images})
            if path.stem in ("initial", "action-03", "reset"):
                contacts.append((worker, path.stem, baseline, central))
    result = {"samples": rows, "max_physics_abs": max(max(row["state_max_abs"].values()) for row in rows),
              "mean_image_mae": float(np.mean([image["mae"] for row in rows for image in row["images"].values()])),
              "max_image_mae": max(image["mae"] for row in rows for image in row["images"].values())}
    motion = []
    for worker in range(2):
        path = args.root / "central" / ("worker-%02d" % worker)
        initial, aimed = np.load(path / "initial.npz"), np.load(path / "action-03.npz")
        motion.append({"worker": worker, "wrist_pose_max_change": float(np.abs(aimed["wrist_poses"] - initial["wrist_poses"]).max()),
                       "image_mae": {name: float(np.abs(aimed[name].astype(float) - initial[name]).mean()) for name in cameras}})
    result["motion"] = motion
    reports = {mode: json.loads((args.root / mode / "result.json").read_text())
               for mode in ("independent", "central")}
    by_seed = {}
    for mode, report in reports.items():
        seeds = {row["pid"]: row["metadata"]["seed"] for row in report["ready"]}
        by_seed[mode] = {seeds[row["pid"]]: row for row in report["done"]}
    config_errors = []
    for seed, first in by_seed["independent"].items():
        second = by_seed["central"][seed]
        for frame, (a, b) in enumerate(zip(first["frames"], second["frames"])):
            assert (a["model_id"], a["action_count"], a["success"]) == (b["model_id"], b["action_count"], b["success"])
            for camera in cameras:
                for name, value in a["camera_config"][camera].items():
                    config_errors.append(float(np.abs(np.asarray(value) - b["camera_config"][camera][name]).max()))
    action_rows = {mode: {row["metadata"]["seed"]: row["actions"] for row in report["ready"]}
                   for mode, report in reports.items()}
    assert action_rows["independent"] == action_rows["central"]
    result["camera_config_max_abs"] = max(config_errors)
    result["actions_identical"] = True
    result["model_step_success_records_identical"] = True
    result["model_sequences"] = {str(seed): [row["frames"][0]["model_id"], row["frames"][-1]["model_id"]]
                                 for seed, row in by_seed["central"].items()}
    report = reports["central"]
    gpu_pids = {int(value) for value in re.findall(r"\|\s+0\s+N/A\s+N/A\s+(\d+)", report["gpu_owners"]["stdout"])}
    result["central_gpu_pids"] = sorted(gpu_pids)
    result["cpu_worker_pids"] = sorted(row["pid"] for row in report["ready"])
    result["only_service_owns_gpu"] = gpu_pids == {report["service"]["pid"]}
    assert result["only_service_owns_gpu"]
    (args.root / "comparison.json").write_text(json.dumps(result, indent=2))
    contact = Image.new("RGB", (4 * 320, len(contacts) * (2 * 240 + 25)), "white")
    draw = ImageDraw.Draw(contact)
    for row, (worker, frame, baseline, central) in enumerate(contacts):
        y = row * 505
        draw.text((5, y + 4), "worker %d / %s: baseline above, centralized below" % (worker, frame), fill="black")
        for i, name in enumerate(cameras):
            contact.paste(Image.fromarray(baseline[name]), (i * 320, y + 25))
            contact.paste(Image.fromarray(central[name]), (i * 320, y + 265))
    contact.save(args.root / "contact-native.png")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
