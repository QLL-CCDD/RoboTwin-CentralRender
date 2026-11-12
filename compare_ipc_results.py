"""Compare matched dynamic inputs, frame ownership, resets, pixels and speed."""
import argparse
import json
from pathlib import Path

import numpy as np

from compare_results import memory


def by_worker(events):
    return {event["worker_id"]: event for event in events}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("independent", type=Path)
    parser.add_argument("central", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a, b = [json.loads(path.read_text()) for path in (args.independent, args.central)]
    assert a["status"] == b["status"] == "complete"
    for key in ("scenes", "frames", "warmup", "robotwin_root", "cameras"):
        assert a["options"][key] == b["options"][key]
    assert a["source_sha256"] == b["source_sha256"]
    assert a["renderer_settings"] == b["renderer_settings"]
    assert a["aggregate"]["images"] == b["aggregate"]["images"]
    ma, mb = by_worker(a["measured"]), by_worker(b["measured"])
    da, db = by_worker(a["done"]), by_worker(b["done"])
    differences, motion = [], []
    cameras = ("head_camera", "front_camera", "left_camera", "right_camera")
    for worker_id in ma:
        assert ma[worker_id]["identities"] == mb[worker_id]["identities"], "Timed input/frame ownership differs"
        sa, sb = da[worker_id]["samples"], db[worker_id]["samples"]
        assert len(sa) == len(sb)
        for x, y in zip(sa, sb):
            assert all(x[k] == y[k] for k in ("sample", "worker_id", "scene_id", "episode_id", "generation", "frame_id", "digest"))
            pa = args.independent.with_suffix("") / ("worker-%02d" % worker_id) / (x["sample"] + ".npz")
            pb = args.central.with_suffix("") / ("worker-%02d" % worker_id) / (y["sample"] + ".npz")
            with np.load(pa) as ia, np.load(pb) as ib:
                for name in ("body_poses", "camera_poses", "robot_qpos"):
                    assert np.array_equal(ia[name], ib[name]), "State payload differs"
                for name in cameras:
                    difference = np.abs(ia[name].astype(np.float32) - ib[name].astype(np.float32))
                    differences.append({"worker_id": worker_id, "sample": x["sample"], "camera": name,
                                        "mae_0_255": float(difference.mean()), "max_0_255": float(difference.max()),
                                        "fraction_exact": float((difference == 0).mean())})
        folder = args.central.with_suffix("") / ("worker-%02d" % worker_id)
        with np.load(folder / "episode0-frame000.npz") as first, np.load(folder / ("episode0-frame%03d.npz" % (a["options"]["frames"] // 2))) as middle, np.load(folder / "episode1-frame000.npz") as reset:
            for name in cameras:
                motion.append({"worker_id": worker_id, "camera": name,
                               "motion_mae": float(np.abs(first[name].astype(float) - middle[name].astype(float)).mean()),
                               "reset_mae": float(np.abs(first[name].astype(float) - reset[name].astype(float)).mean())})
        assert all(item["motion_mae"] > 0 and item["reset_mae"] > 0 for item in motion if item["worker_id"] == worker_id)
    assert all(not event["sapien_imported"] for event in b["warm_ready"]), "Central CPU workers imported SAPIEN"
    # This driver reports duplicate rows for a process with graphics and CUDA.
    gpu_pids = sorted({int(line.split(",")[0]) for line in b["compute_apps_measured"]["stdout"].splitlines() if line.strip()})
    worker_pids = [event["pid"] for event in b["warm_ready"]]
    assert not set(gpu_pids).intersection(worker_pids)
    assert gpu_pids == [b["service"]["pid"]], "Unexpected GPU process during IPC measurement"
    result = {"same_dynamic_inputs_and_frame_ids": True, "reset_generation_verified": True,
              "cpu_workers_without_sapien_or_gpu_context": True, "gpu_process_pids": gpu_pids,
              "worker_pids": worker_pids, "renderer_settings": a["renderer_settings"],
              "independent": {**a["aggregate"], "device_memory": memory(a)},
              "central_ipc": {**b["aggregate"], "device_memory": memory(b)},
              "throughput_ratio": b["aggregate"]["images_per_s"] / a["aggregate"]["images_per_s"],
              "max_sample_mae": max(item["mae_0_255"] for item in differences),
              "minimum_exact_pixel_fraction": min(item["fraction_exact"] for item in differences),
              "image_differences": differences, "motion_and_reset_differences": motion,
              "service_inclusive_by_worker": [x["service"] for x in b["measured"]]}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("image_differences", "motion_and_reset_differences")}, indent=2))


if __name__ == "__main__":
    main()
