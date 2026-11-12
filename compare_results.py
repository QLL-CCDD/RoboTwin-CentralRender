#!/usr/bin/env python3
"""Check equal scene states/work and summarize a completed render pair."""
import argparse
import json
from pathlib import Path
import statistics

import numpy as np


def scenes(result):
    return {s["scene"]: s for event in result["saved"] for s in event["scenes"]}


def memory(result):
    records = [result["gpu_after_warmup"], *result["gpu_samples"], result["gpu_after_measurement"]]
    values = []
    for record in records:
        for line in record.get("stdout", "").splitlines():
            fields = [s.strip() for s in line.split(",")]
            if len(fields) == 6 and fields[0] == "0":
                values.append(float(fields[3]))
    return {"mean_mib": statistics.mean(values), "peak_mib": max(values), "samples": len(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("independent", type=Path)
    parser.add_argument("shared", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a, b = [json.loads(p.read_text(encoding="utf-8")) for p in (args.independent, args.shared)]
    assert a["status"] == b["status"] == "complete", "Both render arms must finish"
    assert a["options"]["mode"] == "independent" and b["options"]["mode"] == "shared"
    keys = ("robotwin_root", "launcher", "scene_source", "schedule", "scenes", "frames", "warmup", "seed", "task", "suite", "cameras")
    assert all(a["options"][k] == b["options"][k] for k in keys), "Workload options differ"
    assert a["renderer_settings"] == b["renderer_settings"]
    assert a["aggregate"]["observations"] == b["aggregate"]["observations"]
    assert a["aggregate"]["images"] == b["aggregate"]["images"]
    sa, sb = scenes(a), scenes(b)
    assert sa.keys() == sb.keys(), "Scene identities differ"
    image_differences = []
    for index in sorted(sa):
        assert sa[index]["state_sha256"] == sb[index]["state_sha256"], "Initial scene/camera states differ"
        assert sa[index]["source_sha256"] == sb[index]["source_sha256"], "Runtime sources differ"
        for sample in ("warm", "last"):
            pa = args.independent.with_suffix("") / ("scene-%02d-%s.npz" % (index, sample))
            pb = args.shared.with_suffix("") / ("scene-%02d-%s.npz" % (index, sample))
            with np.load(pa) as ia, np.load(pb) as ib:
                assert set(ia.files) == set(ib.files), "Camera sets differ"
                for name in ia.files:
                    x, y = ia[name], ib[name]
                    assert x.shape == y.shape and x.dtype == y.dtype
                    delta = np.abs(x.astype(np.float32) - y.astype(np.float32))
                    image_differences.append({"scene": index, "sample": sample, "camera": name,
                                              "shape": list(x.shape), "mae_0_255": float(delta.mean()),
                                              "p99_absolute_0_255": float(np.quantile(delta, .99)),
                                              "fraction_exact": float((x == y).mean()),
                                              "std_independent": float(x.std()), "std_shared": float(y.std())})
    output = {"equal_work_and_scene_states": True,
              "renderer_settings": a["renderer_settings"], "options": {k: a["options"][k] for k in keys},
              "independent": {**a["aggregate"], "device_memory": memory(a),
                              "latencies": [event["simultaneously_ready_request_completion"] for event in a["measured"]]},
              "shared": {**b["aggregate"], "device_memory": memory(b),
                         "latencies": [event["simultaneously_ready_request_completion"] for event in b["measured"]]},
              "throughput_ratio_shared_over_independent": b["aggregate"]["images_per_s"] / a["aggregate"]["images_per_s"],
              "image_differences": image_differences,
              "interpretation": "No IPC, no rollout. RT images may differ stochastically; inspect differences and previews before accepting image equivalence."}
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if k not in ("image_differences", "options")}, indent=2))


if __name__ == "__main__":
    main()
