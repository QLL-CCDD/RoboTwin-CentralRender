"""Summarize completed real K2/G1 arms, their timing boundaries and GPU owners."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def local_time(epoch):
    return datetime.fromtimestamp(epoch, ZoneInfo("Asia/Shanghai")).isoformat(timespec="milliseconds")


def summarize(path):
    arm = json.loads((path / "arm-result.json").read_text())
    assert arm["returncode"] == 0
    assert len(arm["ready"]) == len(arm["episodes"]) == 2
    policies = [row for file in (path / "trace").glob("policy-*.jsonl") for row in lines(file)]
    inference = [row for row in policies if row["kind"] == "policy_act"]
    policy_pids = sorted({row["pid"] for row in inference})
    assert len(policy_pids) == 1 and inference
    sim_pids = sorted(row["pid"] for row in arm["ready"])
    actions, observations = {}, 0
    for ready in arm["ready"]:
        rows = lines(path / "trace" / ("simulator-%d.jsonl" % ready["pid"]))
        actions[ready["seed"]] = [row["action_sha256"] for row in rows if row["event"] == "action"]
        observations += sum(row["event"] == "observation" for row in rows)
    samples = lines(path / "monitor.resources.jsonl")
    measured = [row for row in samples if arm["formal_started_epoch_s"] <= row["timestamp_unix_s"] <= arm["formal_finished_epoch_s"]]
    gpu_pids = {row["pid"] for sample in samples for row in sample.get("gpu_processes", [])}
    graphics = {row["pid"] for sample in samples for row in sample.get("gpu_processes", []) if "G" in row.get("type", "")}
    memories = [gpu["memory_used_mib"] for sample in measured for gpu in sample["gpus"]]
    overall_memory = [gpu["memory_used_mib"] for sample in samples for gpu in sample["gpus"]]
    sdk = json.loads((path / "sdk/result.json").read_text())
    assert sdk["cuda_mps"]["enabled"] is False
    service_pid = None
    if arm["mode"] == "central":
        service_rows = json.loads((path / "render-service.json").read_text())
        ready = next(row for row in service_rows if row["event"] == "service_ready")
        assert ready["native_context_shared"] and ready["visual_scenes_without_physics"]
        assert len(ready["schemas"]) == 2
        service_pid = ready["pid"]
        assert not (set(sim_pids) & gpu_pids)
        assert graphics == {service_pid}
        assert gpu_pids == {service_pid, *policy_pids}
        assert all(not row["render_system"] for row in arm["ready"])
    else:
        assert graphics == set(sim_pids)
        assert gpu_pids == {*sim_pids, *policy_pids}
        assert all(row["render_system"] for row in arm["ready"])
    assert all(row["observation_stride"] == 16 for row in arm["ready"])
    phases = lines(path / "phases.jsonl")
    sdk_start = next(row["timestamp_unix_s"] for row in phases if row["field"] == "evaluation_s" and row["event"] == "start")
    sdk_end = next(row["timestamp_unix_s"] for row in phases if row["field"] == "evaluation_s" and row["event"] == "finish")
    result = {"mode": arm["mode"], "path": str(path), "completed_episodes": 2,
              "successes": sum(row["success"] for row in arm["episodes"]),
              "episodes": [{key: row[key] for key in ("episode_index", "seed", "steps", "step_lim", "success", "termination")}
                           for row in sorted(arm["episodes"], key=lambda row: row["episode_index"])],
              "total_steps": arm["total_steps"], "formal_wall_s": arm["formal_wall_s"], "steps_per_s": arm["steps_per_s"],
              "formal_start": local_time(arm["formal_started_epoch_s"]), "formal_end": local_time(arm["formal_finished_epoch_s"]),
              "total_wall_s": arm["total_wall_s"], "total_start": local_time(arm["started_epoch_s"]),
              "total_end": local_time(arm["finished_epoch_s"]), "sdk_evaluation_wall_s": sdk["phase_timings"]["evaluation_s"],
              "sdk_evaluation_start": local_time(sdk_start), "sdk_evaluation_end": local_time(sdk_end),
              "policy_load_s": sdk["phase_timings"]["policy_load_s"], "mps_enabled": False,
              "simulator_pids": sim_pids, "policy_pids": policy_pids, "actual_K": 2, "actual_G": 1,
              "render_service_pid": service_pid, "gpu_pids": sorted(gpu_pids), "graphics_pids": sorted(graphics),
              "gpu_ownership_verified": True, "actual_model_inference_calls": len(inference),
              "first_model_inference_s": inference[0]["duration_s"],
              "total_model_inference_s": sum(row["duration_s"] for row in inference),
              "subsequent_model_inference_mean_s": statistics.mean(row["duration_s"] for row in inference[1:]),
              "subsequent_model_inference_total_s": sum(row["duration_s"] for row in inference[1:]),
              "model_inference_durations_s": [row["duration_s"] for row in inference],
              "observations": observations, "rgb_images": observations * 4,
              "formal_vram_mean_mib": statistics.mean(memories), "formal_vram_peak_mib": max(memories),
              "overall_vram_peak_mib": max(overall_memory), "formal_resource_samples": len(measured)}
    return result, actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native", type=Path)
    parser.add_argument("central", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sensitivity-native", type=Path)
    args = parser.parse_args()
    native, native_actions = summarize(args.native)
    central, central_actions = summarize(args.central)
    assert [row["seed"] for row in native["episodes"]] == [row["seed"] for row in central["episodes"]]
    result = {"status": "complete", "configuration": {"task": "adjust_bottle", "suite": "demo_clean", "robot": "aloha-agilex",
              "action": "qpos", "K": 2, "G": 1, "episodes_per_arm": 2, "action_horizon": 16, "observation_stride": 16,
              "cameras": 4, "resolution": [320, 240], "shader": "rt", "spp": 32, "path_depth": 8, "denoiser": "oidn", "mps": False},
              "formal_boundary": "Both workers have their first real observations and wait before first real Policy step; release to last completed episode end.",
              "total_boundary": "Before CLI or render-service startup to command/service teardown, including fresh model loading and first-frame preparation.",
              "native": native, "central": central,
              "formal_throughput_gain": central["steps_per_s"] / native["steps_per_s"] - 1,
              "formal_wall_reduction": 1 - central["formal_wall_s"] / native["formal_wall_s"],
              "total_wall_reduction": 1 - central["total_wall_s"] / native["total_wall_s"],
              "same_step_count": native["total_steps"] == central["total_steps"],
              "action_sequences_identical": native_actions == central_actions,
              "first_inference_difference_s": native["first_model_inference_s"] - central["first_model_inference_s"],
              "formal_wall_difference_s": native["formal_wall_s"] - central["formal_wall_s"],
              "rendering_causal_attribution_established": False,
              "timing_caveat": "The native first inference took longer than central by more than the entire formal wall-time difference; the raw throughput difference cannot be attributed wholly to rendering.",
              "scope": "One small two-episode pair; no statistical or 50-task success-rate improvement claim."}
    result["completed_run_order"] = [str(args.native), str(args.central)]
    if args.sensitivity_native:
        sensitivity, sensitivity_actions = summarize(args.sensitivity_native)
        result["native_sensitivity"] = sensitivity
        result["sensitivity_action_sequences_identical"] = sensitivity_actions == native_actions
        result["completed_run_order"].append(str(args.sensitivity_native))
        result["sensitivity_note"] = "The already-launched native repeat completed before the stop instruction was processed. Retained as unpaired sensitivity evidence; no further centralized repeat was launched."
    result["conclusion"] = "The first pair has an observed +20.718% formal throughput difference, identical steps/actions and lower centralized VRAM, but first-inference/cache sensitivity and two episodes prevent a stable rendering-attributed or total-wall speedup claim."
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
