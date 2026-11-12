#!/usr/bin/env python3
"""Run a named frozen RoboTwin research subset through the stock Kit CLI.

The complete committed seed generation is validated by the original factory first.
Only workload selection and episode-window scheduling are changed afterward.
This file is experiment tooling, not a change to the installed Kit or simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--subset-episodes", type=int, default=10)
    parser.add_argument("--subset-task-count", type=int, default=50)
    parser.add_argument("--episode-shard-size", type=int, default=2)
    parser.add_argument("--instrument-timing", action="store_true")
    parser.add_argument("--phase-log", type=Path)
    options, kit_args = parser.parse_known_args()
    if "--episodes" in kit_args:
        parser.error("use --subset-episodes; the original complete generation is validated first")
    if not 1 <= options.subset_episodes <= 100:
        parser.error("--subset-episodes must be between 1 and 100")
    if not 1 <= options.subset_task_count <= 50 or options.episode_shard_size < 1:
        parser.error("task count must be 1..50 and episode shard size must be positive")

    from roboscope_kit.runtime.benchmark import cli
    from roboscope_kit.runtime.benchmark.factories import representative_indices
    from roboscope_kit.runtime.benchmarks.robotwin.adapter import RoboTwinBenchmarkAdapter
    from roboscope_kit.runtime.benchmarks.robotwin.factory import RoboTwinBenchmarkFactory

    args = cli.parse_args(kit_args)
    if args.benchmark != "robotwin":
        parser.error("this runner supports only RoboTwin")
    output_dir = args.output_dir.resolve()
    phase_log = options.phase_log or output_dir.with_name(output_dir.name + ".phases.jsonl")
    phase_log.parent.mkdir(parents=True, exist_ok=True)
    original_handle_phase = cli.BenchmarkTimingRecorder.handle_phase

    def timestamped_phase(self, field, event):
        with phase_log.open("a") as stream:
            stream.write(json.dumps({
                "timestamp_unix_s": time.time(),
                "monotonic_s": time.monotonic(),
                "field": field,
                "event": event,
            }) + "\n")
        return original_handle_phase(self, field, event)

    cli.BenchmarkTimingRecorder.handle_phase = timestamped_phase
    original_load_manifest = cli.load_benchmark_manifest

    def scheduling_manifest(path):
        manifest = original_load_manifest(path)
        return replace(manifest, defaults={
            **manifest.defaults, "episode_shard_size": options.episode_shard_size,
        })

    cli.load_benchmark_manifest = scheduling_manifest
    original_create = RoboTwinBenchmarkFactory.create
    original_probe = RoboTwinBenchmarkFactory.create_probe
    subset_metadata: dict = {}
    instrumentation_dir = output_dir.with_name(output_dir.name + ".instrumentation")

    class ResearchSubsetAdapter(RoboTwinBenchmarkAdapter):
        def metadata(self):
            return {**super().metadata(), "research_subset": dict(subset_metadata),
                    "full_official_evaluation": False}

    if options.instrument_timing:
        instrumentation_dir.mkdir(parents=True, exist_ok=True)
        os.environ["ROBOSCOPE_TIMING_DIR"] = str(instrumentation_dir)
        original_policy = json.loads(args.policy_config.read_text())
        command = original_policy["command"]
        instrumented_policy = {
            **original_policy,
            "command": [command[0], str(Path(__file__).with_name("profile_policy.py")), *command[1:]],
        }
        policy_path = instrumentation_dir / "policy-config.json"
        policy_path.write_text(json.dumps(instrumented_policy, indent=2) + "\n")
        policy_index = kit_args.index("--policy-config") + 1
        kit_args[policy_index] = str(policy_path)

    def instrument(adapter):
        if not options.instrument_timing:
            return adapter
        os.environ["ROBOSCOPE_ORIGINAL_LAUNCHER"] = str(adapter.config.launcher_path)
        return type(adapter)(replace(
            adapter.config,
            launcher_path=Path(__file__).with_name("profile_launcher.py").resolve(),
        ))

    def subset_create(self, request, manifest):
        # The stock factory validates all tasks and all 100 frozen cache entries.
        full = original_create(self, replace(request, episodes=None), manifest)
        if request.quick_test:
            return instrument(full)
        indices = representative_indices(len(full.config.tasks), options.subset_task_count)
        tasks = tuple(full.config.tasks[index] for index in indices)
        adapter = ResearchSubsetAdapter(replace(
            full.config,
            tasks=tasks,
            episodes_per_task=options.subset_episodes,
            episode_shard_size=options.episode_shard_size,
        ))
        subset_metadata.update({
            "name": f"{request.suite}-frozen-prefix-{len(tasks)}x{options.subset_episodes}",
            "full_official_evaluation": False,
            "complete_generation_validated": True,
            "tasks": list(tasks),
            "task_indices": list(indices),
            "cache_episode_indices": list(range(options.subset_episodes)),
            "expected_episodes": len(tasks) * options.subset_episodes,
            "instruction_seed_limit": options.subset_episodes,
            "episode_shard_size": options.episode_shard_size,
            "seed_generation_digest": full.config.seed_generation_digest,
            "benchmark_protocol_id": full.config.protocol_id,
            "original_launcher": str(full.config.launcher_path),
            "instrumented_timing": options.instrument_timing,
        })
        return instrument(adapter)

    def instrumented_probe(self, request, manifest, concurrency):
        # Keep the Kit's own tuning workload and search semantics.
        adapter = original_probe(self, request, manifest, concurrency)
        return instrument(RoboTwinBenchmarkAdapter(replace(
            adapter.config, episode_shard_size=options.episode_shard_size,
        )))

    RoboTwinBenchmarkFactory.create = subset_create
    RoboTwinBenchmarkFactory.create_probe = instrumented_probe
    started = time.time()
    monotonic_started = time.monotonic()
    code = 1
    try:
        code = cli.main(kit_args)
        return code
    finally:
        ended = time.time()
        # The Kit reserves the run directory before starting; do not populate it early.
        if output_dir.exists():
            from importlib.metadata import version

            metadata = {
                **subset_metadata,
                "command": [sys.executable, *sys.argv],
                "effective_kit_arguments": kit_args,
                "kit_version": version("roboscope-kit"),
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "start_epoch_s": started,
                "end_epoch_s": ended,
                "start_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
                "end_utc": datetime.fromtimestamp(ended, timezone.utc).isoformat(),
                "end_to_end_s": time.monotonic() - monotonic_started,
                "exit_code": code,
                "instrumentation_dir": str(instrumentation_dir) if options.instrument_timing else None,
                "phase_log": str(phase_log),
            }
            (output_dir / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
