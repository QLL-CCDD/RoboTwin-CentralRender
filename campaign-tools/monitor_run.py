#!/usr/bin/env python3
"""Run a Linux evaluation with timestamped, one-second resource sampling.

Artifacts stay outside the Kit output directory. NVIDIA XML includes graphics
processes as well as compute processes; MPS can attribute client VRAM to its
server, so the raw process attribution is preserved rather than redistributed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET


def numeric(value):
    try:
        return float(value.split()[0])
    except (AttributeError, ValueError, IndexError):
        return None


def gpu_snapshot(device_map, selected):
    query = subprocess.run(
        ["nvidia-smi", "-q", "-x"], capture_output=True, text=True, timeout=10
    )
    if query.returncode:
        return {"error": query.stderr.strip(), "gpus": [], "gpu_processes": []}
    root = ET.fromstring(query.stdout)
    gpus, processes = [], []
    for ordinal, gpu in enumerate(root.findall("gpu")):
        uuid = gpu.findtext("uuid")
        index = device_map.get(uuid, str(ordinal))
        if selected and index not in selected and uuid not in selected:
            continue
        power = gpu.findtext("gpu_power_readings/power_draw")
        if power is None:
            power = gpu.findtext("gpu_power_readings/average_power_draw")
        if power is None:
            power = gpu.findtext("power_readings/power_draw")
        gpus.append({
            "index": index, "uuid": uuid, "name": gpu.findtext("product_name"),
            "utilization_percent": numeric(gpu.findtext("utilization/gpu_util")),
            "memory_utilization_percent": numeric(gpu.findtext("utilization/memory_util")),
            "memory_used_mib": numeric(gpu.findtext("fb_memory_usage/used")),
            "memory_total_mib": numeric(gpu.findtext("fb_memory_usage/total")),
            "power_w": numeric(power),
            "sm_clock_mhz": numeric(gpu.findtext("clocks/sm_clock")),
            "memory_clock_mhz": numeric(gpu.findtext("clocks/mem_clock")),
            "temperature_c": numeric(gpu.findtext("temperature/gpu_temp")),
            "pcie_rx_kb_s": numeric(gpu.findtext("pci/rx_util")),
            "pcie_tx_kb_s": numeric(gpu.findtext("pci/tx_util")),
            "pcie_link_gen": numeric(gpu.findtext("pci/pci_gpu_link_info/pcie_gen/current_link_gen")),
            "pcie_link_width": gpu.findtext("pci/pci_gpu_link_info/link_widths/current_link_width"),
            "power_limit_w": numeric(gpu.findtext("gpu_power_readings/current_power_limit")),
            "performance_state": gpu.findtext("performance_state"),
            "active_clock_event_reasons": [child.tag for child in
                gpu.findall("clocks_event_reasons/*") if child.text == "Active"],
        })
        for proc in gpu.findall("processes/process_info"):
            processes.append({
                "gpu_index": index, "gpu_uuid": uuid,
                "pid": int(proc.findtext("pid")),
                "type": proc.findtext("type"),
                "name": proc.findtext("process_name"),
                "memory_used_mib": numeric(proc.findtext("used_memory")),
            })
    return {
        "gpus": gpus, "gpu_processes": processes,
        "driver_version": root.findtext("driver_version"),
        "cuda_version": root.findtext("cuda_version"),
    }


def process_snapshot(root_pid, previous, interval_s):
    ticks_per_s = os.sysconf("SC_CLK_TCK")
    page_size = os.sysconf("SC_PAGE_SIZE")
    table = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            stat = (path / "stat").read_text()
            tail = stat.rsplit(")", 1)[1].split()
            table[int(path.name)] = {
                "pid": int(path.name), "ppid": int(tail[1]),
                "name": stat[stat.find("(") + 1:stat.rfind(")")],
                "cpu_ticks": int(tail[11]) + int(tail[12]),
                "start_ticks": int(tail[19]), "rss_bytes": int(tail[21]) * page_size,
            }
        except (OSError, ProcessLookupError):
            continue
    tree = {root_pid}
    while True:
        expanded = tree | {pid for pid, proc in table.items() if proc["ppid"] in tree}
        if expanded == tree:
            break
        tree = expanded
    rows, current = [], {}
    for pid in sorted(tree & table.keys()):
        row = table[pid]
        identity = (pid, row["start_ticks"])
        current[identity] = row["cpu_ticks"]
        before = previous.get(identity)
        row["cpu_percent_one_core"] = (
            100 * (row["cpu_ticks"] - before) / ticks_per_s / interval_s
            if before is not None and interval_s > 0 else None
        )
        rows.append(row)
    return rows, current


def cgroup_snapshot():
    base = Path("/sys/fs/cgroup")
    out = {}
    for name in ("cpu.stat", "cpu.max", "memory.current", "memory.peak", "memory.max"):
        path = base / name
        if path.is_file():
            out[name] = path.read_text().strip()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--gpu-devices", default="all", help="Physical indices or UUIDs, comma separated")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.interval <= 0:
        parser.error("Provide a command after -- and a positive sampling interval")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    samples_path = Path(str(args.output_prefix) + ".resources.jsonl")
    timing_path = Path(str(args.output_prefix) + ".timing.json")
    if samples_path.exists() or timing_path.exists():
        parser.error("The output prefix already has artifacts; choose a new prefix")
    mapping = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=True,
    )
    device_map = {uuid.strip(): index.strip() for index, uuid in
                  (line.split(",", 1) for line in mapping.stdout.splitlines())}
    selected = set(args.gpu_devices.split(",")) if args.gpu_devices != "all" else set()
    started_wall, started = time.time(), time.monotonic()
    proc = subprocess.Popen(command, start_new_session=True)
    timing = {
        "command": command, "root_pid": proc.pid, "started_unix_s": started_wall,
        "started_monotonic_s": started, "sample_interval_s": args.interval,
        "gpu_devices": args.gpu_devices,
    }
    timing_path.write_text(json.dumps(timing, indent=2) + "\n")
    completed = threading.Event()

    def wait_for_exit():
        rc = proc.wait()
        finished = time.monotonic()
        timing.update({
            "finished_unix_s": time.time(), "finished_monotonic_s": finished,
            "end_to_end_s": finished - started, "exit_code": rc,
        })
        completed.set()

    waiter = threading.Thread(target=wait_for_exit, daemon=True)
    waiter.start()

    def forward(signum, _frame):
        if proc.poll() is None:
            os.killpg(proc.pid, signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    previous, previous_t = {}, started
    with samples_path.open("w", buffering=1) as output:
        while True:
            now = time.monotonic()
            stamp = time.time()
            rows, previous = process_snapshot(proc.pid, previous, now - previous_t)
            previous_t = now
            try:
                gpu = gpu_snapshot(device_map, selected)
            except (OSError, subprocess.TimeoutExpired, ET.ParseError) as exc:
                gpu = {"gpus": [], "gpu_processes": [], "error": str(exc)}
            tree_pids = {row["pid"] for row in rows}
            for row in gpu["gpu_processes"]:
                row["in_evaluation_tree"] = row["pid"] in tree_pids
            known_cpu = [row["cpu_percent_one_core"] for row in rows
                         if row["cpu_percent_one_core"] is not None]
            sample = {
                "timestamp_unix_s": stamp, "monotonic_s": now,
                "elapsed_s": now - started, "sampling_elapsed_s": time.monotonic() - now,
                "processes": rows, "tree_rss_bytes": sum(row["rss_bytes"] for row in rows),
                "tree_cpu_cores": sum(known_cpu) / 100 if known_cpu else None,
                "cgroup": cgroup_snapshot(), **gpu,
            }
            output.write(json.dumps(sample, separators=(",", ":")) + "\n")
            if completed.is_set():
                break
            completed.wait(max(0, args.interval - (time.monotonic() - now)))
    waiter.join()
    timing_path.write_text(json.dumps(timing, indent=2) + "\n")
    rc = timing["exit_code"]
    return rc if rc >= 0 else 128 - rc


if __name__ == "__main__":
    sys.exit(main())
