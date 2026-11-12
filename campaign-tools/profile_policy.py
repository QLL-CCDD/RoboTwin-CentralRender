#!/usr/bin/env python3
"""Diagnostic-only server service and queue-lock timings for an unchanged Policy."""

from __future__ import annotations

import atexit
import json
import os
import runpy
import sys
import threading
import time
from pathlib import Path

from roboscope_kit.server import PolicyServer


destination = Path(os.environ["ROBOSCOPE_TIMING_DIR"])
destination.mkdir(parents=True, exist_ok=True)
stream = (destination / ("policy-%s.jsonl" % os.getpid())).open("a", buffering=1)
atexit.register(stream.close)
write_lock = threading.Lock()
nvtx = None
if os.environ.get("ROBOSCOPE_DIAGNOSTIC_NVTX") == "1":
    from torch.cuda import nvtx


def emit(name, epoch, elapsed, **extra):
    with write_lock:
        stream.write(json.dumps({
            "kind": name,
            "pid": os.getpid(),
            "thread_id": threading.get_ident(),
            "start_epoch_s": epoch,
            "duration_s": elapsed,
            "nvtx_enabled": nvtx is not None,
            **extra,
        }, separators=(",", ":")) + "\n")


class TimedRequestLock:
    def __init__(self, lock):
        self.lock = lock
        self.local = threading.local()

    def __enter__(self):
        self.acquire()
        return self

    def acquire(self, *args, **kwargs):
        epoch, started = time.time(), time.perf_counter()
        acquired = self.lock.acquire(*args, **kwargs)
        if acquired:
            self.local.wait = (epoch, time.perf_counter() - started)
        return acquired

    def release(self):
        self.lock.release()

    def __exit__(self, *args):
        self.release()


original_init = PolicyServer.__init__
original_handle = PolicyServer._handle


def timed_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self._request_lock = TimedRequestLock(self._request_lock)
    original_act = self.policy.act

    def timed_act(*act_args, **act_kwargs):
        epoch, started = time.time(), time.perf_counter()
        if nvtx is not None:
            nvtx.range_push("policy.act")
        try:
            return original_act(*act_args, **act_kwargs)
        finally:
            if nvtx is not None:
                nvtx.range_pop()
            elapsed = time.perf_counter() - started
            memory = {}
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_initialized():
                memory = {
                    "torch_max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "torch_max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
            emit("policy_act", epoch, elapsed, **memory)

    self.policy.act = timed_act


def timed_handle(self, request):
    queue_epoch, queue_elapsed = self._request_lock.local.wait
    emit("policy_queue_wait", queue_epoch, queue_elapsed,
         operation=request.get("operation"), request_id=request.get("request_id"))
    epoch, started = time.time(), time.perf_counter()
    try:
        return original_handle(self, request)
    finally:
        emit("policy_service", epoch, time.perf_counter() - started,
             operation=request.get("operation"), request_id=request.get("request_id"))


PolicyServer.__init__ = timed_init
PolicyServer._handle = timed_handle
target = Path(sys.argv[1]).resolve()
sys.argv = sys.argv[1:]
sys.path.insert(0, str(target.parent))
runpy.run_path(str(target), run_name="__main__")
