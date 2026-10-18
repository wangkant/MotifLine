"""Which GPUs a run may use, and one card per GPU job.

`motifline run` fixes the device pool once and passes it to the workflow in
MOTIFLINE_GPU_DEVICES. Each GPU stage then claims a free device from the pool with a
file lock and runs with CUDA_VISIBLE_DEVICES set to that device alone, so concurrent GPU
jobs land on different cards. With an empty pool nothing is set and every GPU stage sees
whatever the caller's environment shows.
"""
from __future__ import annotations

import fcntl
import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

POOL_VAR = "MOTIFLINE_GPU_DEVICES"
LOCK_DIR = ".motifline/gpu_locks"


def visible_devices() -> list[str]:
    """CUDA_VISIBLE_DEVICES if the caller set it, else every card nvidia-smi lists."""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return parse(os.environ["CUDA_VISIBLE_DEVICES"])
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return []
    p = subprocess.run([smi, "--query-gpu=index", "--format=csv,noheader"],
                       capture_output=True, text=True)
    return parse(p.stdout.replace("\n", ",")) if p.returncode == 0 else []


def parse(spec: str) -> list[str]:
    """'0,2' or 'GPU-…,GPU-…' -> list of device ids; blanks dropped, order kept."""
    return [d.strip() for d in spec.split(",") if d.strip()]


def pool(devices: str | None) -> list[str]:
    """The run's devices: an explicit --devices list, else the visible ones."""
    return parse(devices) if devices is not None else visible_devices()


@contextmanager
def claim(devices: list[str], lock_dir=LOCK_DIR, wait: float = 5.0):
    """Hold one device of `devices` for the duration of the block; waits while all are taken."""
    lock_dir = Path(lock_dir)
    lock_dir.mkdir(parents=True, exist_ok=True)
    while True:
        for d in devices:
            f = open(lock_dir / f"{d.replace('/', '_')}.lock", "w")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                f.close()
                continue
            try:
                yield d
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
                f.close()
            return
        time.sleep(wait)
