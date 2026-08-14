"""Pick fully-free GPUs by UUID, always holding one working card back."""
from __future__ import annotations
import contextlib
import os
import subprocess
import time
from pathlib import Path


def _nvidia_smi(args):
    return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True,
                          check=True).stdout


def free_gpu_uuids(leave_free: int = 1, _query=None) -> list[str]:
    query = _query or _nvidia_smi
    busy = set()
    for line in query(["--query-compute-apps=gpu_uuid,pid",
                       "--format=csv,noheader"]).strip().splitlines():
        if line.strip():
            busy.add(line.split(",")[0].strip())
    free = []
    for line in query(["--query-gpu=uuid,memory.used,utilization.gpu",
                       "--format=csv,noheader,nounits"]).strip().splitlines():
        uuid, mem, util = (x.strip() for x in line.split(","))
        if uuid not in busy and int(mem) < 100 and int(util) == 0:
            free.append(uuid)
    if len(free) <= leave_free:
        return []
    return free[: len(free) - leave_free]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _try_claim(lock_dir: Path, uuid: str) -> bool:
    """Create the lock for ``uuid`` atomically, reclaiming it if its holder is gone."""
    path = lock_dir / uuid
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            holder = int(path.read_text().split()[0])
        except (ValueError, IndexError, OSError):
            return False
        if _pid_alive(holder):
            return False
        # The holder died without releasing; its card is ours to take, but only if
        # the unlink wins the race against whoever else noticed the same thing.
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return _try_claim(lock_dir, uuid)
    os.write(fd, f"{os.getpid()} {time.time():.0f}\n".encode())
    os.close(fd)
    return True


@contextlib.contextmanager
def gpu_lease(lock_dir, leave_free: int = 1, wait_s: float = 0.0,
              poll_s: float = 30.0, _query=None):
    """Hold one fully-free GPU exclusively for the duration of the block.

    ``free_gpu_uuids`` alone is not enough once two jobs start at the same time:
    both read nvidia-smi before either has allocated anything and both pick the
    same card, which is exactly the sharing the constraints forbid. The lock
    directory makes the pick atomic between motifline processes, while nvidia-smi
    keeps other users' cards out of the pool. Cards freed a moment ago still
    report memory, so an empty pool is waited on rather than treated as fatal.
    """
    lock_dir = Path(lock_dir)
    lock_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + wait_s
    while True:
        for uuid in free_gpu_uuids(leave_free=leave_free, _query=_query):
            if _try_claim(lock_dir, uuid):
                try:
                    yield uuid
                finally:
                    (lock_dir / uuid).unlink(missing_ok=True)
                return
        if time.time() >= deadline:
            raise RuntimeError(
                f"no fully-free unclaimed GPU while leaving {leave_free} free "
                f"(waited {wait_s:.0f}s); not sharing a card with a running process")
        time.sleep(poll_s)
