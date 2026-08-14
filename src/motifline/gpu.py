"""Pick fully-free GPUs by UUID, always holding one working card back."""
from __future__ import annotations
import subprocess


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
