from motifline.gpu import free_gpu_uuids

GPUQ = ("GPU-aaa, 3, 0\nGPU-bbb, 40000, 97\nGPU-ccc, 5, 0\nGPU-ddd, 2, 0\n")
APPS_NONE = ""
APPS_BBB = "GPU-bbb, 12345\n"

def fake_query(gpu_out, apps_out):
    def q(args):
        return apps_out if "compute-apps" in " ".join(args) else gpu_out
    return q

def test_excludes_busy_and_leaves_one_free():
    got = free_gpu_uuids(leave_free=1, _query=fake_query(GPUQ, APPS_BBB))
    assert "GPU-bbb" not in got and len(got) == 2   # 3 free, keep 1 back

def test_returns_empty_when_only_one_free():
    one = "GPU-aaa, 3, 0\nGPU-bbb, 40000, 97\n"
    assert free_gpu_uuids(leave_free=1, _query=fake_query(one, APPS_BBB)) == []

def test_stale_context_counts_as_busy_via_mem_threshold():
    q = "GPU-aaa, 500, 0\nGPU-bbb, 3, 0\nGPU-ccc, 4, 0\n"
    got = free_gpu_uuids(leave_free=1, _query=fake_query(q, APPS_NONE))
    assert "GPU-aaa" not in got and len(got) == 1
