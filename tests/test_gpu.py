import contextlib
import pytest
from motifline.gpu import free_gpu_uuids, gpu_lease

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


def test_concurrent_leases_never_hand_out_the_same_card(tmp_path):
    # nvidia-smi still reports both cards free while the first lease's process is
    # starting up, so only the lock keeps the second lease off the same card.
    q = fake_query(GPUQ, APPS_BBB)
    with gpu_lease(tmp_path, leave_free=1, _query=q) as a, \
         gpu_lease(tmp_path, leave_free=1, _query=q) as b:
        assert a != b


def test_lease_releases_its_card_on_exit(tmp_path):
    q = fake_query(GPUQ, APPS_BBB)
    with gpu_lease(tmp_path, leave_free=1, _query=q) as a:
        pass
    with gpu_lease(tmp_path, leave_free=1, _query=q) as b:
        assert b == a
    assert not list(tmp_path.iterdir())


def test_lease_refuses_rather_than_shares_when_pool_is_exhausted(tmp_path):
    q = fake_query(GPUQ, APPS_BBB)
    with contextlib.ExitStack() as st:
        st.enter_context(gpu_lease(tmp_path, leave_free=1, _query=q))
        st.enter_context(gpu_lease(tmp_path, leave_free=1, _query=q))
        with pytest.raises(RuntimeError, match="not sharing a card"):
            st.enter_context(gpu_lease(tmp_path, leave_free=1, _query=q))


def test_lease_reclaims_a_lock_whose_holder_is_gone(tmp_path):
    (tmp_path / "GPU-aaa").write_text("2147483646 0\n")     # pid that cannot be running
    with gpu_lease(tmp_path, leave_free=1, _query=fake_query(GPUQ, APPS_BBB)) as a:
        assert a == "GPU-aaa"
