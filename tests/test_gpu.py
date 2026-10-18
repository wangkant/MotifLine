"""Device pool and one-card-per-job claims."""
import threading

from motifline import gpu


def test_parse_keeps_order_and_drops_blanks():
    assert gpu.parse("2, 0,,GPU-ab") == ["2", "0", "GPU-ab"]
    assert gpu.parse("") == []


def test_cuda_visible_devices_wins_over_nvidia_smi(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    assert gpu.visible_devices() == ["3", "1"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert gpu.visible_devices() == []


def test_default_pool_is_nvidia_smi_uuids(monkeypatch, tmp_path):
    smi = tmp_path / "nvidia-smi"
    smi.write_text('#!/bin/sh\n[ "$1" = "--query-gpu=uuid" ] || exit 1\n'
                   'printf "GPU-aa\\nGPU-bb\\n"\n')
    smi.chmod(0o755)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert gpu.visible_devices() == ["GPU-aa", "GPU-bb"]


def test_explicit_devices_override_the_environment(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert gpu.pool("4,5") == ["4", "5"]


def test_concurrent_claims_get_different_cards(tmp_path):
    got, inside = [], threading.Barrier(2)

    def job():
        with gpu.claim(["0", "1"], lock_dir=tmp_path, wait=0.01) as d:
            got.append(d)
            inside.wait(timeout=5)

    ts = [threading.Thread(target=job) for _ in range(2)]
    [t.start() for t in ts]
    [t.join(timeout=10) for t in ts]
    assert sorted(got) == ["0", "1"]


def test_a_claim_waits_until_a_card_is_free(tmp_path):
    order = []

    def second():
        with gpu.claim(["0"], lock_dir=tmp_path, wait=0.01):
            order.append("second")

    with gpu.claim(["0"], lock_dir=tmp_path, wait=0.01):
        t = threading.Thread(target=second)
        t.start()
        t.join(timeout=0.2)
        order.append("first released")
    t.join(timeout=5)
    assert order == ["first released", "second"]
