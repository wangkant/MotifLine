import pytest
from motifline.versions import dist_info_version

def _mk(tmp_path, *names):
    for n in names:
        (tmp_path / n).mkdir()
    return tmp_path

def test_reads_version_from_dist_info(tmp_path):
    sp = _mk(tmp_path, "finemo-0.30.dist-info", "numpy-2.0.1.dist-info")
    assert dist_info_version(sp, "finemo") == "0.30"

def test_raises_when_absent(tmp_path):
    sp = _mk(tmp_path, "numpy-2.0.1.dist-info")
    with pytest.raises(LookupError):
        dist_info_version(sp, "finemo")

def test_raises_on_ambiguous(tmp_path):
    sp = _mk(tmp_path, "finemo-0.30.dist-info", "finemo-0.41.dist-info")
    with pytest.raises(LookupError):
        dist_info_version(sp, "finemo")

def test_does_not_match_prefix_package(tmp_path):
    sp = _mk(tmp_path, "finemo_extras-1.0.dist-info", "finemo-0.30.dist-info")
    assert dist_info_version(sp, "finemo") == "0.30"
