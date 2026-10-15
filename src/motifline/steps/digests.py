"""File and checkout digests, shared by the steps and motifline.provenance.

Stdlib only: the steps import it as a sibling module in the compute env.
"""
import hashlib
import subprocess


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def git_commit(path):
    """HEAD of the checkout holding `path`, plus a digest of uncommitted changes."""
    def git(*args):
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True,
                              check=True).stdout
    try:
        head, diff = git("rev-parse", "HEAD").decode().strip(), git("diff", "HEAD")
    except (OSError, subprocess.CalledProcessError):
        return None
    return head if not diff else f"{head}+{hashlib.sha256(diff).hexdigest()[:12]}"
