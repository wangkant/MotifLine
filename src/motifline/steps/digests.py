"""File and checkout digests, shared by the steps and motifline.provenance.

Stdlib only: the steps import it as a sibling module in the compute env.
"""
import hashlib
import subprocess
from pathlib import Path


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def git_commit(path):
    """HEAD of the checkout holding `path`, plus a digest of uncommitted changes and of
    untracked .py files under `path`.

    None unless `path` itself is tracked: an installed package inside some other
    repository (a venv in a project) must not be stamped with that repository's HEAD.
    """
    def git(*args):
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True,
                              check=True).stdout
    try:
        git("ls-files", "--error-unmatch", "--", ".")
        head, diff = git("rev-parse", "HEAD").decode().strip(), git("diff", "HEAD")
        new = sorted(p for p in git("ls-files", "--others", "--exclude-standard", "-z",
                                    "--", "*.py").split(b"\0") if p)
    except (OSError, subprocess.CalledProcessError):
        return None
    if not diff and not new:
        return head
    h = hashlib.sha256(diff)
    for rel in new:
        h.update(rel + b"\0" + (Path(path) / rel.decode()).read_bytes())
    return f"{head}+{h.hexdigest()[:12]}"


def tree_sha256(path):
    """Digest of the .py files and Snakefile under `path`, for code outside git."""
    root = Path(path)
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if (p.is_file() and "__pycache__" not in rel.parts
                and (p.suffix == ".py" or p.name == "Snakefile")):
            h.update(f"{rel.as_posix()}\0{file_sha256(p)}\n".encode())
    return h.hexdigest()
