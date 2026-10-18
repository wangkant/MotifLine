"""Record tangermeme's DeepLIFT convergence check instead of switching it off.

tangermeme warns when |f(x) - f(ref) - sum(attr)| exceeds warning_threshold for a
batch of (sequence, reference) pairs and puts the deltas in the message. Catching
those warnings changes no score. Stdlib only; the steps import it as a sibling.
"""
import re
import warnings

THRESHOLD = 1e-3
_PREFIX = "Convergence deltas too high"
_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d*)?(?:[eE][-+]?\d+)?")


class Deltas:
    """Pairs over THRESHOLD and the largest delta inside the `with` block.

    tangermeme warns only for batches with some pair over the threshold, so `max` is
    the largest delta among those warned batches, 0.0 when no pair exceeded it.
    """

    def __init__(self):
        self.over = 0
        self.max = 0.0

    def __enter__(self):
        self._cm = warnings.catch_warnings(record=True)
        self._caught = self._cm.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, *exc):
        self._cm.__exit__(*exc)
        for w in self._caught:
            msg = str(w.message)
            if msg.startswith(_PREFIX):
                vals = [float(v) for v in _NUM.findall(msg[len(_PREFIX):])]
                self.over += sum(v > THRESHOLD for v in vals)
                self.max = max([self.max, *vals])
            else:
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        return False


def summary(pairs, over, max_delta):
    return {"threshold": THRESHOLD, "pairs": int(pairs), "pairs_over": int(over),
            "max_delta": float(max_delta)}
