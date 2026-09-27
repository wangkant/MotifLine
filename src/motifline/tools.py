"""Where config differs from the defaults the installed tools print.

Both tools print ``(default: X)`` in ``--help``, so the defaults are read from the binary
that is going to run. A deviation is reported, never blocked.
"""
from __future__ import annotations

import ast
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

#: How each tool's help is reached.
TOOLS = {
    "modisco-lite": {"argv": ["modisco", "motifs", "--help"]},
    "finemo": {"argv": ["finemo", "call-hits", "--help"]},
}


@dataclass(frozen=True)
class Parameter:
    """One config key, and the tool flag its value actually reaches."""
    key: str
    tool: str
    flag: str
    #: False for a flag the CLI requires, which therefore has no default to deviate from.
    official_default_exists: bool = True
    note: str = ""


#: Every scientific parameter this pipeline sets on a tool. The mapping is not always
#: guessable from the config key, and two of them are actively misleading.
PARAMETERS = (
    Parameter("modisco.max_seqlets", "modisco-lite", "--max_seqlets",
              official_default_exists=False,
              note="-n is required by the CLI; there is no default to adopt"),
    Parameter("modisco.window", "modisco-lite", "--window"),
    Parameter("modisco.n_leiden", "modisco-lite", "--n_leiden"),
    Parameter("finemo.mode", "finemo", "--mode"),
    Parameter("finemo.threshold", "finemo", "--cwm-trim-threshold",
              note="-t trims the CWM; it is not a hit score threshold"),
    Parameter("finemo.alpha", "finemo", "--global-lambda",
              note="passed as -l. finemo also has a DEPRECATED --alpha, "
                   "which this is not"),
    Parameter("finemo.batch", "finemo", "--batch-size"),
)

_OPTION = re.compile(r"^  {0,2}(-\S)")
_LONG = re.compile(r"--([A-Za-z0-9][\w-]*)")
_DEFAULT = re.compile(r"\(default:\s*(.*?)\)\s*$", re.S)


def _value(text: str):
    text = " ".join(text.split())
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def parse_help_defaults(help_text: str) -> dict:
    """``{long-flag-name: default}`` for every option whose help states a default.

    argparse wraps, so ``(default:`` and its value routinely land on different lines;
    the blocks are rejoined before the default is read. Options with no stated default
    are absent rather than ``None`` — ``None`` is itself a default argparse prints.
    """
    lines = help_text.splitlines()
    blocks, current = [], None
    for line in lines:
        if _OPTION.match(line):
            if current is not None:
                blocks.append(current)
            current = [line]
        elif current is not None:
            if line.strip() == "" or not line.startswith(" "):
                blocks.append(current)
                current = None
            else:
                current.append(line)
    if current is not None:
        blocks.append(current)

    out = {}
    for block in blocks:
        joined = " ".join(l.strip() for l in block)
        names = _LONG.findall(joined)
        if not names or names[0] == "help":
            continue
        m = _DEFAULT.search(joined)
        if m:
            out[names[0]] = _value(m.group(1))
    return out


@lru_cache(maxsize=None)
def live_defaults(tool: str, env_prefix: str) -> dict:
    """The defaults the installed tool prints, read from the binary that will run."""
    argv = TOOLS[tool]["argv"]
    exe = Path(env_prefix) / "bin" / argv[0]
    done = subprocess.run([str(exe), *argv[1:]], capture_output=True, text=True,
                          timeout=300)
    return parse_help_defaults(done.stdout + done.stderr)


@dataclass(frozen=True)
class Deviation:
    key: str
    tool: str
    flag: str
    value: object
    official: object
    note: str = ""

    def line(self) -> str:
        s = (f"{self.key} = {self.value!r} but {self.tool} {self.flag} "
             f"defaults to {self.official!r}")
        return s + (f"  ({self.note})" if self.note else "")


def deviations(cfg: dict, live: dict) -> list[Deviation]:
    """Where config differs from ``live``, ``{tool: {flag-name: default}}``."""
    out: list[Deviation] = []
    for p in PARAMETERS:
        section, key = p.key.split(".")
        if key not in (cfg.get(section) or {}) or not p.official_default_exists:
            continue
        official = (live.get(p.tool) or {}).get(p.flag.lstrip("-"))
        value = cfg[section][key]
        if official is not None and value != official:
            out.append(Deviation(p.key, p.tool, p.flag, value, official, p.note))
    return out


def render_deviations(rows) -> list[str]:
    """The deviation report as lines; never empty, so silence cannot read as unchecked."""
    if not rows:
        return ["every parameter this pipeline sets is on its tool's official default"]
    return ([f"{len(rows)} parameter(s) deviate from the tool's default:"]
            + [f"  - {r.line()}" for r in rows])
