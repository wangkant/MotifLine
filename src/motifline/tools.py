"""Report where config differs from the defaults printed by the installed tools' --help."""
from __future__ import annotations

import ast
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

TOOLS = {
    "modisco-lite": {"argv": ["modisco", "motifs", "--help"]},
    "finemo": {"argv": ["finemo", "call-hits", "--help"]},
}


@dataclass(frozen=True)
class Parameter:
    key: str
    tool: str
    flag: str
    official_default_exists: bool = True
    note: str = ""


PARAMETERS = (
    Parameter("modisco.max_seqlets", "modisco-lite", "--max_seqlets",
              official_default_exists=False,
              note="-n is required by the CLI; there is no default to adopt"),
    Parameter("modisco.window", "modisco-lite", "--window"),
    Parameter("modisco.n_leiden", "modisco-lite", "--n_leiden",
              note="the CLI default; TFMoDISco() and the original TF-MoDISco use 50"),
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
    """{long-flag: default} for options whose help states one. Wrapped option
    blocks are rejoined first; options without a stated default are left out."""
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
    argv = TOOLS[tool]["argv"]
    exe = Path(env_prefix) / "bin" / argv[0]
    done = subprocess.run([str(exe), *argv[1:]], capture_output=True, text=True,
                          timeout=300)
    if done.returncode != 0:
        raise subprocess.CalledProcessError(done.returncode, done.args, done.stdout,
                                            done.stderr)
    found = parse_help_defaults(done.stdout + done.stderr)
    if not found:
        # an empty dict would read as "every parameter is on its default"
        raise subprocess.SubprocessError(f"{exe} {' '.join(argv[1:])} printed no defaults")
    return found


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
    if not rows:
        return ["every parameter this pipeline sets is on its tool's official default"]
    return ([f"{len(rows)} parameter(s) deviate from the tool's default:"]
            + [f"  - {r.line()}" for r in rows])
