"""Is this the official configuration of the official tools?

The point of this pipeline is not that it runs MoDISco and FiNeMo — anyone can run
those. It is that it runs them at parameters that are the tools' own defaults, with
the handful of deviations stated and argued. That claim used to live in prose, which
means it was true on the day it was written and unchecked ever after.

This module turns it into a measurement, made twice:

* **config against the installed tool.** Both tools print ``(default: X)`` in
  ``--help``, so the defaults can be read from the binary that is actually going to
  run. A deviation is reported, never blocked: deviating on purpose is ordinary
  science, deviating without knowing is the accident.
* **the installed tool against the pinned table.** ``tool_defaults.yaml`` records
  what each tool's defaults *were* at the pinned version. Without this second
  comparison a tool that moves a default is invisible whenever config moves with it —
  the config would still match the installed tool while no longer being the
  configuration the published runs used.

Nothing here is a gate. `motifline check` and `motifline run` print what they found.
"""
from __future__ import annotations

import ast
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

#: How each tool's help is reached, and which distribution carries its version.
TOOLS = {
    "modisco-lite": {"argv": ["modisco", "motifs", "--help"], "dist": "modisco-lite"},
    "finemo": {"argv": ["finemo", "call-hits", "--help"], "dist": "finemo"},
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


@lru_cache(maxsize=None)
def tool_version(tool: str, env_prefix: str) -> str | None:
    """The version the compute env reports, not the one the orchestrator has."""
    from motifline.provenance import tool_versions
    return tool_versions(env_prefix, (TOOLS[tool]["dist"],)).get(TOOLS[tool]["dist"])


@lru_cache(maxsize=1)
def _pinned() -> dict:
    return yaml.safe_load(Path(__file__).with_name("tool_defaults.yaml").read_text())


PINNED = "<pinned>"


def pinned_version(tool: str) -> str:
    return _pinned()[tool]["version"]


def pinned_defaults(tool: str) -> dict:
    return _pinned()[tool]["defaults"]


@dataclass(frozen=True)
class Deviation:
    kind: str           # "config" | "tool-default-moved"
    key: str
    tool: str
    flag: str
    value: object
    official: object
    note: str = ""

    def line(self) -> str:
        if self.kind == "config":
            s = (f"{self.key} = {self.value!r} but {self.tool} {self.flag} "
                 f"defaults to {self.official!r}")
        else:
            s = (f"{self.tool} {self.flag} now defaults to {self.value!r}; it was "
                 f"{self.official!r} at {self.tool} {pinned_version(self.tool)}, "
                 f"which is what this pipeline's parameters were chosen against")
        return s + (f"  ({self.note})" if self.note else "")


def deviations(cfg: dict, live: dict, pinned=None) -> list[Deviation]:
    """Where config differs from the tools, and where the tools differ from the pin.

    ``live`` is ``{tool: {flag-name: default}}``. Passing ``pinned`` turns on the
    second comparison; it is the module constant :data:`PINNED`, spelled out at the
    call site so the two questions stay visibly separate.
    """
    out: list[Deviation] = []
    for p in PARAMETERS:
        section, key = p.key.split(".")
        if key not in (cfg.get(section) or {}):
            continue
        if not p.official_default_exists:
            continue
        name = p.flag.lstrip("-")
        official = (live.get(p.tool) or {}).get(name)
        value = cfg[section][key]
        if official is not None and value != official:
            out.append(Deviation("config", p.key, p.tool, p.flag, value, official,
                                 p.note))

    if pinned is not None:
        for tool, defaults in sorted((live or {}).items()):
            if tool not in TOOLS:
                continue
            for name, was in sorted(pinned_defaults(tool).items()):
                now = defaults.get(name)
                if now is not None and now != was:
                    out.append(Deviation("tool-default-moved", "", tool, f"--{name}",
                                         now, was))
    return out


def render_deviations(rows) -> list[str]:
    """The deviation section, as lines. Never empty: silence would read as unchecked."""
    if not rows:
        return ["every parameter this pipeline sets is on its tool's official default"]
    out = []
    config_rows = [r for r in rows if r.kind == "config"]
    moved_rows = [r for r in rows if r.kind == "tool-default-moved"]
    if config_rows:
        out.append(f"{len(config_rows)} parameter(s) deviate from the tool's default "
                   f"(legitimate if deliberate; this is a report, not a refusal):")
        out += [f"  - {r.line()}" for r in config_rows]
    if moved_rows:
        out.append(f"{len(moved_rows)} tool default(s) have MOVED since this pipeline "
                   f"pinned them; re-pin src/motifline/tool_defaults.yaml once you have "
                   f"decided each change is one you meant to adopt:")
        out += [f"  - {r.line()}" for r in moved_rows]
    return out


_IMPORT_PROBE = """
import sys
sys.path.insert(0, sys.argv[1])
try:
    __import__(sys.argv[2])
    print("OK")
except Exception as e:
    print(f"{type(e).__name__}: {e}")
"""


def can_import(module: str, sys_path_entry: str, env_prefix: str) -> tuple[bool, str]:
    """Can the compute env import ``module`` with this checkout on sys.path?

    This is how the attribution steps reach model code -- ``sys.path.insert`` on the
    directory named by ``chrombpnet_src`` / a run's ``ag_src``, never an installed
    package -- so it is the only probe that answers the question a run will ask.
    """
    python = Path(env_prefix) / "bin" / "python"
    try:
        done = subprocess.run([str(python), "-c", _IMPORT_PROBE, sys_path_entry, module],
                              capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"{type(e).__name__}: {e}"
    detail = (done.stdout or done.stderr).strip().splitlines()
    detail = detail[-1] if detail else "no output"
    return detail == "OK", detail
