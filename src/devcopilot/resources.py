"""Resource metrics of the process and of the machine it runs on, without new dependencies.

Where the agents run, this is the sandbox: inside the microVM, /proc describes the VM,
so these gauges are the sandbox's own utilisation, exported over the OTLP port that the
sandbox policy already allows. GPU memory comes from Ollama (/api/ps: VRAM taken by the
loaded models) over the already allowed model port: there is no ready exporter for an
AMD GPU on Windows, and the models are what fills the GPU anyway.

Linux reads /proc; elsewhere (the Mac, Windows) the gauges that cannot be read are
simply absent: a missing series is honest, a zero would be a lie.
"""

import os
import shutil
import time
from pathlib import Path

import httpx
from opentelemetry.metrics import CallbackOptions, Observation

from devcopilot.config import get_settings

_PROC = Path("/proc")


def _meminfo() -> dict[str, int]:
    """/proc/meminfo in bytes: {"MemTotal": ..., "MemAvailable": ...}."""
    out = {}
    for line in (_PROC / "meminfo").read_text(encoding="utf-8").splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts:
            out[key] = int(parts[0]) * (1024 if parts[-1] == "kB" else 1)
    return out


def _process_rss() -> int | None:
    status = _PROC / "self" / "status"
    if status.exists():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return None


def process_cpu(options: CallbackOptions):
    # CPU seconds of this process; rate() over it is the utilisation in cores.
    t = os.times()
    yield Observation(t.user + t.system, {"cpu.mode": "user+system"})


def process_memory(options: CallbackOptions):
    if (rss := _process_rss()) is not None:
        yield Observation(rss)


def system_memory(options: CallbackOptions):
    if (_PROC / "meminfo").exists():
        m = _meminfo()
        yield Observation(m["MemTotal"] - m.get("MemAvailable", m.get("MemFree", 0)), {"state": "used"})
        yield Observation(m["MemTotal"], {"state": "total"})


def system_load(options: CallbackOptions):
    if hasattr(os, "getloadavg"):
        yield Observation(os.getloadavg()[0], {"window": "1m"})
    yield Observation(os.cpu_count() or 0, {"window": "cpus"})


def workspace_disk(options: CallbackOptions):
    s = get_settings()
    path = s.workspace_dir if s.workspace_dir.exists() else Path.cwd()
    usage = shutil.disk_usage(path)
    yield Observation(usage.used, {"state": "used"})
    yield Observation(usage.total, {"state": "total"})


class _OllamaVram:
    """VRAM of the models loaded in Ollama, cached: the reader asks every 10 s, and a
    request per export is enough; a down Ollama gives no observation, not an error."""

    def __init__(self) -> None:
        self._at, self._models = 0.0, []

    def __call__(self, options: CallbackOptions):
        if time.monotonic() - self._at > 5:
            self._at = time.monotonic()
            try:
                r = httpx.get(f"{get_settings().ollama_base_url}/api/ps", timeout=1)
                self._models = r.json().get("models", [])
            except (httpx.HTTPError, ValueError):
                self._models = []
        for m in self._models:
            yield Observation(m.get("size_vram", 0), {"model": m.get("name", "?"), "memory": "vram"})
            yield Observation(m.get("size", 0) - m.get("size_vram", 0), {"model": m.get("name", "?"), "memory": "ram"})


def register(meter, ollama: bool = True) -> None:
    """Observable instruments: read only when the metric reader exports. ollama: also the
    models' memory (one process is enough to ask: the CLI, not every MCP server)."""
    meter.create_observable_counter("process.cpu.time", [process_cpu], unit="s")
    meter.create_observable_gauge("process.memory.usage", [process_memory], unit="By")
    meter.create_observable_gauge("system.memory.usage", [system_memory], unit="By")
    meter.create_observable_gauge("system.cpu.load_average", [system_load], unit="{process}")
    meter.create_observable_gauge("devcopilot.workspace.disk", [workspace_disk], unit="By")
    if ollama:
        meter.create_observable_gauge("devcopilot.ollama.model_memory", [_OllamaVram()], unit="By")
