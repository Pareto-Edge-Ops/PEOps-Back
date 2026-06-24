"""Host/system metrics for telemetry snapshots. psutil is optional — without
it, CPU% falls back to load-average and RSS to resource.getrusage. GPU metrics
are collected via nvidia-ml-py (pynvml) when present (the [gpu] extra); on a
machine without an NVIDIA GPU every GPU field is simply absent — never an error.

Two kinds of fields:
  • static identity   — collected once (runtime_fingerprint): cpu model, cores,
                        total RAM, available/active ORT providers, GPU name/VRAM.
  • dynamic sample    — collected every snapshot (system_sample): cpu%, rss,
                        and (if a GPU is present) GPU util%, GPU mem used, temp.
"""

from __future__ import annotations

import os
import platform
import socket
import sys

# nvidia-ml-py is imported lazily and at most once; _NVML is the cached handle
# (or False once we know it's unavailable) so we never pay the import twice.
_NVML: object | None = None


def _cpu_pct() -> float:
    try:
        import psutil

        return float(psutil.cpu_percent(interval=None))
    except ImportError:
        try:
            load1, _, _ = os.getloadavg()
            n = os.cpu_count() or 1
            return round(min(100.0, 100.0 * load1 / n), 1)
        except OSError:
            return 0.0


def _rss_mb() -> float:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / 1e6, 1)
    except ImportError:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is bytes on macOS, kilobytes on Linux.
        divisor = 1e6 if sys.platform == "darwin" else 1e3
        return round(rss / divisor, 1)


def _cpu_model() -> str:
    """Best-effort human CPU name. platform.processor() is empty on many Linux
    boxes and a bare arch ('arm', 'i386') on macOS, so fall back to
    /proc/cpuinfo (Linux) or sysctl's brand string (macOS)."""
    name = platform.processor() or ""
    arch = platform.machine() or ""
    # Treat a name that's just the architecture (or shorter) as "no real name".
    generic = (not name) or name.lower() in {arch.lower(), "arm", "i386", "x86_64"}
    if generic and sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        name = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
    if generic and sys.platform == "darwin":
        try:
            import subprocess

            brand = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=1.0,
            ).stdout.strip()
            if brand:
                name = brand
        except (OSError, ValueError):
            pass
    return (name or arch or "")[:96]


def _cpu_cores() -> int:
    try:
        import psutil

        return int(psutil.cpu_count(logical=True) or os.cpu_count() or 0)
    except ImportError:
        return int(os.cpu_count() or 0)


def _ram_total_mb() -> float:
    try:
        import psutil

        return round(psutil.virtual_memory().total / 1e6, 1)
    except ImportError:
        try:
            return round(
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e6, 1)
        except (ValueError, OSError, AttributeError):
            return 0.0


def _ort_version() -> str:
    try:
        import onnxruntime

        return onnxruntime.__version__
    except ImportError:
        return ""


def _available_providers() -> list[str]:
    try:
        import onnxruntime

        return list(onnxruntime.get_available_providers())
    except ImportError:
        return []


def _provider() -> str:
    avail = _available_providers()
    return avail[0] if avail else ""


def _nvml():
    """Return a live pynvml module (initialized) or None. Cached: the first
    failure disables every subsequent attempt so serving never pays for it."""
    global _NVML
    if _NVML is False:
        return None
    if _NVML is not None:
        return _NVML
    try:
        import pynvml  # nvidia-ml-py exposes the `pynvml` module

        pynvml.nvmlInit()
        _NVML = pynvml
        return pynvml
    except Exception:  # noqa: BLE001 — missing lib, no driver, no GPU, etc.
        _NVML = False
        return None


def _gpu_static() -> dict:
    """Static GPU identity (name, count, total VRAM, CUDA/driver). Empty dict
    when no NVIDIA GPU is visible — callers treat absence as 'no GPU'."""
    nvml = _nvml()
    if nvml is None:
        return {}
    try:
        count = int(nvml.nvmlDeviceGetCount())
        if count <= 0:
            return {}
        handle = nvml.nvmlDeviceGetHandleByIndex(0)
        name = nvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        mem = nvml.nvmlDeviceGetMemoryInfo(handle)
        try:
            driver = nvml.nvmlSystemGetDriverVersion()
            if isinstance(driver, bytes):
                driver = driver.decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            driver = ""
        try:
            raw = int(nvml.nvmlSystemGetCudaDriverVersion())
            cuda = f"{raw // 1000}.{(raw % 1000) // 10}"
        except Exception:  # noqa: BLE001
            cuda = ""
        return {
            "gpuName": str(name)[:96],
            "gpuCount": count,
            "gpuMemTotalMb": round(mem.total / 1e6, 1),
            "cudaVersion": cuda,
            "driverVersion": str(driver)[:32],
        }
    except Exception:  # noqa: BLE001
        return {}


def _gpu_sample() -> dict:
    """Dynamic GPU sample (util%, mem used, temp). Empty dict without a GPU."""
    nvml = _nvml()
    if nvml is None:
        return {}
    try:
        handle = nvml.nvmlDeviceGetHandleByIndex(0)
        util = nvml.nvmlDeviceGetUtilizationRates(handle)
        mem = nvml.nvmlDeviceGetMemoryInfo(handle)
        out = {
            "gpuUtilPct": float(util.gpu),
            "gpuMemUsedMb": round(mem.used / 1e6, 1),
        }
        try:
            out["gpuTempC"] = float(
                nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU))
        except Exception:  # noqa: BLE001 — temp is best-effort
            pass
        return out
    except Exception:  # noqa: BLE001
        return {}


def runtime_fingerprint(sdk_version: str, active_provider: str | None = None) -> dict:
    """Static runtime + hardware identity — sent with every snapshot.

    `active_provider` is the ORT execution provider actually selected by the
    serving session (vs the first *available* one); the runner passes it so the
    dashboard can attribute latency to the hardware that really served it."""
    avail = _available_providers()
    fp = {
        "sdkVersion": sdk_version,
        "pythonVersion": platform.python_version(),
        "ortVersion": _ort_version(),
        "os": platform.system(),
        "arch": platform.machine(),
        "provider": avail[0] if avail else "",
        "host": socket.gethostname(),
        # Hardware identity (new).
        "cpuModel": _cpu_model(),
        "cpuCores": _cpu_cores(),
        "ramTotalMb": _ram_total_mb(),
        "availableProviders": ",".join(avail),
        "activeProvider": active_provider or (avail[0] if avail else ""),
    }
    fp.update(_gpu_static())  # gpuName/gpuCount/gpuMemTotalMb/cudaVersion if present
    return fp


def system_sample() -> dict:
    """Dynamic part of a snapshot (cheap; called every ~30s)."""
    sample = {"cpuPct": _cpu_pct(), "rssMb": _rss_mb()}
    sample.update(_gpu_sample())  # gpuUtilPct/gpuMemUsedMb/gpuTempC if a GPU exists
    return sample
