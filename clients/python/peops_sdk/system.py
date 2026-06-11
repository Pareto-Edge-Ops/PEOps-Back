"""Host/system metrics for telemetry snapshots. psutil is optional — without
it, CPU% falls back to load-average and RSS to resource.getrusage."""

from __future__ import annotations

import os
import platform
import socket
import sys


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


def _ort_version() -> str:
    try:
        import onnxruntime

        return onnxruntime.__version__
    except ImportError:
        return ""


def _provider() -> str:
    try:
        import onnxruntime

        avail = onnxruntime.get_available_providers()
        return avail[0] if avail else ""
    except ImportError:
        return ""


def runtime_fingerprint(sdk_version: str) -> dict:
    """Static runtime identity — sent with every snapshot."""
    return {
        "sdkVersion": sdk_version,
        "pythonVersion": platform.python_version(),
        "ortVersion": _ort_version(),
        "os": platform.system(),
        "arch": platform.machine(),
        "provider": _provider(),
        "host": socket.gethostname(),
    }


def system_sample() -> dict:
    """Dynamic part of a snapshot (cheap; called every ~30s)."""
    return {"cpuPct": _cpu_pct(), "rssMb": _rss_mb()}
