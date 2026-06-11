"""LocalRunner — serve a PEOps-deployed artifact on your own hardware.

    from peops_sdk import LocalRunner

    runner = LocalRunner.from_deployment(
        base_url="https://app.example.com",
        deployment_id="dep_ab12cd34ef",
        api_key="peops_sk_live_…",
    )
    out = runner.run({"input": my_numpy_array})    # local onnxruntime inference
    runner.close()

The artifact is pulled once via the API-key-authed
GET /api/v1/artifacts/{deployment_id} and cached on disk keyed by its sha256,
so restarts don't re-download. Every run() is measured (pre/infer/post) and
shipped to the PEOps dashboard by the background TelemetryReporter, together
with periodic system snapshots and windowed input/output stats — the same
closed loop hosted serving gets, but on your hardware.

Requires the [serve] extra:  pip install 'peops-sdk[serve]'
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ._http import HttpSession
from .telemetry import TelemetryReporter

_DEFAULT_CACHE = "~/.cache/peops"


class RunnerError(Exception):
    pass


def _require_serve_extra():
    try:
        import numpy
        import onnxruntime
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise RunnerError(
            "Local serving needs onnxruntime + numpy — install the extra: "
            "pip install 'peops-sdk[serve]'"
        ) from exc
    return numpy, onnxruntime


class LocalRunner:
    """Local ONNX serving with built-in telemetry."""

    def __init__(
        self,
        model_path: str,
        *,
        reporter: TelemetryReporter | None = None,
        providers: list[str] | None = None,
    ) -> None:
        np, ort = _require_serve_extra()
        self._np = np
        self._reporter = reporter
        self._session = ort.InferenceSession(
            model_path,
            providers=providers or ort.get_available_providers(),
        )
        self._input_meta = self._session.get_inputs()
        self._output_names = [o.name for o in self._session.get_outputs()]
        self.model_path = model_path

    # ── construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_deployment(
        cls,
        base_url: str,
        deployment_id: str,
        api_key: str,
        *,
        cache_dir: str = _DEFAULT_CACHE,
        report_telemetry: bool = True,
        providers: list[str] | None = None,
        timeout: float = 60.0,
    ) -> "LocalRunner":
        """Pull (or reuse) the deployed artifact and build a runner for it."""
        _require_serve_extra()
        path = pull_artifact(
            base_url, deployment_id, api_key,
            cache_dir=cache_dir, timeout=timeout,
        )
        from . import __version__

        reporter = TelemetryReporter(
            base_url, deployment_id, api_key,
            sdk_version=__version__, enabled=report_telemetry,
        )
        return cls(str(path), reporter=reporter, providers=providers)

    # ── serving ──────────────────────────────────────────────────────────────

    def run(
        self,
        inputs: dict[str, Any] | None = None,
        *,
        region: str = "local",
    ) -> dict[str, Any]:
        """One inference. `inputs` maps input name → ndarray/nested list; pass
        None to synthesize a random valid probe (smoke tests / benchmarks)."""
        np = self._np
        t0 = time.perf_counter()
        error_code: str | None = None
        feeds: dict[str, Any] = {}
        try:
            feeds = self._prepare(inputs)
            pre_ms = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            raw = self._session.run(self._output_names, feeds)
            infer_ms = (time.perf_counter() - t1) * 1000

            t2 = time.perf_counter()
            outputs = [
                {"name": n, "shape": list(np.asarray(o).shape)}
                for n, o in zip(self._output_names, raw)
            ]
            first = raw[0] if raw else None
            if self._reporter is not None:
                self._reporter.observe(feeds, first)
            post_ms = (time.perf_counter() - t2) * 1000
        except Exception as exc:
            error_code = "bad_input" if isinstance(exc, (ValueError, KeyError)) else "inference_error"
            if self._reporter is not None:
                self._reporter.record_event(
                    latency_ms=0.0, success=False, error_code=error_code,
                    batch_size=self._batch_of(feeds), region=region,
                    input_sig=self._signature(feeds),
                )
            raise

        if self._reporter is not None:
            self._reporter.record_event(
                latency_ms=infer_ms, pre_ms=pre_ms, post_ms=post_ms,
                success=True, batch_size=self._batch_of(feeds), region=region,
                input_sig=self._signature(feeds),
            )
        return {
            "latencyMs": round(infer_ms, 3),
            "preMs": round(pre_ms, 3),
            "postMs": round(post_ms, 3),
            "outputs": outputs,
            "raw": raw,
        }

    def _prepare(self, inputs: dict[str, Any] | None) -> dict[str, Any]:
        np = self._np
        feeds: dict[str, Any] = {}
        if inputs is None:
            for meta in self._input_meta:
                shape = [d if isinstance(d, int) and d > 0 else 1 for d in meta.shape]
                feeds[meta.name] = np.random.default_rng().standard_normal(
                    shape).astype(np.float32)
            return feeds
        for meta in self._input_meta:
            if meta.name not in inputs:
                raise ValueError(f"missing input '{meta.name}'")
            arr = np.asarray(inputs[meta.name])
            if arr.dtype.kind == "f":
                arr = arr.astype(np.float32)
            feeds[meta.name] = arr
        return feeds

    def _batch_of(self, feeds: dict[str, Any]) -> int:
        for arr in feeds.values():
            shape = getattr(arr, "shape", None)
            if shape:
                return int(shape[0])
        return 1

    def _signature(self, feeds: dict[str, Any]) -> str | None:
        parts = []
        for name, arr in feeds.items():
            shape = getattr(arr, "shape", ())
            dtype = getattr(arr, "dtype", "")
            parts.append(f"{name}:{'x'.join(str(d) for d in shape)}:{dtype}")
        return ";".join(parts) or None

    # ── lifecycle ────────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._reporter is not None:
            self._reporter.close()

    def __enter__(self) -> "LocalRunner":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def pull_artifact(
    base_url: str,
    deployment_id: str,
    api_key: str,
    *,
    cache_dir: str = _DEFAULT_CACHE,
    timeout: float = 60.0,
) -> Path:
    """Download the deployed artifact (sha256-cached under cache_dir)."""
    http = HttpSession(base_url, api_key, timeout=timeout)
    try:
        info = http.request(
            "GET", f"/api/v1/artifacts/{deployment_id}/info").json()
        sha = info["sha256"]
        root = Path(cache_dir).expanduser() / deployment_id
        root.mkdir(parents=True, exist_ok=True)
        target = root / f"{sha}{Path(info['fileName']).suffix or '.onnx'}"
        if target.exists() and target.stat().st_size == info["sizeBytes"]:
            return target
        resp = http.request("GET", f"/api/v1/artifacts/{deployment_id}")
        tmp = target.with_suffix(".part")
        tmp.write_bytes(resp.content)
        tmp.replace(target)
        (root / "meta.json").write_text(json.dumps(info, indent=2))
        return target
    finally:
        http.close()
