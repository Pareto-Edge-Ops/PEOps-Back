// Host/system metrics for telemetry snapshots — stdlib only (os/process).
// Mirrors clients/python/astra_sdk/system.py, minus GPU: Node has no NVML, so
// GPU fields are honestly omitted rather than fabricated. ORT version/providers
// are passed in by the runner (which already imported onnxruntime-node) instead
// of re-importing the native binary here.

import * as os from "node:os";

const round1 = (x: number): number => Math.round(x * 10) / 10;

function platformToSystem(p: NodeJS.Platform): string {
  switch (p) {
    case "darwin":
      return "Darwin";
    case "linux":
      return "Linux";
    case "win32":
      return "Windows";
    default:
      return p.charAt(0).toUpperCase() + p.slice(1);
  }
}

function archToMachine(a: string): string {
  // Map Node's os.arch() to Python platform.machine() spellings so the dashboard
  // groups Node and Python hosts on the same hardware under one identity.
  switch (a) {
    case "x64":
      return "x86_64";
    case "ia32":
      return "i386";
    default:
      return a; // arm64, arm, ppc64, s390x map through unchanged
  }
}

function cpuModel(): string {
  const cpus = os.cpus();
  const name = cpus[0]?.model ?? "";
  return (name || os.arch() || "").slice(0, 96);
}

function cpuPct(): number {
  // 1-minute load average as a percent of cores — the same honest proxy the
  // Python SDK falls back to without psutil. os.loadavg() is [0,0,0] on Windows.
  const load1 = os.loadavg()[0] ?? 0;
  const n = os.cpus().length || 1;
  return round1(Math.min(100, (100 * load1) / n));
}

export interface OrtRuntimeInfo {
  ortVersion?: string;
  availableProviders?: string[];
  activeProvider?: string;
}

/** Static runtime + hardware identity — sent with every snapshot. */
export function runtimeFingerprint(
  sdkVersion: string,
  ort: OrtRuntimeInfo = {},
): Record<string, unknown> {
  const avail = ort.availableProviders ?? [];
  return {
    sdkVersion,
    pythonVersion: "", // node host — no python
    ortVersion: ort.ortVersion ?? "",
    os: platformToSystem(process.platform),
    arch: archToMachine(os.arch()),
    provider: avail[0] ?? ort.activeProvider ?? "",
    host: os.hostname(),
    cpuModel: cpuModel(),
    cpuCores: os.cpus().length || 0,
    ramTotalMb: round1(os.totalmem() / 1e6),
    availableProviders: avail.join(","),
    activeProvider: ort.activeProvider ?? avail[0] ?? "",
  };
}

/** Dynamic part of a snapshot (cheap; called every ~30s). */
export function systemSample(): Record<string, number> {
  return {
    cpuPct: cpuPct(),
    rssMb: round1(process.memoryUsage().rss / 1e6),
  };
}
