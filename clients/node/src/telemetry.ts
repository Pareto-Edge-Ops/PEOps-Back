// Background telemetry reporter — fault-tolerant by construction.
// Mirrors clients/python/astra_sdk/telemetry.py.
//
// Design contract: NOTHING here may ever throw into the caller's serving path.
// Events buffer into a bounded array (drop-oldest under pressure, with the drop
// count itself reported); an unref'd interval timer flushes batches to
// POST /api/v1/telemetry/{deployment_id}/batch with backoff; a `beforeExit` hook
// performs a final best-effort flush within a small budget.
//
// Disable entirely with enabled:false or ASTRA_SDK_TELEMETRY=0.

import { randomBytes } from "node:crypto";

import { HttpSession } from "./http.js";
import { type Sampleable, WindowAggregator } from "./stats.js";
import { type OrtRuntimeInfo, runtimeFingerprint, systemSample } from "./system.js";

function envFloat(name: string, def: number): number {
  const v = process.env[name];
  if (v === undefined) return def;
  const n = Number(v);
  return Number.isFinite(n) ? n : def;
}

const QUEUE_MAX = 10_000;
const BATCH_MAX = 450; // below the server's 500-item cap
const FLUSH_INTERVAL_S = envFloat("ASTRA_SDK_FLUSH_INTERVAL_S", 5);
const SNAPSHOT_INTERVAL_S = envFloat("ASTRA_SDK_SNAPSHOT_INTERVAL_S", 30);
const WINDOW_INTERVAL_S = envFloat("ASTRA_SDK_WINDOW_INTERVAL_S", 60);
const WINDOW_MAX_REQUESTS = Math.trunc(envFloat("ASTRA_SDK_WINDOW_MAX_REQUESTS", 200));
const ATEXIT_BUDGET_MS = 3000;

const round = (x: number, p: number): number => {
  const f = 10 ** p;
  return Math.round(x * f) / f;
};

export function telemetryEnabled(flag?: boolean): boolean {
  if (flag === false) return false;
  const v = process.env.ASTRA_SDK_TELEMETRY;
  return !(v === "0" || v === "false" || v === "no");
}

export interface TelemetryReporterOptions {
  sdkVersion: string;
  enabled?: boolean;
  activeProvider?: string;
  ortVersion?: string;
  availableProviders?: string[];
}

export interface RecordEventOptions {
  latencyMs: number;
  preMs?: number;
  postMs?: number;
  success?: boolean;
  errorCode?: string;
  batchSize?: number;
  region?: string;
  inputSig?: string;
}

export class TelemetryReporter {
  readonly enabled: boolean;
  private readonly clientId: string;
  private readonly deploymentId: string;
  private readonly fingerprint: Record<string, unknown>;
  private readonly aggregator = new WindowAggregator();

  private events: Record<string, unknown>[] = [];
  private snapshots: Record<string, unknown>[] = [];
  private windows: Record<string, unknown>[] = [];
  private dropped = 0;
  private sentEvents = 0;
  private windowRequests = 0;

  private http: HttpSession | null = null;
  private timer: ReturnType<typeof setInterval> | null = null;
  private beforeExitHandler: (() => void) | null = null;
  private closed = false;
  private flushing = false;
  private lastSnapshot = Date.now();
  private lastWindow = Date.now();
  private throughputMarkerTime = Date.now();
  private throughputMarkerN = 0;

  constructor(
    baseUrl: string,
    deploymentId: string,
    apiKey: string,
    opts: TelemetryReporterOptions,
  ) {
    this.enabled = telemetryEnabled(opts.enabled);
    this.clientId = `sdk_${randomBytes(5).toString("hex")}`;
    this.deploymentId = deploymentId;
    const ort: OrtRuntimeInfo = {
      ortVersion: opts.ortVersion,
      availableProviders: opts.availableProviders,
      activeProvider: opts.activeProvider,
    };
    this.fingerprint = runtimeFingerprint(opts.sdkVersion, ort);

    if (this.enabled) {
      this.http = new HttpSession(baseUrl, apiKey, {
        timeout: 10,
        maxAttempts: 2,
        maxBackoff: 30,
      });
      this.timer = setInterval(() => {
        void this.tick();
      }, FLUSH_INTERVAL_S * 1000);
      this.timer.unref();
      this.beforeExitHandler = () => {
        void this.close();
      };
      process.once("beforeExit", this.beforeExitHandler);
    }
  }

  // ── recording (hot path — must be cheap and never raise) ──────────────────

  recordEvent(opts: RecordEventOptions): void {
    if (!this.enabled) return;
    try {
      const event: Record<string, unknown> = {
        ts: new Date().toISOString(),
        latencyMs: round(opts.latencyMs, 3),
        success: opts.success ?? true,
        batchSize: Math.trunc(opts.batchSize ?? 1),
        region: opts.region ?? "local",
      };
      if (opts.preMs !== undefined) event.preMs = round(opts.preMs, 3);
      if (opts.postMs !== undefined) event.postMs = round(opts.postMs, 3);
      if (opts.errorCode) event.errorCode = opts.errorCode;
      if (opts.inputSig) event.inputSig = opts.inputSig;

      if (this.events.length >= QUEUE_MAX) {
        this.events.shift();
        this.dropped += 1;
      }
      this.events.push(event);
      this.sentEvents += 1;
      this.windowRequests += 1;
      if (this.windowRequests >= WINDOW_MAX_REQUESTS) this.takeWindow();
    } catch {
      /* telemetry must never break serving */
    }
  }

  observe(inputs: Record<string, Sampleable> | null | undefined, output: Sampleable | null): void {
    if (!this.enabled) return;
    try {
      this.aggregator.observe(inputs, output);
    } catch {
      /* ignore */
    }
  }

  // ── background loop ───────────────────────────────────────────────────────

  private async tick(): Promise<void> {
    if (this.closed || this.flushing) return;
    this.flushing = true;
    try {
      const now = Date.now();
      if (now - this.lastSnapshot >= SNAPSHOT_INTERVAL_S * 1000) {
        this.takeSnapshot();
        this.lastSnapshot = now;
      }
      if (
        now - this.lastWindow >= WINDOW_INTERVAL_S * 1000 ||
        this.windowRequests >= WINDOW_MAX_REQUESTS
      ) {
        this.takeWindow();
        this.lastWindow = now;
      }
      await this.flush();
    } catch {
      /* the loop must survive anything */
    } finally {
      this.flushing = false;
    }
  }

  private takeSnapshot(): void {
    const now = Date.now();
    const elapsedMin = Math.max(1e-6, (now - this.throughputMarkerTime) / 60000);
    const rpm = (this.sentEvents - this.throughputMarkerN) / elapsedMin;
    this.throughputMarkerTime = now;
    this.throughputMarkerN = this.sentEvents;
    this.snapshots.push({
      ts: new Date().toISOString(),
      ...systemSample(),
      throughputRpm: round(rpm, 2),
      droppedEvents: this.dropped,
      ...this.fingerprint,
    });
    if (this.snapshots.length > 64) this.snapshots.shift();
  }

  private takeWindow(): void {
    this.windowRequests = 0;
    const w = this.aggregator.flush();
    if (w) {
      this.windows.push(w);
      if (this.windows.length > 64) this.windows.shift();
    }
  }

  private async flush(): Promise<boolean> {
    if (!this.http) return true;
    const events = this.events.splice(0, Math.min(BATCH_MAX, this.events.length));
    const snapshots = this.snapshots.splice(0, this.snapshots.length);
    const windows = this.windows.splice(0, this.windows.length);
    if (!events.length && !snapshots.length && !windows.length) return true;
    try {
      await this.http.request("POST", `/api/v1/telemetry/${this.deploymentId}/batch`, {
        json: { clientId: this.clientId, events, snapshots, windows },
      });
      return true;
    } catch {
      // Re-queue at the FRONT so ordering survives one failed flush; the bounded
      // queue drops oldest under sustained failure.
      for (let i = events.length - 1; i >= 0; i--) {
        if (this.events.length >= QUEUE_MAX) {
          this.dropped += 1;
          break;
        }
        this.events.unshift(events[i]!);
      }
      this.snapshots.unshift(...snapshots);
      this.windows.unshift(...windows);
      return false;
    }
  }

  // ── shutdown ──────────────────────────────────────────────────────────────

  /** Final best-effort flush within a small budget; idempotent. */
  async close(): Promise<void> {
    if (!this.enabled || this.closed) return;
    this.closed = true;
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
    if (this.beforeExitHandler) {
      process.removeListener("beforeExit", this.beforeExitHandler);
      this.beforeExitHandler = null;
    }
    try {
      // Always ship at least one snapshot per session — it carries the runtime
      // fingerprint the dashboard's client-hosts table shows (short sessions
      // would otherwise never hit the 30s cadence).
      this.takeSnapshot();
      this.takeWindow();
      const deadline = Date.now() + ATEXIT_BUDGET_MS;
      while (Date.now() < deadline) {
        const ok = await this.flush();
        if (ok && this.events.length === 0) break;
      }
    } catch {
      /* best effort */
    } finally {
      this.http?.close();
    }
  }
}
