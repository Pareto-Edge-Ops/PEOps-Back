import { afterEach, describe, expect, it } from "vitest";

import { TelemetryReporter } from "../src/telemetry.js";
import { installFetch } from "./_mock.js";

interface Batch {
  clientId: string;
  events?: Record<string, unknown>[];
  snapshots?: Record<string, unknown>[];
  windows?: Record<string, unknown>[];
}

function makeBackend() {
  const batches: Batch[] = [];
  const state = { failNext: 0 };
  installFetch((req) => {
    expect(req.path).toBe("/api/v1/telemetry/dep_x/batch");
    if (state.failNext > 0) {
      state.failNext -= 1;
      return { status: 503, json: { detail: { code: "unavailable" } } };
    }
    batches.push(req.body as Batch);
    return { json: { accepted: {}, dropped: 0 } };
  });
  return {
    batches,
    setFail: (n: number) => {
      state.failNext = n;
    },
    events: () => batches.flatMap((b) => b.events ?? []),
    windows: () => batches.flatMap((b) => b.windows ?? []),
  };
}

function reporter(opts: Partial<{ enabled: boolean }> = {}) {
  return new TelemetryReporter("http://test", "dep_x", "peops_sk_test", {
    sdkVersion: "0.2.0",
    ...opts,
  });
}

afterEach(() => {
  delete process.env.PEOPS_SDK_TELEMETRY;
});

describe("TelemetryReporter", () => {
  it("flushes buffered events on close", async () => {
    const backend = makeBackend();
    const rep = reporter();
    for (let i = 0; i < 25; i++) {
      rep.recordEvent({ latencyMs: i, preMs: 0.1, postMs: 0.1 });
    }
    await rep.close();
    expect(backend.events()).toHaveLength(25);
    const ev = backend.events()[0]!;
    for (const k of ["ts", "latencyMs", "success", "batchSize", "region", "preMs", "postMs"]) {
      expect(ev).toHaveProperty(k);
    }
  });

  it("recording is cheap and never blocks the hot path", async () => {
    const backend = makeBackend();
    const rep = reporter();
    const t0 = performance.now();
    for (let i = 0; i < 5000; i++) rep.recordEvent({ latencyMs: 1.0 });
    expect(performance.now() - t0).toBeLessThan(1000);
    await rep.close();
    expect(backend.events().length).toBe(5000);
  });

  it("survives one failed flush and recovers the events", async () => {
    const backend = makeBackend();
    backend.setFail(1);
    const rep = reporter();
    for (let i = 0; i < 10; i++) rep.recordEvent({ latencyMs: i });
    await rep.close(); // close retries within its budget
    expect(backend.events()).toHaveLength(10);
  });

  it("is disabled via the enabled flag", async () => {
    const backend = makeBackend();
    const rep = reporter({ enabled: false });
    rep.recordEvent({ latencyMs: 1.0 });
    await rep.close();
    expect(backend.batches).toHaveLength(0);
  });

  it("is disabled via PEOPS_SDK_TELEMETRY=0", async () => {
    process.env.PEOPS_SDK_TELEMETRY = "0";
    const backend = makeBackend();
    const rep = reporter();
    rep.recordEvent({ latencyMs: 1.0 });
    await rep.close();
    expect(backend.batches).toHaveLength(0);
  });

  it("flushes the open window stats on close", async () => {
    const backend = makeBackend();
    const rep = reporter();
    for (let i = 0; i < 20; i++) {
      const x = Array.from({ length: 8 }, () => Math.random() - 0.5);
      const logits = Array.from({ length: 5 }, () => Math.random());
      rep.observe({ input: { data: x, dims: [1, 8] } }, { data: logits, dims: [1, 5] });
      rep.recordEvent({ latencyMs: 1.0 });
    }
    await rep.close();
    const windows = backend.windows();
    expect(windows.length).toBeGreaterThan(0);
    const w = windows[0]! as Record<string, unknown>;
    expect(w.n).toBe(20);
    const inputs = w.inputs as Record<string, Record<string, number>>;
    expect(inputs).toHaveProperty("input");
    expect(Math.abs(inputs.input!.mean!)).toBeLessThan(1.0);
    const output = w.output as Record<string, unknown>;
    expect(output).toHaveProperty("classDist");
    expect((output.hist as number[]).length).toBe(16);
  });
});
