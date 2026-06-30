import { describe, expect, it } from "vitest";

import { WindowAggregator } from "../src/stats.js";

describe("WindowAggregator", () => {
  it("detects classifier-shaped output and summarizes it", () => {
    const agg = new WindowAggregator();
    for (let i = 0; i < 50; i++) {
      agg.observe(
        { x: { data: [0, 0, 0, 0], dims: [1, 4] } },
        { data: [0.1, 0.2, 5.0, 0.1], dims: [1, 4] },
      );
    }
    const w = agg.flush();
    expect(w).not.toBeNull();
    const out = w!.output as Record<string, unknown>;
    expect(w!.n).toBe(50);
    expect(out.classDist).toEqual({ "2": 1.0 });
    expect(out.top1ConfMean as number).toBeGreaterThan(0.9);
    expect((out.hist as number[]).reduce((a, b) => a + b, 0)).toBeGreaterThan(0);
  });

  it("falls back to a value histogram for non-classifier output", () => {
    const agg = new WindowAggregator();
    for (let i = 0; i < 5; i++) {
      agg.observe(
        { x: { data: [0, 0], dims: [1, 2] } },
        { data: new Array(64).fill(0), dims: [1, 8, 8] },
      );
    }
    const out = agg.flush()!.output as Record<string, unknown>;
    expect(out.classDist).toBeUndefined();
    expect((out.hist as number[]).length).toBe(16);
  });

  it("computes input stats including NaN%", () => {
    const agg = new WindowAggregator();
    agg.observe({ x: { data: [1, 2, NaN, 3], dims: [4] } }, null);
    const stats = (agg.flush()!.inputs as Record<string, Record<string, number>>).x;
    expect(stats!.nanPct).toBeCloseTo(25, 1);
    expect(stats!.mean).toBeCloseTo(2, 6);
  });

  it("resets after flush; empty flush returns null", () => {
    const agg = new WindowAggregator();
    agg.observe({ x: { data: [0, 0, 0], dims: [3] } }, null);
    expect(agg.flush()).not.toBeNull();
    expect(agg.flush()).toBeNull();
  });

  it("bounds the reservoir but counts every observation", () => {
    const agg = new WindowAggregator();
    for (let i = 0; i < 10_000; i++) {
      agg.observe({ x: { data: [0, 0], dims: [2] } }, null);
    }
    expect(agg.reservoirSize).toBeLessThanOrEqual(32);
    expect(agg.flush()!.n).toBe(10_000);
  });
});
