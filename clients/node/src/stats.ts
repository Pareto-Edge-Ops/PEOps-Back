// Windowed input/output distribution statistics.
// Mirrors clients/python/peops_sdk/stats.py (numpy-free).
//
// A bounded reservoir of requests per window is reduced to compact stats:
// per-input tensor mean/std/min/max/NaN%, and — when the output looks
// classifier-shaped ([B, C], 1 < C <= 10000) — the argmax class distribution,
// a 16-bin top-1 confidence histogram, mean entropy and mean top-1 confidence.
// These windows are what the PEOps drift monitor compares against a deployment's
// reference to raise prediction/input drift alerts.

const RESERVOIR = 32;
const HIST_BINS = 16;
const MAX_CLASSES = 10_000;
const TOP_CLASSES = 10;

const round = (x: number, p: number): number => {
  const f = 10 ** p;
  return Math.round(x * f) / f;
};

/** A flat tensor: data buffer + dims. The numpy-free analogue of an ndarray. */
export interface Sampleable {
  data: ArrayLike<number | bigint>;
  dims: readonly number[];
}

type Inputs = Record<string, Sampleable>;

function softmax(row: number[]): number[] {
  let m = -Infinity;
  for (const x of row) if (x > m) m = x;
  const e = row.map((x) => Math.exp(x - m));
  let s = 0;
  for (const x of e) s += x;
  const denom = Math.max(1e-12, s);
  return e.map((x) => x / denom);
}

function valueHist(vals: number[], bins: number): number[] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of vals) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const counts = new Array<number>(bins).fill(0);
  const span = hi - lo;
  if (span === 0) {
    counts[0] = vals.length;
    return counts;
  }
  for (const v of vals) {
    let idx = Math.floor(((v - lo) / span) * bins);
    if (idx >= bins) idx = bins - 1;
    if (idx < 0) idx = 0;
    counts[idx] = (counts[idx] ?? 0) + 1;
  }
  return counts;
}

export class WindowAggregator {
  private windowStart!: Date;
  private nCount = 0;
  private seen = 0;
  private inputsRes: Inputs[] = [];
  private outputsRes: (Sampleable | null)[] = [];

  constructor() {
    this.reset();
  }

  private reset(): void {
    this.windowStart = new Date();
    this.nCount = 0;
    this.seen = 0;
    this.inputsRes = [];
    this.outputsRes = [];
  }

  /** Observed request count this window. */
  get n(): number {
    return this.nCount;
  }

  /** Current reservoir occupancy (bounded by RESERVOIR) — for tests. */
  get reservoirSize(): number {
    return this.inputsRes.length;
  }

  /** Reservoir-sample one request (cheap references only; never throws here). */
  observe(inputs: Inputs | null | undefined, output: Sampleable | null): void {
    this.nCount += 1;
    this.seen += 1;
    if (this.inputsRes.length < RESERVOIR) {
      this.store(inputs ?? {}, output);
    } else {
      const j = Math.floor(Math.random() * this.seen);
      if (j < RESERVOIR) this.store(inputs ?? {}, output, j);
    }
  }

  private store(inputs: Inputs, output: Sampleable | null, at?: number): void {
    if (at === undefined) {
      this.inputsRes.push(inputs);
      this.outputsRes.push(output);
    } else {
      this.inputsRes[at] = inputs;
      this.outputsRes[at] = output;
    }
  }

  /** Emit the window stats (null when nothing was observed) and reset. */
  flush(): Record<string, unknown> | null {
    if (this.nCount === 0) return null;
    let window: Record<string, unknown> | null;
    try {
      window = this.build();
    } catch {
      window = null; // stats must never break serving
    }
    this.reset();
    return window;
  }

  private build(): Record<string, unknown> {
    const inputStats: Record<string, unknown> = {};
    const names = new Set<string>();
    for (const s of this.inputsRes) for (const k of Object.keys(s)) names.add(k);

    for (const name of names) {
      const flat: number[] = [];
      for (const s of this.inputsRes) {
        const t = s[name];
        if (!t) continue;
        const d = t.data;
        const lim = Math.min(d.length, 4096);
        for (let i = 0; i < lim; i++) flat.push(Number(d[i]));
      }
      if (flat.length === 0) continue;

      let finiteCount = 0;
      let sum = 0;
      let min = Infinity;
      let max = -Infinity;
      for (const v of flat) {
        if (Number.isFinite(v)) {
          finiteCount += 1;
          sum += v;
          if (v < min) min = v;
          if (v > max) max = v;
        }
      }
      const nanPct = 100 * (1 - finiteCount / Math.max(1, flat.length));
      if (finiteCount === 0) {
        inputStats[name] = { mean: 0, std: 0, min: 0, max: 0, nanPct: round(nanPct, 3) };
        continue;
      }
      const mean = sum / finiteCount;
      let varSum = 0;
      for (const v of flat) if (Number.isFinite(v)) varSum += (v - mean) ** 2;
      inputStats[name] = {
        mean: round(mean, 6),
        std: round(Math.sqrt(varSum / finiteCount), 6),
        min: round(min, 6),
        max: round(max, 6),
        nanPct: round(nanPct, 3),
      };
    }

    let output: Record<string, unknown> = {};
    const outs = this.outputsRes.filter((o): o is Sampleable => o != null);
    if (outs.length) {
      const first = outs[0]!;
      const classes = first.dims.length === 2 ? Number(first.dims[1]) : 0;
      if (first.dims.length === 2 && classes > 1 && classes <= MAX_CLASSES) {
        const classCounts = new Map<number, number>();
        const hist = new Array<number>(HIST_BINS).fill(0);
        let confSum = 0;
        let entSum = 0;
        let rows = 0;
        for (const o of outs) {
          if (o.dims.length !== 2 || Number(o.dims[1]) !== classes) continue;
          const batch = Number(o.dims[0]);
          for (let r = 0; r < batch; r++) {
            const row: number[] = [];
            for (let c = 0; c < classes; c++) row.push(Number(o.data[r * classes + c]));
            const probs = softmax(row);
            let top = 0;
            for (let i = 1; i < classes; i++)
              if ((probs[i] ?? 0) > (probs[top] ?? 0)) top = i;
            const conf = probs[top] ?? 0;
            classCounts.set(top, (classCounts.get(top) ?? 0) + 1);
            const bin = Math.min(HIST_BINS - 1, Math.floor(conf * HIST_BINS));
            hist[bin] = (hist[bin] ?? 0) + 1;
            confSum += conf;
            let ent = 0;
            for (const p of probs) ent += -(p * Math.log(p + 1e-12));
            entSum += ent;
            rows += 1;
          }
        }
        if (rows) {
          const sorted = [...classCounts.entries()].sort((a, b) => b[1] - a[1]);
          const classDist: Record<string, number> = {};
          for (const [k, v] of sorted.slice(0, TOP_CLASSES))
            classDist[String(k)] = round(v / rows, 4);
          output = {
            classDist,
            hist,
            top1ConfMean: round(confSum / rows, 4),
            entropyMean: round(entSum / rows, 4),
          };
        }
      } else {
        const flat: number[] = [];
        const d = first.data;
        const lim = Math.min(d.length, 4096);
        for (let i = 0; i < lim; i++) {
          const v = Number(d[i]);
          if (Number.isFinite(v)) flat.push(v);
        }
        if (flat.length) output = { hist: valueHist(flat, HIST_BINS) };
      }
    }

    return {
      windowStart: this.windowStart.toISOString(),
      windowEnd: new Date().toISOString(),
      n: this.nCount,
      inputs: inputStats,
      output,
    };
  }
}
