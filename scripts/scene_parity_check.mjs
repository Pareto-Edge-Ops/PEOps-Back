#!/usr/bin/env node
/**
 * Scene parity check — recomputes every 3D-scene value in JavaScript (the
 * same IEEE-754 arithmetic the SPA runs) and compares it `===` against the
 * live backend's /architecture/scene and /pareto/scene responses.
 *
 * Uses the front's ACTUAL functions where importable:
 *   - mapRange  (src/lib/three/scales.ts)         → pareto point positions
 *   - viridis   (src/lib/three/colorRamp.ts)      → sensitivity colormap
 * and inlines the (non-exported) constants/formulas from LayerGraph3D.tsx /
 * ParetoFrontierPlot3D.tsx / AxesGrid.tsx.
 *
 * Usage:
 *   node scripts/scene_parity_check.mjs [--base http://localhost:8000] \
 *        [--front ../PEOps-Front] [--model-id m_uploaded_xxx]
 */

import { createRequire } from "node:module";
import path from "node:path";
import { pathToFileURL } from "node:url";
import process from "node:process";

const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, a, i, arr) => {
    if (a.startsWith("--")) acc.push([a.slice(2), arr[i + 1]]);
    return acc;
  }, []),
);
const BASE = args.base ?? "http://localhost:8000";
const FRONT = path.resolve(args.front ?? "../PEOps-Front");

const frontRequire = createRequire(path.join(FRONT, "package.json"));
let vitePath;
try {
  vitePath = frontRequire.resolve("vite/dist/node/index.js");
} catch {
  vitePath = frontRequire.resolve("vite");
}
const viteMod = await import(pathToFileURL(vitePath).href);
const createServer = viteMod.createServer ?? viteMod.default?.createServer;
const vite = await createServer({
  root: FRONT,
  configFile: false,
  // Separate cache — sharing node_modules/.vite with a running dev server
  // corrupts its optimized-deps state (permanent 504 Outdated Optimize Dep).
  cacheDir: `/tmp/.vite-${path.basename(import.meta.url).replace(/\W/g, "_")}`,
  server: { middlewareMode: true, hmr: false },
  appType: "custom",
  logLevel: "error",
});

const { mapRange } = await vite.ssrLoadModule("/src/lib/three/scales.ts");
const { viridis } = await vite.ssrLoadModule("/src/lib/three/colorRamp.ts");

let pass = 0;
const failures = [];
function report(label, mismatches, total) {
  if (mismatches.length === 0) {
    pass++;
    console.log(`  ✓ ${label} (${total} values)`);
  } else {
    failures.push(`${label}: ${mismatches.length}/${total} mismatches — first: ${mismatches[0]}`);
    console.log(`  ✗ ${label} (${mismatches.length}/${total} mismatches)`);
  }
}

// ── LayerGraph3D constants/formulas (component-internal, mirrored) ─────────
const COL_SPACING = 1.55;
const ROW_SPACING = 0.22;
const SENS_THRESHOLD = 0.55;
const KIND_WIDTH = {
  input: 1, output: 1, embed: 18, conv: 16, attn: 16, ffn: 20,
  dense: 14, lstm: 14, bn: 10, norm: 10, relu: 8, pool: 8,
  softmax: 8, upsample: 16,
};
const widthFor = (node) =>
  typeof node.width === "number" && node.width > 0
    ? node.width
    : (KIND_WIDTH[node.kind] ?? 12);

// Cookie jar — every /api route is session-gated (Node fetch has no jar).
let COOKIE = "";
async function afetch(url, opts = {}) {
  const res = await fetch(url, {
    ...opts,
    headers: { ...(opts.headers ?? {}), ...(COOKIE ? { cookie: COOKIE } : {}) },
  });
  const setCookie = res.headers.get("set-cookie");
  if (setCookie) COOKIE = setCookie.split(";")[0];
  return res;
}
async function signup() {
  const email = `scene_${Date.now()}@peops.dev`;
  const r = await afetch(`${BASE}/api/auth/signup`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ email, password: "scene-pass-1234", name: "Scene" }),
  });
  if (!r.ok) throw new Error(`signup failed: HTTP ${r.status}`);
}

async function getJson(url) {
  const res = await afetch(BASE + url);
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  return res.json();
}

async function checkArchScene(modelId) {
  const arch = await getJson(`/api/models/${modelId}/architecture`);
  const scene = await getJson(`/api/models/${modelId}/architecture/scene`);
  const mm = [];
  let total = 0;

  // 1) neuron positions — exact LayerGraph3D math in JS doubles
  let cursor = 0;
  arch.nodes.forEach((node, li) => {
    const w = widthFor(node);
    const layer = scene.layers[li];
    if (layer.effectiveWidth !== w) mm.push(`layer ${node.id} width ${layer.effectiveWidth}≠${w}`);
    if (layer.neuronStart !== cursor) mm.push(`layer ${node.id} start`);
    const x = node.depth * COL_SPACING;
    const cz = node.zCol ?? 0;
    for (let i = 0; i < w; i++) {
      const n = scene.neurons[cursor + i];
      const y = node.col + (i - (w - 1) / 2) * ROW_SPACING;
      total += 3;
      if (n.x !== x) mm.push(`${node.id}[${i}].x ${n.x}≠${x}`);
      if (n.y !== y) mm.push(`${node.id}[${i}].y ${n.y}≠${y}`);
      if (n.z !== cz) mm.push(`${node.id}[${i}].z ${n.z}≠${cz}`);
    }
    cursor += w;
  });
  if (scene.counts.neurons !== cursor) mm.push(`neuron count ${scene.counts.neurons}≠${cursor}`);

  // 2) sensitivity colors via the front's REAL viridis()
  for (const layer of scene.layers) {
    total += 2;
    const [r, g, b] = viridis(layer.sensitivity);
    const hex = `#${[r, g, b].map((v) => v.toString(16).padStart(2, "0")).join("")}`;
    if (layer.colors.viridis !== hex) mm.push(`${layer.id} viridis ${layer.colors.viridis}≠${hex}`);
    const sens = layer.sensitivity >= SENS_THRESHOLD;
    const expected = sens ? "#7783e3" : "#6a6b6e";
    if (layer.colors.sensitivity !== expected) mm.push(`${layer.id} sens color`);
    if (layer.isSensitive !== sens) mm.push(`${layer.id} isSensitive`);
  }

  // 3) camera framing (component effect formula)
  const ns = scene.neurons;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity;
  for (const n of ns) {
    if (n.x < minX) minX = n.x; if (n.x > maxX) maxX = n.x;
    if (n.y < minY) minY = n.y; if (n.y > maxY) maxY = n.y;
    if (n.z < minZ) minZ = n.z; if (n.z > maxZ) maxZ = n.z;
  }
  const xC = (minX + maxX) / 2, yC = (minY + maxY) / 2, zC = (minZ + maxZ) / 2;
  const xSpan = maxX - minX + 1.2, ySpan = maxY - minY + 1.2;
  const zSpan = Math.max(0.001, maxZ - minZ);
  const fovRad = 38 * (Math.PI / 180);
  const zForY = ySpan / 2 / Math.tan(fovRad / 2);
  const zForX = xSpan / 2 / (Math.tan(fovRad / 2) * 1.55);
  const dist = Math.max(10, Math.max(zForX, zForY) + 1.5 + zSpan * 0.6);
  const camExpected = [
    xC + Math.min(dist * 0.18, xSpan * 0.18),
    yC + Math.min(dist * 0.18, ySpan * 0.55 + 1.2),
    zC + dist,
  ];
  total += 6;
  camExpected.forEach((v, i) => {
    if (scene.camera.position[i] !== v) mm.push(`camera.position[${i}] ${scene.camera.position[i]}≠${v}`);
  });
  [xC, yC, zC].forEach((v, i) => {
    if (scene.camera.target[i] !== v) mm.push(`camera.target[${i}]`);
  });

  // 4) bipartite edge segment counts
  const idx = new Map(arch.nodes.map((n, i) => [n.id, i]));
  arch.edges.forEach((e, i) => {
    total += 1;
    const exp = widthFor(arch.nodes[idx.get(e.from)]) * widthFor(arch.nodes[idx.get(e.to)]);
    if (scene.edges[i].segmentCount !== exp) mm.push(`edge ${e.from}→${e.to} segments`);
  });

  report(`architecture/scene ${modelId}`, mm, total);
}

async function checkParetoScene(modelId) {
  const exp = await getJson(`/api/models/${modelId}/pareto`);
  const scene = await getJson(`/api/models/${modelId}/pareto/scene`);
  const mm = [];
  let total = 0;

  const AXIS = 4;
  const pad = (lo, hi, frac = 0.06) => {
    const span = hi - lo;
    const m = span * frac;
    return [lo - m, hi + m];
  };
  const lat = exp.trials.map((t) => t.latency);
  const acc = exp.trials.map((t) => t.accuracy);
  const size = exp.trials.map((t) => t.size);
  const latD = pad(Math.min(...lat), Math.max(...lat));
  const accD = pad(Math.min(...acc), Math.max(...acc));
  const sizeD = pad(Math.min(...size), Math.max(...size));

  // category highlights — exact mirror of features/pareto/lib/highlights.ts
  const HL_COLORS = { best: "#FFC857", accuracy: "#40BF6B", size: "#5EEAD4", latency: "#F29926" };
  const argBy = (arr, fn, better) => arr.reduce((a, b) => (better(fn(b), fn(a)) ? b : a));
  const hlById = {};
  if (exp.trials.length) {
    hlById[argBy(exp.trials, (t) => t.latency, (b, a) => b < a).id] = "latency";
    hlById[argBy(exp.trials, (t) => t.size, (b, a) => b < a).id] = "size";
    hlById[argBy(exp.trials, (t) => t.accuracy, (b, a) => b > a).id] = "accuracy";
    hlById[argBy(exp.trials, (t) => t.score, (b, a) => b > a).id] = "best";
  }

  total += 6;
  if (scene.axis.x.domain[0] !== latD[0] || scene.axis.x.domain[1] !== latD[1]) mm.push("x domain");
  if (scene.axis.y.domain[0] !== accD[0] || scene.axis.y.domain[1] !== accD[1]) mm.push("y domain");
  if (scene.axis.z.domain[0] !== sizeD[0] || scene.axis.z.domain[1] !== sizeD[1]) mm.push("z domain");

  // point positions via the front's REAL mapRange()
  exp.trials.forEach((t, i) => {
    const p = scene.points[i];
    total += 3;
    const x = mapRange(t.latency, latD[0], latD[1], 0, AXIS);
    const y = mapRange(t.accuracy, accD[0], accD[1], 0, AXIS);
    const z = mapRange(t.size, sizeD[0], sizeD[1], 0, AXIS);
    if (p.position.x !== x) mm.push(`${t.id}.x ${p.position.x}≠${x}`);
    if (p.position.y !== y) mm.push(`${t.id}.y ${p.position.y}≠${y}`);
    if (p.position.z !== z) mm.push(`${t.id}.z ${p.position.z}≠${z}`);
    // colors / scales / dimming (budget defaults) — incl. category highlights
    total += 3;
    const hl = hlById[t.id];
    const expColor = hl ? HL_COLORS[hl] : t.onFrontier ? "#E1FF6B" : "#ADB4F3";
    const expScale = hl ? 1.8 : t.onFrontier ? 1.4 : 1;
    if (p.color !== expColor) mm.push(`${t.id} color`);
    if (p.scale !== expScale) mm.push(`${t.id} scale`);
    if ((p.highlight ?? null) !== (hl ?? null)) mm.push(`${t.id} highlight`);
    const passes =
      t.latency <= exp.budget.maxLatency &&
      exp.baseAccuracy - t.accuracy <= exp.budget.maxAccuracyDrop &&
      t.size <= exp.budget.maxSize;
    if (p.dimmed !== (hl ? false : !passes)) mm.push(`${t.id} dimmed`);
    // tooltip strings vs JS toFixed
    total += 4;
    const fmtMs = (v) => (v >= 100 ? `${v.toFixed(0)} ms` : v >= 10 ? `${v.toFixed(1)} ms` : `${v.toFixed(2)} ms`);
    const fmtMb = (v) => (v >= 100 ? `${v.toFixed(0)} MB` : v >= 10 ? `${v.toFixed(1)} MB` : `${v.toFixed(2)} MB`);
    if (p.tooltip.accuracy !== `${t.accuracy.toFixed(2)}%`) mm.push(`${t.id} tip.acc "${p.tooltip.accuracy}"≠"${t.accuracy.toFixed(2)}%"`);
    if (p.tooltip.latency !== fmtMs(t.latency)) mm.push(`${t.id} tip.lat "${p.tooltip.latency}"≠"${fmtMs(t.latency)}"`);
    if (p.tooltip.size !== fmtMb(t.size)) mm.push(`${t.id} tip.size`);
    if (p.tooltip.score !== t.score.toFixed(1)) mm.push(`${t.id} tip.score`);
  });

  // axis ticks (AxesGrid tickValue formula + adaptive formatters)
  for (const [key, dom, fmt] of [
    ["x", latD, (v) => (v >= 100 ? `${v.toFixed(0)} ms` : v >= 10 ? `${v.toFixed(1)} ms` : `${v.toFixed(2)} ms`)],
    ["y", accD, (v) => `${v.toFixed(1)}%`],
    ["z", sizeD, (v) => (v >= 100 ? `${v.toFixed(0)} MB` : v >= 10 ? `${v.toFixed(1)} MB` : `${v.toFixed(2)} MB`)],
  ]) {
    for (let i = 0; i <= 4; i++) {
      total += 3;
      const tick = scene.axis[key].ticks[i];
      const value = dom[0] + ((dom[1] - dom[0]) * i) / 4;
      if (tick.position !== (i * AXIS) / 4) mm.push(`${key} tick ${i} pos`);
      if (tick.value !== value) mm.push(`${key} tick ${i} value ${tick.value}≠${value}`);
      if (tick.label !== fmt(value)) mm.push(`${key} tick ${i} label "${tick.label}"≠"${fmt(value)}"`);
    }
  }

  report(`pareto/scene ${modelId}`, mm, total);
}

console.log(`── scene parity check against ${BASE}\n   (front fns from ${FRONT}: mapRange, viridis)`);

// All data is REAL — provision a model through the live pipeline (or use --model-id).
async function provisionModel() {
  if (args["model-id"]) return args["model-id"];
  const r = await afetch(`${BASE}/api/models/import`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ fileName: "scene-parity.onnx" }),
  });
  if (!r.ok) throw new Error(`import failed: HTTP ${r.status}`);
  const { modelId, runId } = await r.json();
  process.stdout.write(`   provisioning real model ${modelId} `);
  const deadline = Date.now() + 180_000;
  for (;;) {
    const s = await (await afetch(`${BASE}/api/models/${modelId}/ingestion/${runId}`)).json();
    if (s.status !== "streaming") {
      if (s.status !== "completed") throw new Error(`pipeline ${s.status}: ${s.error ?? ""}`);
      break;
    }
    if (Date.now() > deadline) throw new Error("pipeline timed out");
    process.stdout.write(".");
    await new Promise((res) => setTimeout(res, 1000));
  }
  console.log(" done");
  return modelId;
}

await signup();
const MODEL = await provisionModel();
await checkArchScene(MODEL);
await checkParetoScene(MODEL);

await vite.close();

console.log(`\n${"═".repeat(60)}\n  PASS ${pass}  ·  FAIL ${failures.length}`);
for (const f of failures) console.log(`  ✗ ${f}`);
process.exit(failures.length ? 1 : 0);
