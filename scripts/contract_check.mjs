#!/usr/bin/env node
/**
 * Contract check — validates the LIVE backend JSON against the frontend's
 * ACTUAL zod schemas (PEOps-Front/src/features/*\/types.ts), loaded through
 * the front's own vite (ssrLoadModule transpiles the TS; vite + zod ship with
 * the front, this repo needs no node deps).
 *
 * Usage:
 *   node scripts/contract_check.mjs [--base http://localhost:8000] \
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

// Resolve vite from the FRONT's node_modules regardless of this script's home.
const frontRequire = createRequire(path.join(FRONT, "package.json"));
let vitePath;
try {
  vitePath = frontRequire.resolve("vite/dist/node/index.js"); // ESM entry
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

// Cookie jar — every /api route is session-gated, so we sign up once and
// replay the session cookie on every request (Node fetch has no jar).
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
  const email = `contract_${Date.now()}@peops.dev`;
  const r = await afetch(`${BASE}/api/auth/signup`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ email, password: "contract-pass-1234", name: "Contract" }),
  });
  if (!r.ok) throw new Error(`signup failed: HTTP ${r.status}`);
}

const load = (p) => vite.ssrLoadModule(p);
const [dash, models, arch, pareto, telemetry, sdk] = await Promise.all([
  load("/src/features/dashboard/types.ts"),
  load("/src/features/models/types.ts"),
  load("/src/features/architecture/types.ts"),
  load("/src/features/pareto/types.ts"),
  load("/src/features/telemetry/types.ts"),
  load("/src/features/sdk-hub/types.ts"),
]);

let pass = 0;
const failures = [];

/**
 * Provision a REAL model through the live backend (no fixtures exist —
 * the DB only ever contains real pipeline results). Returns its modelId.
 */
async function provisionModel() {
  if (args["model-id"]) return args["model-id"];
  const r = await afetch(`${BASE}/api/models/import`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ fileName: "contract-check.onnx" }),
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
  await afetch(`${BASE}/api/models/${modelId}/ingestion/complete`, { method: "POST" });
  console.log(" done");
  return modelId;
}

const arrayOf = (schema) => ({
  parse(value) {
    if (!Array.isArray(value)) throw new Error("expected an array");
    value.forEach((v) => schema.parse(v));
  },
});
const recordOf = (schema) => ({
  parse(value) {
    if (typeof value !== "object" || value === null || Array.isArray(value))
      throw new Error("expected an object record");
    Object.values(value).forEach((v) => schema.parse(v));
  },
});

async function check(label, url, schema) {
  try {
    const res = await afetch(BASE + url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    schema.parse(await res.json());
    pass++;
    console.log(`  ✓ ${label}`);
  } catch (err) {
    failures.push(`${label}: ${String(err.message ?? err).slice(0, 400)}`);
    console.log(`  ✗ ${label}`);
  }
}

console.log(`── zod contract check against ${BASE}\n   (schemas: ${FRONT}/src/features/*/types.ts)`);

// Authenticate, then provision a real model through the live pipeline.
await signup();
const MODEL = await provisionModel();

// dashboard (now backed by the real run that just completed)
await check("KpiSummary", "/api/dashboard/summary", dash.KpiSummary);
await check("DashboardRun[]", "/api/dashboard/runs", arrayOf(dash.DashboardRun));
await check("DashboardRun[] (filtered)", "/api/dashboard/runs?status=running", arrayOf(dash.DashboardRun));
await check("ParetoSnapshot", "/api/dashboard/pareto-snapshot", dash.ParetoSnapshot);
await check("TopModel[]", "/api/dashboard/top-models", arrayOf(dash.TopModel));
await check("ComputeCost", "/api/dashboard/compute-cost", dash.ComputeCost);
await check("ActivityEvent[]", "/api/dashboard/activity?limit=8", arrayOf(dash.ActivityEvent));

// models
await check("ModelListItem[]", "/api/models", arrayOf(models.ModelListItem));
await check("ModelListItem[] (sorted)", "/api/models?sort=bestAccuracy:desc&onlyDeployed=1", arrayOf(models.ModelListItem));
await check(`ModelListItem ${MODEL}`, `/api/models/${MODEL}`, models.ModelListItem);

// architecture + pareto — mapped from the actual ONNX graph + real Optuna trials
await check(`Architecture ${MODEL}`, `/api/models/${MODEL}/architecture`, arch.Architecture);
await check(`ParetoExperiment ${MODEL}`, `/api/models/${MODEL}/pareto`, pareto.ParetoExperiment);

// telemetry — derived from the real benchmark the pipeline just ran
await check("TelemetryKpi", `/api/models/${MODEL}/telemetry/kpi`, telemetry.TelemetryKpi);
await check("TelemetryPoint[]", `/api/models/${MODEL}/telemetry/series`, arrayOf(telemetry.TelemetryPoint));
await check("Percentiles", `/api/models/${MODEL}/telemetry/percentiles`, telemetry.Percentiles);
await check("Deployment[]", `/api/models/${MODEL}/telemetry/deployments`, arrayOf(telemetry.Deployment));
await check("Alert[]", `/api/models/${MODEL}/telemetry/alerts`, arrayOf(telemetry.Alert));

// sdk hub
await check("Record<string,SdkSnippet>", "/api/sdk/snippets", recordOf(sdk.SdkSnippet));
await check("Recipe[]", "/api/sdk/recipes", arrayOf(sdk.Recipe));

await vite.close();

console.log(`\n${"═".repeat(60)}\n  PASS ${pass}  ·  FAIL ${failures.length}`);
for (const f of failures) console.log(`  ✗ ${f}`);
process.exit(failures.length ? 1 : 0);
