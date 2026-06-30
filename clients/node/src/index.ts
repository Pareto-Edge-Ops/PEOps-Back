// peops-sdk — serve PEOps-compressed models anywhere, with telemetry built in.
//
// Hosted inference (zero extra deps):
//
//     import { PeopsClient } from "peops-sdk";
//     const client = new PeopsClient(baseUrl, deploymentId, apiKey);
//     const out = await client.infer({ input: [[0.1, 0.2]] });
//
// Local serving (npm i onnxruntime-node) — pulls the compressed artifact and
// runs it on YOUR hardware while the dashboard keeps monitoring it:
//
//     import { LocalRunner } from "peops-sdk";
//     const runner = await LocalRunner.fromDeployment({ baseUrl, deploymentId, apiKey });
//     const out = await runner.run({ input: { data: myFloats, dims: [1, 3, 224, 224] } });
//     await runner.close();

export { ApiError } from "./http.js";
export { InferenceError, PeopsClient } from "./client.js";
export { LocalRunner, pullArtifact, requireServeExtra, RunnerError } from "./runner.js";
export { TelemetryReporter, telemetryEnabled } from "./telemetry.js";
export { VERSION } from "./version.js";

export type { InferOptions } from "./client.js";
export type { FromDeploymentOptions, PullArtifactOptions } from "./runner.js";
export type { InputMeta } from "./tensor.js";
export type { RecordEventOptions, TelemetryReporterOptions } from "./telemetry.js";
export type { ArtifactInfo, RunInput, RunOutput, TensorData, TensorInput } from "./types.js";
