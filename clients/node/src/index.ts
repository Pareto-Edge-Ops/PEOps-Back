// astra-sdk — serve Astra-compressed models anywhere, with telemetry built in.
//
// Hosted inference (zero extra deps):
//
//     import { AstraClient } from "astra-sdk";
//     const client = new AstraClient(baseUrl, deploymentId, apiKey);
//     const out = await client.infer({ input: [[0.1, 0.2]] });
//
// Local serving (npm i onnxruntime-node) — pulls the compressed artifact and
// runs it on YOUR hardware while the dashboard keeps monitoring it:
//
//     import { LocalRunner } from "astra-sdk";
//     const runner = await LocalRunner.fromDeployment({ baseUrl, deploymentId, apiKey });
//     const out = await runner.run({ input: { data: myFloats, dims: [1, 3, 224, 224] } });
//     await runner.close();

export { ApiError } from "./http.js";
export { InferenceError, AstraClient } from "./client.js";
export { LocalRunner, pullArtifact, requireServeExtra, RunnerError } from "./runner.js";
export { TelemetryReporter, telemetryEnabled } from "./telemetry.js";
export { VERSION } from "./version.js";

export type { InferOptions } from "./client.js";
export type { FromDeploymentOptions, PullArtifactOptions } from "./runner.js";
export type { InputMeta } from "./tensor.js";
export type { RecordEventOptions, TelemetryReporterOptions } from "./telemetry.js";
export type { ArtifactInfo, RunInput, RunOutput, TensorData, TensorInput } from "./types.js";
