// Thin HTTP client for the Astra hosted inference endpoint.
// Mirrors clients/python/astra_sdk/client.py.

import { resolveBaseUrl } from "./http.js";

export class InferenceError extends Error {
  readonly status: number;
  readonly code: string;
  constructor(status: number, code: string, message: string) {
    super(`[${status}] ${code}: ${message}`);
    this.name = "InferenceError";
    this.status = status;
    this.code = code;
  }
}

export interface InferOptions {
  region?: string;
  batch?: number;
}

/** Minimal client for POST /api/v1/infer/{deployment_id}. */
export class AstraClient {
  private readonly baseUrl: string;
  readonly deploymentId: string;
  private readonly apiKey: string;
  private readonly timeoutMs: number;

  constructor(
    deploymentId: string,
    apiKey: string,
    opts: { baseUrl?: string; timeout?: number } = {},
  ) {
    this.baseUrl = resolveBaseUrl(opts.baseUrl);
    this.deploymentId = deploymentId;
    this.apiKey = apiKey;
    this.timeoutMs = (opts.timeout ?? 30) * 1000;
  }

  private get url(): string {
    return `${this.baseUrl}/api/v1/infer/${this.deploymentId}`;
  }

  /**
   * Run one inference. `inputs` maps input name → nested list; pass null to let
   * the server synthesize a valid random probe (handy for smoke tests).
   */
  async infer(
    inputs: Record<string, unknown> | null = null,
    opts: InferOptions = {},
  ): Promise<Record<string, unknown>> {
    const body: Record<string, unknown> = { inputs };
    if (opts.region !== undefined) body.region = opts.region;
    if (opts.batch !== undefined) body.batch = opts.batch;

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    let resp: Response;
    try {
      resp = await fetch(this.url, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${this.apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
    } finally {
      clearTimeout(timer);
    }

    if (resp.status >= 400) {
      const raw = await resp.text();
      let code = "error";
      let message = raw;
      try {
        const detail = (JSON.parse(raw) as { detail?: unknown }).detail;
        if (detail && typeof detail === "object") {
          const d = detail as Record<string, unknown>;
          code = typeof d.code === "string" ? d.code : "error";
          if (typeof d.message === "string") message = d.message;
        }
      } catch {
        /* not JSON — keep raw text */
      }
      throw new InferenceError(resp.status, code, message);
    }
    return (await resp.json()) as Record<string, unknown>;
  }

  // Symmetry with the Python context-manager API; nothing to release.
  async close(): Promise<void> {
    /* no-op */
  }
}
