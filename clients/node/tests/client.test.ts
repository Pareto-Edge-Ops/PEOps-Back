import { describe, expect, it } from "vitest";

import { ApiError, HttpSession } from "../src/http.js";
import { InferenceError, PeopsClient } from "../src/client.js";
import { installFetch } from "./_mock.js";

describe("PeopsClient", () => {
  it("infer happy path hits the deployment URL with bearer auth", async () => {
    const { calls } = installFetch((req) => {
      expect(req.path).toBe("/api/v1/infer/dep_x");
      expect(req.headers["authorization"]).toBe("Bearer k");
      return { json: { latencyMs: 1.2, outputs: [] } };
    });
    const client = new PeopsClient("dep_x", "k", { baseUrl: "http://t" });
    const out = await client.infer({ input: [[1.0]] });
    expect(out.latencyMs).toBe(1.2);
    expect(calls).toHaveLength(1);
    await client.close();
  });

  it("maps a 4xx error body to InferenceError", async () => {
    installFetch(() => ({
      status: 404,
      json: { detail: { code: "deployment_not_found", message: "nope" } },
    }));
    const client = new PeopsClient("dep_x", "k", { baseUrl: "http://t" });
    await expect(client.infer()).rejects.toMatchObject({
      name: "InferenceError",
      code: "deployment_not_found",
      status: 404,
    });
    expect(InferenceError).toBeDefined();
  });
});

describe("HttpSession", () => {
  it("retries transient 503s and then succeeds", async () => {
    let n = 0;
    installFetch(() => {
      n += 1;
      return n < 3 ? { status: 503, json: {} } : { json: { ok: true } };
    });
    const s = new HttpSession("http://t", "k", { maxAttempts: 3, maxBackoff: 0 });
    const resp = await s.request("GET", "/x");
    expect(await resp.json()).toEqual({ ok: true });
    expect(n).toBe(3);
    s.close();
  });

  it("does not retry a 401 and raises ApiError", async () => {
    let n = 0;
    installFetch(() => {
      n += 1;
      return {
        status: 401,
        json: { detail: { code: "invalid_api_key", message: "bad" } },
      };
    });
    const s = new HttpSession("http://t", "k", { maxAttempts: 3, maxBackoff: 0 });
    await expect(s.request("GET", "/x")).rejects.toMatchObject({
      name: "ApiError",
      code: "invalid_api_key",
      status: 401,
    });
    expect(n).toBe(1);
    expect(ApiError).toBeDefined();
  });
});
