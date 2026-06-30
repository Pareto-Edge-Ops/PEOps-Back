import { vi } from "vitest";

export interface MockRequest {
  url: string;
  path: string;
  method: string;
  headers: Record<string, string>;
  body: unknown;
}

export interface MockResponse {
  status?: number;
  json?: unknown;
  text?: string;
  bytes?: Uint8Array;
  headers?: Record<string, string>;
}

export type Handler = (req: MockRequest) => MockResponse | void;

/**
 * Replace global.fetch with a handler — the Node analogue of httpx.MockTransport
 * used in the Python SDK tests. Returns the captured request list so tests can
 * assert on URL/path/method/headers/body.
 */
export function installFetch(handler: Handler): { calls: MockRequest[] } {
  const calls: MockRequest[] = [];
  const fn = vi.fn(
    async (input: string | URL | Request, init: RequestInit = {}) => {
      const url = typeof input === "string" ? input : input.toString();
      const u = new URL(url);
      const headers: Record<string, string> = {};
      const h = (init.headers ?? {}) as Record<string, string>;
      for (const k of Object.keys(h)) headers[k.toLowerCase()] = h[k] as string;
      let body: unknown;
      if (typeof init.body === "string") {
        try {
          body = JSON.parse(init.body);
        } catch {
          body = init.body;
        }
      }
      const req: MockRequest = {
        url,
        path: u.pathname,
        method: (init.method ?? "GET").toUpperCase(),
        headers,
        body,
      };
      calls.push(req);
      const res = handler(req) ?? {};
      const status = res.status ?? 200;
      let payload: BodyInit;
      if (res.bytes !== undefined) payload = res.bytes;
      else if (res.text !== undefined) payload = res.text;
      else payload = JSON.stringify(res.json ?? {});
      return new Response(payload, { status, headers: res.headers });
    },
  );
  globalThis.fetch = fn as unknown as typeof fetch;
  return { calls };
}
