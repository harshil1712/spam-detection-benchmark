// Inference shim: receives a model alias and a prepared Workers AI request
// body, runs it through the `AI` binding and reports the result together with
// the latency measured around `env.AI.run()`.
//
// Under `wrangler dev` no token is required. Once deployed, the
// BENCHMARK_TOKEN secret must be set and sent as a bearer token so a public
// URL cannot be used to run inference on your account.

import { resolveModel } from "../scripts/lib.mjs";

const LOCAL_HOSTS = new Set(["localhost", "127.0.0.1", "[::1]", "0.0.0.0"]);

function json(body, status = 200) {
  return Response.json(body, { status });
}

function isAuthorized(request, env, url) {
  if (!env.BENCHMARK_TOKEN) return LOCAL_HOSTS.has(url.hostname) ? true : "unconfigured";
  const header = request.headers.get("authorization") ?? "";
  const token = header.startsWith("Bearer ") ? header.slice(7) : null;
  return token !== null && timingSafeEqual(token, env.BENCHMARK_TOKEN);
}

function timingSafeEqual(a, b) {
  const encoder = new TextEncoder();
  const x = encoder.encode(a);
  const y = encoder.encode(b);
  let diff = x.length ^ y.length;
  for (let i = 0; i < Math.max(x.length, y.length); i++) diff |= (x[i] ?? 0) ^ (y[i] ?? 0);
  return diff === 0;
}

export async function classify(env, { model, input }) {
  const resolved = resolveModel(model);
  if (!resolved) return { status: 400, body: { ok: false, error: { message: `Unknown model: ${String(model)}` } } };
  if (!input || typeof input !== "object") return { status: 400, body: { ok: false, error: { message: "Missing input" } } };
  const started = Date.now();
  try {
    const result = await env.AI.run(resolved.id, input);
    return { status: 200, body: { ok: true, model: resolved.alias, modelId: resolved.id, latencyMs: Date.now() - started, result } };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const retryable = /429|rate limit|capacity|5\d\d|timeout|temporar/i.test(message);
    return {
      status: retryable ? 503 : 502,
      body: { ok: false, model: resolved.alias, modelId: resolved.id, latencyMs: Date.now() - started, error: { message } },
    };
  }
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/health") return json({ ok: true, binding: typeof env.AI?.run === "function" });
    if (url.pathname !== "/classify") return json({ ok: false, error: { message: "Not found" } }, 404);
    if (request.method !== "POST") return json({ ok: false, error: { message: "Method not allowed" } }, 405);
    const auth = isAuthorized(request, env, url);
    if (auth === "unconfigured") {
      return json({ ok: false, error: { message: "BENCHMARK_TOKEN secret is not configured on this deployment" } }, 503);
    }
    if (auth !== true) return json({ ok: false, error: { message: "Unauthorized" } }, 401);
    let payload;
    try {
      payload = await request.json();
    } catch {
      return json({ ok: false, error: { message: "Invalid JSON body" } }, 400);
    }
    const { status, body } = await classify(env, payload ?? {});
    return json(body, status);
  },
};
