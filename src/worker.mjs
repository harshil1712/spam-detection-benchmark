// Inference shim: receives a model alias and a prepared Workers AI request
// body, runs it through the `AI` binding and returns the result together with
// the latency measured around `env.AI.run()`.
//
// Under `wrangler dev` (localhost) no token is required. Once deployed, the
// BENCHMARK_TOKEN secret must be set and sent as a bearer token so a public
// URL cannot be used to run inference on your account.

import { MODELS } from "./lib.mjs";

const LOCAL_HOSTS = new Set(["localhost", "127.0.0.1", "[::1]"]);

// Constant-time comparison; `crypto.subtle.timingSafeEqual` is Workers-only and
// this module is also unit-tested under Node.
function tokensMatch(header, secret) {
  const provided = header?.replace(/^Bearer /, "") ?? "";
  let diff = provided.length ^ secret.length;
  for (let i = 0; i < secret.length; i++) diff |= provided.charCodeAt(i) ^ secret.charCodeAt(i);
  return diff === 0;
}

function authorize(request, env) {
  if (env.BENCHMARK_TOKEN) {
    return tokensMatch(request.headers.get("authorization"), env.BENCHMARK_TOKEN) ? null : [401, "Unauthorized"];
  }
  if (LOCAL_HOSTS.has(new URL(request.url).hostname)) return null;
  return [503, "BENCHMARK_TOKEN secret is not configured on this deployment"];
}

export async function classify(env, { model, input } = {}) {
  const resolved = MODELS[model];
  if (!resolved) return [400, { ok: false, error: `Unknown model: ${String(model)}` }];
  if (!input || typeof input !== "object") return [400, { ok: false, error: "Missing input" }];
  const started = Date.now();
  try {
    const result = await env.AI.run(resolved.id, input);
    return [200, { ok: true, modelId: resolved.id, latencyMs: Date.now() - started, result }];
  } catch (error) {
    return [502, { ok: false, modelId: resolved.id, error: error instanceof Error ? error.message : String(error) }];
  }
}

export default {
  async fetch(request, env) {
    const { pathname } = new URL(request.url);
    if (pathname === "/health") return Response.json({ ok: true, binding: typeof env.AI?.run === "function" });
    if (pathname !== "/classify") return Response.json({ ok: false, error: "Not found" }, { status: 404 });
    if (request.method !== "POST") return Response.json({ ok: false, error: "Method not allowed" }, { status: 405 });
    const denied = authorize(request, env);
    if (denied) return Response.json({ ok: false, error: denied[1] }, { status: denied[0] });
    let payload;
    try {
      payload = await request.json();
    } catch {
      return Response.json({ ok: false, error: "Invalid JSON body" }, { status: 400 });
    }
    const [status, body] = await classify(env, payload);
    return Response.json(body, { status });
  },
};
