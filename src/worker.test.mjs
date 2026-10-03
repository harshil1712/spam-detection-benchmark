import { test } from "node:test";
import assert from "node:assert/strict";
import worker, { classify } from "./worker.mjs";

const makeEnv = (overrides = {}) => ({
  AI: {
    calls: [],
    async run(model, input) {
      this.calls.push({ model, input });
      if (overrides.fail) throw new Error(overrides.fail);
      return { answers: { spam: { type: "noul", noul: 0.9 } } };
    },
  },
  ...overrides.env,
});

const post = (url, body, headers = {}) =>
  new Request(url, { method: "POST", headers: { "content-type": "application/json", ...headers }, body: JSON.stringify(body) });

test("classify runs the binding with the resolved model ID and measures latency", async () => {
  const env = makeEnv();
  const { status, body } = await classify(env, { model: "clef", input: { model: "clef", state: "s", questions: {} } });
  assert.equal(status, 200);
  assert.equal(body.ok, true);
  assert.equal(body.modelId, "@cf/cloudflare/clef");
  assert.equal(env.AI.calls[0].model, "@cf/cloudflare/clef");
  assert.equal(typeof body.latencyMs, "number");
  assert.deepEqual(body.result.answers.spam.noul, 0.9);
});

test("classify rejects unknown models and missing input without calling the binding", async () => {
  const env = makeEnv();
  assert.equal((await classify(env, { model: "llama", input: {} })).status, 400);
  assert.equal((await classify(env, { model: "gemma" })).status, 400);
  assert.equal(env.AI.calls.length, 0);
});

test("classify maps binding errors to retryable/non-retryable statuses without leaking the stack", async () => {
  const transient = await classify(makeEnv({ fail: "AiError: 429 rate limited" }), { model: "gemma", input: { messages: [] } });
  assert.equal(transient.status, 503);
  assert.equal(transient.body.ok, false);
  assert.equal(transient.body.error.message, "AiError: 429 rate limited");
  const fatal = await classify(makeEnv({ fail: "bad request" }), { model: "gemma", input: { messages: [] } });
  assert.equal(fatal.status, 502);
});

test("fetch handler: routing, local access without a token, and health", async () => {
  const env = makeEnv();
  assert.equal((await worker.fetch(new Request("http://127.0.0.1:8787/classify"), env)).status, 405);
  assert.equal((await worker.fetch(new Request("http://127.0.0.1:8787/nope"), env)).status, 404);
  const health = await (await worker.fetch(new Request("http://127.0.0.1:8787/health"), env)).json();
  assert.deepEqual(health, { ok: true, binding: true });
  const ok = await worker.fetch(post("http://localhost:8787/classify", { model: "clef-flash", input: { state: "x" } }), env);
  assert.equal(ok.status, 200);
  const bad = await worker.fetch(new Request("http://localhost:8787/classify", { method: "POST", body: "{nope" }), env);
  assert.equal(bad.status, 400);
});

test("fetch handler: deployed Worker requires the BENCHMARK_TOKEN secret", async () => {
  const unconfigured = await worker.fetch(post("https://bench.example.workers.dev/classify", { model: "clef", input: {} }), makeEnv());
  assert.equal(unconfigured.status, 503);
  const env = makeEnv({ env: { BENCHMARK_TOKEN: "s3cret" } });
  const missing = await worker.fetch(post("https://bench.example.workers.dev/classify", { model: "clef", input: {} }), env);
  assert.equal(missing.status, 401);
  const wrong = await worker.fetch(post("https://bench.example.workers.dev/classify", { model: "clef", input: {} }, { authorization: "Bearer nope" }), env);
  assert.equal(wrong.status, 401);
  const local = await worker.fetch(post("http://localhost:8787/classify", { model: "clef", input: {} }), env);
  assert.equal(local.status, 401, "token is enforced even locally once configured");
  const right = await worker.fetch(post("https://bench.example.workers.dev/classify", { model: "clef", input: {} }, { authorization: "Bearer s3cret" }), env);
  assert.equal(right.status, 200);
  assert.equal(env.AI.calls.length, 1);
});
