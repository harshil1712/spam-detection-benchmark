import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import {
  buildEmailState,
  buildModelInput,
  parseModelResult,
  computeMetrics,
  createReport,
  renderMarkdown,
  zeroEventUpperBound,
  exceedsErrorGate,
  FIELD_LIMITS,
  MODELS,
} from "./lib.mjs";
import { deterministicSample, verifyArchive, decodeMessage, sha256Hex } from "./corpus.mjs";
import { parseCliOptions, mapWithConcurrency, classifyWithRetry } from "./benchmark.mjs";

test("buildEmailState bounds and normalises every field", () => {
  const state = buildEmailState({
    from: "a".repeat(1000),
    subject: "  hello\n\nworld  ",
    listUnsubscribe: true,
    precedence: "b".repeat(1000),
    body: "x ".repeat(2000),
  });
  assert.equal(state.from.length, FIELD_LIMITS.from);
  assert.equal(state.subject, "hello world");
  assert.equal(state.listUnsubscribe, "yes");
  assert.equal(state.precedence.length, FIELD_LIMITS.precedence);
  assert.equal(state.body.length, FIELD_LIMITS.body);
  assert.equal(buildEmailState({}).listUnsubscribe, "no");
});

test("buildModelInput produces the documented request shapes", () => {
  const state = buildEmailState({ from: "x@y.z", subject: "s", body: "b" });
  const gemma = buildModelInput("gemma", state);
  assert.equal(gemma.temperature, 0);
  assert.equal(gemma.max_completion_tokens, 64);
  assert.equal(gemma.chat_template_kwargs.enable_thinking, false);
  assert.equal(gemma.messages.length, 2);
  const clef = buildModelInput("clef-flash", state);
  assert.equal(clef.model, "clef-flash");
  assert.equal(clef.questions.spam.type, "noul");
  assert.match(clef.state, /From: x@y.z/);
  assert.throws(() => buildModelInput("nope", state), /Unknown model/);
});

test("parseModelResult handles Gemma verdicts, unsure, reasoning and malformed output", () => {
  const ok = (content) => parseModelResult("gemma", { choices: [{ message: { content, reasoning_content: "spam spam spam" } }] });
  assert.deepEqual(ok('{"verdict":"spam"}'), { score: 1, verdict: "spam" });
  assert.deepEqual(ok('```json\n{"verdict": "Ham"}\n```'), { score: 0, verdict: "ham" });
  assert.deepEqual(ok("unsure"), { score: null, verdict: "unsure" });
  assert.deepEqual(parseModelResult("gemma", { response: "ham" }), { score: 0, verdict: "ham" });
  assert.match(ok("").error, /Empty/);
  assert.match(ok("I think it is probably fine").error, /Unrecognised/);
  assert.match(parseModelResult("gemma", { choices: [{ message: { reasoning_content: "spam" } }] }).error, /No assistant content/);
});

test("parseModelResult validates Clef noul scores", () => {
  const clef = (noul) => parseModelResult("clef", { answers: { spam: { type: "noul", noul } } });
  assert.deepEqual(clef(0.42), { score: 0.42, verdict: null });
  assert.match(clef(undefined).error, /Missing/);
  assert.match(clef(null).error, /Missing/);
  assert.match(clef("0.9").error, /Missing/);
  assert.match(clef(1.5).error, /outside/);
  assert.match(parseModelResult("clef", {}).error, /Missing/);
});

test("deterministicSample is stable for a seed and sensitive to it", () => {
  const items = Array.from({ length: 200 }, (_, i) => `m${String(i).padStart(5, "0")}`);
  const a = deterministicSample(items, 10, "seed-a");
  const b = deterministicSample([...items].reverse(), 10, "seed-a");
  const c = deterministicSample(items, 10, "seed-b");
  assert.deepEqual(a, b);
  assert.notDeepEqual(a, c);
  assert.equal(new Set(a).size, 10);
  assert.equal(deterministicSample(items, 999, "s").length, 200);
});

test("verifyArchive rejects checksum mismatches", async () => {
  const dir = await mkdtemp(path.join(tmpdir(), "bench-"));
  const file = path.join(dir, "x.tar.bz2");
  await writeFile(file, "not really an archive");
  await assert.rejects(verifyArchive(file, "0".repeat(64)), /Checksum mismatch/);
  assert.equal(await verifyArchive(file, sha256Hex(Buffer.from("not really an archive"))), sha256Hex(Buffer.from("not really an archive")));
});

test("decodeMessage decodes MIME headers/bodies and converts HTML-only mail", async () => {
  const quotedPrintable = [
    "From: =?UTF-8?B?SsO8cmdlbg==?= <j@example.org>",
    "Subject: =?UTF-8?Q?Caf=C3=A9_menu?=",
    "List-Unsubscribe: <mailto:leave@example.org>",
    "Precedence: bulk",
    "Content-Type: text/plain; charset=utf-8",
    "Content-Transfer-Encoding: quoted-printable",
    "",
    "Cr=C3=A8me br=C3=BBl=C3=A9e",
  ].join("\r\n");
  const plain = await decodeMessage(quotedPrintable);
  assert.match(plain.from, /Jürgen/);
  assert.equal(plain.subject, "Café menu");
  assert.equal(plain.listUnsubscribe, true);
  assert.equal(plain.precedence, "bulk");
  assert.match(plain.body, /Crème brûlée/);

  const htmlOnly = ["From: a@b.c", "Subject: html", "Content-Type: text/html", "", "<html><body><p>Hello <b>there</b></p><script>evil()</script></body></html>"].join("\r\n");
  const html = await decodeMessage(htmlOnly);
  assert.match(html.body, /Hello\s+there/);
  assert.doesNotMatch(html.body, /evil|<b>/);
  assert.equal(html.listUnsubscribe, false);
});

test("computeMetrics keeps errors and unsure in denominators and reports the zero-event bound", () => {
  const predictions = [
    { model: "clef", label: "ham", score: 0.1, latencyMs: 100 },
    { model: "clef", label: "ham", score: 0.96, latencyMs: 300 },
    { model: "clef", label: "ham", score: null, error: "boom" },
    { model: "clef", label: "spam", score: 0.99, latencyMs: 200 },
    { model: "clef", label: "spam", score: null, error: "boom" },
    { model: "gemma", label: "ham", score: null, verdict: "unsure", latencyMs: 50 },
    { model: "gemma", label: "spam", score: 1, verdict: "spam", latencyMs: 150 },
  ];
  const metrics = computeMetrics(predictions, [0.5, 0.99]);
  const clef = metrics.clef;
  assert.equal(clef.errors, 2);
  assert.equal(clef.errorRate, 0.4);
  assert.equal(clef.meanLatencyMs, 200);
  assert.deepEqual(
    clef.thresholds.map((t) => [t.hamFlagged, t.hamTotal, t.spamFlagged, t.spamTotal, t.flaggedPrecision]),
    [
      [1, 3, 1, 2, 0.5],
      [0, 3, 1, 2, 1],
    ],
  );
  assert.equal(clef.thresholds[0].hamFalsePositiveUpperBound95, null);
  assert.ok(Math.abs(clef.thresholds[1].hamFalsePositiveUpperBound95 - zeroEventUpperBound(3)) < 1e-12);
  assert.ok(Math.abs(zeroEventUpperBound(60) - 0.0487) < 0.0005);
  const gemma = metrics.gemma;
  assert.equal(gemma.unscored, 1);
  assert.equal(gemma.errors, 0);
  assert.equal(gemma.thresholds[0].hamFlagged, 0);
  assert.equal(gemma.thresholds[0].spamRecall, 1);
  assert.deepEqual(exceedsErrorGate(metrics), ["clef"]);
  const empty = computeMetrics([{ model: "clef", label: "ham", score: 0.1, latencyMs: 1 }], [0.5]);
  assert.equal(empty.clef.thresholds[0].flaggedPrecision, null);
  assert.equal(empty.clef.thresholds[0].spamRecall, null);
});

test("classifyWithRetry retries retryable failures and not auth failures", async () => {
  const calls = [];
  const sleeps = [];
  const mkFetch = (responses) => async () => {
    const next = responses.shift();
    calls.push(next);
    if (next === "network") throw new TypeError("fetch failed");
    return new Response(JSON.stringify(next.body ?? { ok: next.status === 200, result: { r: 1 }, latencyMs: 7 }), { status: next.status });
  };
  const base = { workerUrl: "http://w", workerToken: null, timeoutMs: 1000 };
  const job = { model: "clef", input: {} };
  const ok = await classifyWithRetry({ ...base, fetchImpl: mkFetch(["network", { status: 503 }, { status: 200 }]) }, job, { sleepImpl: async (ms) => sleeps.push(ms) });
  assert.deepEqual(ok, { result: { r: 1 }, latencyMs: 7, attempts: 3 });
  assert.deepEqual(sleeps, [500, 1000]);

  calls.length = 0;
  const auth = await classifyWithRetry({ ...base, fetchImpl: mkFetch([{ status: 401 }, { status: 200 }]) }, job, { sleepImpl: async () => {} });
  assert.equal(calls.length, 1);
  assert.match(auth.error, /BENCHMARK_WORKER_TOKEN/);

  calls.length = 0;
  const exhausted = await classifyWithRetry({ ...base, fetchImpl: mkFetch([{ status: 500 }, { status: 500 }, { status: 500 }, { status: 500 }, { status: 200 }]) }, job, { sleepImpl: async () => {} });
  assert.equal(calls.length, 4);
  assert.match(exhausted.error, /HTTP 500/);
  assert.equal(exhausted.attempts, 4);
});

test("mapWithConcurrency bounds in-flight work and preserves order", async () => {
  let inFlight = 0;
  let peak = 0;
  const out = await mapWithConcurrency([5, 1, 3, 2, 4], 2, async (n) => {
    inFlight++;
    peak = Math.max(peak, inFlight);
    await new Promise((r) => setTimeout(r, n));
    inFlight--;
    return n * 10;
  });
  assert.deepEqual(out, [50, 10, 30, 20, 40]);
  assert.equal(peak, 2);
});

test("parseCliOptions validates arguments", () => {
  const defaults = parseCliOptions([]);
  assert.equal(defaults.samplePerGroup, 100);
  assert.equal(defaults.concurrency, 6);
  assert.deepEqual(defaults.models, ["gemma", "clef", "clef-flash"]);
  assert.equal(defaults.workerUrl, "http://127.0.0.1:8787");
  const custom = parseCliOptions(["--sample-per-group", "30", "--models", "clef,clef", "--worker-url", "https://x.example/"], {
    BENCHMARK_WORKER_TOKEN: "t",
  });
  assert.deepEqual(custom.models, ["clef"]);
  assert.equal(custom.workerUrl, "https://x.example");
  assert.equal(custom.workerToken, "t");
  assert.throws(() => parseCliOptions(["--sample-per-group", "0"]), /between 1 and 500/);
  assert.throws(() => parseCliOptions(["--sample-per-group", "501"]), /between 1 and 500/);
  assert.throws(() => parseCliOptions(["--concurrency", "21"]), /between 1 and 20/);
  assert.throws(() => parseCliOptions(["--models", "llama"]), /Unknown model alias/);
  assert.throws(() => parseCliOptions(["--worker-url", "not a url"]), /Invalid worker URL/);
  assert.throws(() => parseCliOptions(["--bogus"]));
});

test("createReport and renderMarkdown omit email content and secrets", () => {
  const samples = [
    { id: "spam/00001.x", group: "spam", label: "spam", contentHash: "h1", path: "/secret/path" },
    { id: "easy_ham/00002.y", group: "easy_ham", label: "ham", contentHash: "h2", path: "/secret/path2" },
  ];
  const predictions = [
    { sampleId: samples[0].id, model: "clef", label: "spam", score: 0.97, latencyMs: 10, attempts: 1 },
    { sampleId: samples[1].id, model: "clef", label: "ham", score: 0.03, latencyMs: 12, attempts: 1 },
  ];
  const report = createReport({
    predictions,
    samples,
    corpus: { source: "https://example.org/", archives: [], groups: [{ name: "spam" }, { name: "easy_ham" }], samplePerGroup: 1, seed: "s", uniqueContentCounts: {} },
    transport: { name: "workers-ai-binding", detail: "test" },
    hashes: { "scripts/lib.mjs": "abc" },
    run: { generatedAt: "2026-01-01T00:00:00.000Z", models: ["clef"], node: "v24" },
  });
  const json = JSON.stringify(report);
  for (const forbidden of ["/secret/path", "VIAGRA", "accountId", "account_id", "Bearer", "Authorization"]) assert.ok(!json.includes(forbidden), forbidden);
  assert.deepEqual(Object.keys(report.models), ["clef"]);
  assert.equal(report.requestTemplates.clef.modelId, MODELS.clef.id);
  assert.equal(report.predictions.length, 2);
  assert.equal(report.samples[0].path, undefined);
  const md = renderMarkdown(report);
  assert.match(md, /\| clef \| 0\.95 \| 0\/1 \(0\.00%\) — 95% upper bound 95\.00% \| 1\/1 \(100\.00%\) \| 100\.00% \|/);
  assert.match(md, /Transport: workers-ai-binding/);
});
