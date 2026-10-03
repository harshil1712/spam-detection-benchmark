#!/usr/bin/env node
// CLI: prepares the corpus locally, sends every (sample, model) pair to the
// Worker's /classify endpoint (which calls the Workers AI binding), and writes
// a privacy-preserving JSON + Markdown report.

import { parseArgs } from "node:util";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  MODEL_ALIASES,
  MODELS,
  buildModelInput,
  parseModelResult,
  createReport,
  renderMarkdown,
  exceedsErrorGate,
  MAX_ERROR_RATE,
} from "./lib.mjs";
import { CORPUS_ARCHIVES, CORPUS_BASE_URL, prepareCorpus, sampleCorpus, loadEmailState, uniqueContentCounts, sha256Hex } from "./corpus.mjs";

export const DEFAULTS = Object.freeze({
  samplePerGroup: 100,
  concurrency: 6,
  seed: "spam-detection-benchmark-v1",
  workerUrl: "http://127.0.0.1:8787",
  timeoutMs: 60_000,
  maxAttempts: 4,
  output: "report.json",
  summary: "report.md",
});

const RETRYABLE_STATUS = new Set([429, 500, 502, 503, 504]);

export function parseCliOptions(argv, env = {}) {
  const { values } = parseArgs({
    args: argv,
    options: {
      "sample-per-group": { type: "string" },
      concurrency: { type: "string" },
      models: { type: "string" },
      seed: { type: "string" },
      "worker-url": { type: "string" },
      output: { type: "string" },
      summary: { type: "string" },
      timeout: { type: "string" },
      help: { type: "boolean", short: "h" },
    },
    strict: true,
  });
  if (values.help) return { help: true };
  const integer = (raw, fallback, name, min, max) => {
    if (raw === undefined) return fallback;
    const n = Number(raw);
    if (!Number.isInteger(n) || n < min || n > max) throw new Error(`--${name} must be an integer between ${min} and ${max}`);
    return n;
  };
  const models = (values.models ?? MODEL_ALIASES.join(",")).split(",").map((m) => m.trim()).filter(Boolean);
  for (const m of models) if (!MODELS[m]) throw new Error(`Unknown model alias "${m}". Known: ${MODEL_ALIASES.join(", ")}`);
  if (models.length === 0) throw new Error("--models must list at least one model");
  const workerUrl = values["worker-url"] ?? env.BENCHMARK_WORKER_URL ?? DEFAULTS.workerUrl;
  try {
    new URL(workerUrl);
  } catch {
    throw new Error(`Invalid worker URL: ${workerUrl}`);
  }
  return {
    samplePerGroup: integer(values["sample-per-group"], DEFAULTS.samplePerGroup, "sample-per-group", 1, 500),
    concurrency: integer(values.concurrency, DEFAULTS.concurrency, "concurrency", 1, 20),
    timeoutMs: integer(values.timeout, DEFAULTS.timeoutMs, "timeout", 1_000, 600_000),
    models: [...new Set(models)],
    seed: values.seed ?? DEFAULTS.seed,
    workerUrl: workerUrl.replace(/\/$/, ""),
    workerToken: env.BENCHMARK_WORKER_TOKEN ?? null,
    output: values.output ?? DEFAULTS.output,
    summary: values.summary ?? DEFAULTS.summary,
  };
}

export const HELP = `Usage: node scripts/benchmark.mjs [options]

Options:
  --sample-per-group <n>   Messages per corpus group, 1-500 (default ${DEFAULTS.samplePerGroup})
  --concurrency <n>        In-flight requests, 1-20 (default ${DEFAULTS.concurrency})
  --models <list>          Comma-separated aliases (default ${MODEL_ALIASES.join(",")})
  --seed <string>          Sampling seed (default ${DEFAULTS.seed})
  --worker-url <url>       Worker base URL (default $BENCHMARK_WORKER_URL or ${DEFAULTS.workerUrl})
  --timeout <ms>           Per-attempt timeout (default ${DEFAULTS.timeoutMs})
  --output <path>          JSON report path (default ${DEFAULTS.output})
  --summary <path>         Markdown report path (default ${DEFAULTS.summary})

Environment:
  BENCHMARK_WORKER_URL     Worker base URL
  BENCHMARK_WORKER_TOKEN   Bearer token for a deployed Worker (not needed for wrangler dev)
`;

export function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Run `fn` over `items` with at most `limit` promises in flight, preserving order. */
export async function mapWithConcurrency(items, limit, fn) {
  const results = new Array(items.length);
  let next = 0;
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (next < items.length) {
      const index = next++;
      results[index] = await fn(items[index], index);
    }
  });
  await Promise.all(workers);
  return results;
}

export class ClassifyError extends Error {
  constructor(message, { status = null, retryable = false } = {}) {
    super(message);
    this.name = "ClassifyError";
    this.status = status;
    this.retryable = retryable;
  }
}

async function classifyOnce({ workerUrl, workerToken, timeoutMs, fetchImpl }, job) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const headers = { "content-type": "application/json" };
    if (workerToken) headers.authorization = `Bearer ${workerToken}`;
    let response;
    try {
      response = await fetchImpl(`${workerUrl}/classify`, {
        method: "POST",
        headers,
        body: JSON.stringify({ model: job.model, input: job.input }),
        signal: controller.signal,
      });
    } catch (error) {
      const aborted = error?.name === "AbortError";
      throw new ClassifyError(aborted ? `Timed out after ${timeoutMs} ms` : "Network error", { retryable: true });
    }
    let body = null;
    try {
      body = await response.json();
    } catch {
      body = null;
    }
    if (response.status === 401 || response.status === 403) {
      throw new ClassifyError("Worker rejected the request (check BENCHMARK_WORKER_TOKEN)", { status: response.status });
    }
    if (!response.ok || !body?.ok) {
      throw new ClassifyError(`Worker returned HTTP ${response.status}`, {
        status: response.status,
        retryable: RETRYABLE_STATUS.has(response.status),
      });
    }
    return body;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Classify one job with retries. Returns {result, latencyMs, attempts} or
 * {error, attempts}. Latency is the Worker-measured time around env.AI.run()
 * for the successful attempt.
 */
export async function classifyWithRetry(options, job, { maxAttempts = DEFAULTS.maxAttempts, backoffMs = 500, sleepImpl = sleep } = {}) {
  let lastError;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      const body = await classifyOnce(options, job);
      return { result: body.result, latencyMs: body.latencyMs, attempts: attempt };
    } catch (error) {
      lastError = error;
      if (!(error instanceof ClassifyError) || !error.retryable || attempt === maxAttempts) break;
      await sleepImpl(backoffMs * 2 ** (attempt - 1));
    }
  }
  const message = lastError instanceof ClassifyError ? lastError.message : "Unexpected client error";
  return { error: message, attempts: Math.min(maxAttempts, lastError?.attempts ?? maxAttempts) };
}

export async function scriptHashes(rootDir) {
  const files = ["scripts/lib.mjs", "scripts/corpus.mjs", "scripts/benchmark.mjs", "src/worker.mjs", "package-lock.json"];
  const entries = await Promise.all(
    files.map(async (file) => [file, sha256Hex(await readFile(path.join(rootDir, file)))]),
  );
  return Object.fromEntries(entries);
}

async function fetchWorkerHealth(options) {
  const response = await fetch(`${options.workerUrl}/health`);
  const body = await response.json().catch(() => null);
  if (!response.ok || !body?.ok) throw new Error(`Worker health check failed at ${options.workerUrl}/health (HTTP ${response.status})`);
  if (!body.binding) throw new Error("Worker is reachable but has no AI binding; check wrangler.jsonc");
}

export async function runBenchmark(options, { log = console.error, fetchImpl = fetch } = {}) {
  const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const startedAt = new Date();
  await fetchWorkerHealth(options);
  const corpus = await prepareCorpus({ log });
  const samples = await sampleCorpus(corpus, { perGroup: options.samplePerGroup, seed: options.seed });
  log(`Prepared ${samples.length} samples (${options.samplePerGroup} per group)`);
  const states = new Map();
  for (const sample of samples) states.set(sample.id, await loadEmailState(sample));

  const jobs = [];
  for (const model of options.models) {
    for (const sample of samples) jobs.push({ sampleId: sample.id, label: sample.label, model, input: buildModelInput(model, states.get(sample.id)) });
  }
  log(`Sending ${jobs.length} requests to ${options.workerUrl} with concurrency ${options.concurrency}`);
  let done = 0;
  const predictions = await mapWithConcurrency(jobs, options.concurrency, async (job) => {
    const outcome = await classifyWithRetry({ ...options, fetchImpl }, job);
    done++;
    if (done % 50 === 0 || done === jobs.length) log(`  ${done}/${jobs.length}`);
    const base = { sampleId: job.sampleId, model: job.model, label: job.label, attempts: outcome.attempts };
    if (outcome.error) return { ...base, score: null, error: outcome.error };
    const parsed = parseModelResult(job.model, outcome.result);
    if (parsed.error) return { ...base, score: null, error: parsed.error, latencyMs: outcome.latencyMs };
    return { ...base, score: parsed.score, verdict: parsed.verdict, latencyMs: outcome.latencyMs };
  });

  const report = createReport({
    predictions,
    samples,
    corpus: {
      source: CORPUS_BASE_URL,
      archives: CORPUS_ARCHIVES.map(({ name, label, file, sha256 }) => ({ name, label, file, sha256 })),
      groups: corpus.groups.map((g) => ({ name: g.name, label: g.label, available: g.count, sampled: samples.filter((s) => s.group === g.name).length })),
      samplePerGroup: options.samplePerGroup,
      seed: options.seed,
      uniqueContentCounts: uniqueContentCounts(samples),
    },
    transport: {
      name: "workers-ai-binding",
      detail: "Node CLI -> Worker /classify -> env.AI.run(); latency measured inside the Worker",
      concurrency: options.concurrency,
      timeoutMs: options.timeoutMs,
      maxAttempts: DEFAULTS.maxAttempts,
      retryOn: ["network error", "timeout", ...RETRYABLE_STATUS].map(String),
    },
    hashes: await scriptHashes(rootDir),
    run: {
      generatedAt: startedAt.toISOString(),
      durationMs: Date.now() - startedAt.getTime(),
      models: options.models,
      node: process.version,
    },
  });

  await mkdir(path.dirname(path.resolve(options.output)), { recursive: true });
  await mkdir(path.dirname(path.resolve(options.summary)), { recursive: true });
  await writeFile(options.output, JSON.stringify(report, null, 2) + "\n");
  await writeFile(options.summary, renderMarkdown(report));
  log(`Wrote ${options.output} and ${options.summary}`);
  const failed = exceedsErrorGate(report.metrics);
  if (failed.length) log(`Error-rate gate (> ${MAX_ERROR_RATE * 100}%) failed for: ${failed.join(", ")}`);
  return { report, failed };
}

async function main() {
  let options;
  try {
    options = parseCliOptions(process.argv.slice(2), process.env);
  } catch (error) {
    console.error(error.message);
    console.error(HELP);
    process.exit(2);
  }
  if (options.help) {
    console.log(HELP);
    return;
  }
  const { failed } = await runBenchmark(options);
  process.exit(failed.length ? 1 : 0);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    console.error(error.message);
    process.exit(1);
  });
}
