# Workers AI spam detection benchmark

An independent, reproducible benchmark comparing three [Workers AI](https://developers.cloudflare.com/workers-ai/) models for email spam classification:

| Alias | Model | Interface |
|---|---|---|
| `gemma` | `@cf/google/gemma-4-26b-a4b-it` | Generative chat; asked for a JSON verdict (`spam` / `ham` / `unsure`), temperature 0, thinking disabled, 64 completion tokens |
| `clef` | `@cf/cloudflare/clef` | Typed decision model; one `noul` question returns P(spam) |
| `clef-flash` | `@cf/cloudflare/clef-flash` | Same as Clef, smaller/faster model |

All inference goes through a Worker's **AI binding** (`env.AI.run()`), not the REST API. The latest results live in [`results/latest.md`](results/latest.md) and [`results/latest.json`](results/latest.json).

## How it works

```
scripts/benchmark.mjs ──POST /classify──▶ src/worker.mjs ──env.AI.run()──▶ Workers AI
   (Node: corpus, sampling,                 (Worker: AI binding,
    metrics, reports)                        latency measurement)
```

1. **Corpus** (`scripts/corpus.mjs`): downloads three archives of the [Apache SpamAssassin public corpus](https://spamassassin.apache.org/old/publiccorpus/) (`easy_ham`, `hard_ham`, `spam`), verifies pinned SHA-256 checksums, extracts them with `tar`, and samples N messages per group with a seeded, deterministic shuffle. Labels come from the corpus groups.
2. **Email state** (`src/lib.mjs`): MIME is decoded with `mailparser`; HTML-only mail is converted to text. Every model receives the same bounded fields — From (≤320 chars), Subject (≤500), whether `List-Unsubscribe` is present, Precedence (≤100) and a whitespace-normalised body excerpt (≤500). Prompts tell the model the email is untrusted content.
3. **Inference** (`src/worker.mjs`): the Worker exposes `POST /classify` with `{ model, input }`, calls `env.AI.run(modelId, input)` and returns the raw result plus the latency measured around the binding call. The CLI runs with bounded concurrency (default 6), a 60 s per-attempt timeout and up to four attempts with exponential backoff on network errors and HTTP 429/5xx. Authentication failures are not retried.
4. **Metrics**: at thresholds 0.5, 0.75, 0.9, 0.95 and 0.99 the report lists the ham false-positive rate (primary metric), spam recall and flagged precision. Errors and Gemma's explicit `unsure` never flag mail but stay in the denominators. When zero ham false positives are observed, a one-sided 95% upper bound `1 - 0.05^(1/n)` is included. A run exits non-zero if any model's error rate exceeds 5% (an operational gate, not a quality gate).
5. **Reports**: `report.json` (metrics, per-sample predictions keyed by opaque IDs and content hashes, request templates, script and corpus hashes, transport) and `report.md`. No email content, headers, credentials or account IDs are written.

## Running it yourself

Requirements: Node.js 22.13+ (or 24+), `tar` with bzip2 support, and a Cloudflare account with Workers AI enabled. Inference is billed to your account.

```sh
npm ci --ignore-scripts
npx wrangler login            # or export CLOUDFLARE_API_TOKEN / CLOUDFLARE_ACCOUNT_ID
```

### Option A: local Worker via `wrangler dev` (simplest)

The AI binding always runs inference remotely, even in local dev, so no deployment is needed.

```sh
npm run dev                   # Worker on http://127.0.0.1:8787
npm run benchmark -- --sample-per-group 100 --concurrency 6
```

### Option B: deployed Worker

```sh
npx wrangler secret put BENCHMARK_TOKEN      # any long random string
npm run deploy
BENCHMARK_WORKER_URL=https://spam-detection-benchmark.<subdomain>.workers.dev \
BENCHMARK_WORKER_TOKEN=<same string> \
npm run benchmark
```

A deployed Worker refuses `/classify` until `BENCHMARK_TOKEN` is set, so a public URL cannot run inference on your account.

### CLI options

```
--sample-per-group <n>   1-500 messages per group (default 100 → 300 messages, 900 requests)
--concurrency <n>        1-20 in-flight requests (default 6)
--models <list>          default gemma,clef,clef-flash
--seed <string>          sampling seed (default spam-detection-benchmark-v1)
--worker-url <url>       default $BENCHMARK_WORKER_URL or http://127.0.0.1:8787
--output / --summary     report paths (default report.json / report.md)
```

To refresh the published results, copy `report.json` and `report.md` to `results/latest.json` and `results/latest.md`.

## Development

```sh
npm run lint     # eslint
npm run check    # node --check on every module
npm test         # credential-free regression tests (node:test)
```

CI runs the same checks plus a `wrangler deploy --dry-run` on pushes and pull requests; it never runs paid inference.

## Limitations

- The SpamAssassin corpus dates from 2002–2003 and does not reflect modern phishing or personal inbox preferences. Training-data contamination cannot be ruled out.
- The sample is small and the ham/spam balance is artificial. Messages are not deduplicated; the report lists unique-content counts per group.
- Thresholds are inspected on the same sample they are reported on, not on a held-out set.
- Latency is wall-clock time around `env.AI.run()` inside the Worker during one run; it is not a controlled latency benchmark.
- Gemma and Clef receive the same email state but necessarily different prompts (generative vs. typed decision). Full request templates are in the JSON report.
- This project does not choose a production threshold or change any mail delivery.

## License

MIT for the benchmark code. The SpamAssassin corpus is distributed by the Apache SpamAssassin project under its own terms and is downloaded at run time, never committed.
