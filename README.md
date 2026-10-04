# Workers AI spam detection benchmark

A small, reproducible benchmark comparing three [Workers AI](https://developers.cloudflare.com/workers-ai/) models for email spam classification:

| Alias        | Model                           | Interface                                                                            |
| ------------ | ------------------------------- | ------------------------------------------------------------------------------------ |
| `gemma`      | `@cf/google/gemma-4-26b-a4b-it` | Generative chat; asked for a JSON verdict (`spam` / `ham` / `unsure`), temperature 0 |
| `clef`       | `@cf/cloudflare/clef`           | Typed decision model; one `noul` question returns P(spam)                            |
| `clef-flash` | `@cf/cloudflare/clef-flash`     | Same as Clef, smaller/faster model                                                   |

All inference goes through a Worker's **AI binding** (`env.AI.run()`), not the REST API. Results are committed as [`results.json`](results.json).

## How it works

Two files:

- `src/index.ts` — a 20-line Worker: `POST { model, input }` → `env.AI.run(modelId, input)` → `{ result, latencyMs }`. Run it locally with `npm run dev`; the AI binding does remote inference even in local dev, so nothing is deployed.
- `scripts/benchmark.ts` — a Node script that downloads the [SpamAssassin public corpus](https://spamassassin.apache.org/old/publiccorpus/) (`easy_ham`, `hard_ham`, `spam`; SHA-256 pinned), takes the first 100 messages of each group, decodes them with `mailparser` into a bounded text (From, Subject, List-Unsubscribe present, Precedence, first 500 chars of body), sends every email to each model via the Worker, and prints a table:

```
model       | ham flagged @0.5 | spam caught @0.5 | ham flagged @0.9 | spam caught @0.9 | unsure | errors | avg ms
```

"Ham flagged" (legitimate mail marked as spam) is the number to minimise. Gemma gives a categorical verdict, so its score is 0 or 1 and both thresholds read the same; Clef returns a probability. `results.json` also holds one row per (message, model) with the message's corpus filename and score — never the email text.

## Running it yourself

Requirements: Node.js 22.18+, `tar` with bzip2 support, and a Cloudflare account with Workers AI enabled (inference is billed to your account).

```sh
npm ci
npx wrangler login       # or export CLOUDFLARE_API_TOKEN / CLOUDFLARE_ACCOUNT_ID
npm run dev              # Worker on http://127.0.0.1:8787
npm run benchmark        # in another terminal; ~900 requests
```

Settings (messages per group, concurrency, prompts) are constants at the top of `scripts/benchmark.ts`. The Worker has no authentication — keep it on localhost; don't deploy it as-is.

## Limitations

- The SpamAssassin corpus is from 2002–2003 and may be in the models' training data; it says little about modern phishing or personal inbox preferences.
- 300 messages is a small sample and the ham/spam balance is artificial.
- Latency is wall-clock time around `env.AI.run()` during one run, not a controlled latency benchmark.
- This project does not pick a production threshold or change any mail delivery.

## License

MIT for the benchmark code. The SpamAssassin corpus is distributed by the Apache SpamAssassin project under its own terms and is downloaded at run time, never committed.
