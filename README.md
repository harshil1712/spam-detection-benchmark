# Workers AI spam detection benchmark

A small, reproducible benchmark comparing three [Workers AI](https://developers.cloudflare.com/workers-ai/) models for email spam classification:

| Alias        | Model                           | Interface                                                                            |
| ------------ | ------------------------------- | ------------------------------------------------------------------------------------ |
| `gemma`      | `@cf/google/gemma-4-26b-a4b-it` | Generative chat; asked for a JSON verdict (`spam` / `ham` / `unsure`), temperature 0 |
| `clef`       | `@cf/cloudflare/clef`           | Typed decision model; one `noul` question returns P(spam)                            |
| `clef-flash` | `@cf/cloudflare/clef-flash`     | Same as Clef, smaller/faster model                                                   |

All inference goes through a Worker's **AI binding** (`env.AI.run()`), not the REST API. The full run is saved in [`results.json`](results.json).

## How it works

Two files:

- `src/index.ts` — a thin Worker: authenticated `POST { model, input }` → `env.AI.run(modelId, input)` → `{ result, latencyMs }`. Run it locally with `npm run dev` or deploy it; the AI binding does remote inference in either case.
- `scripts/benchmark.ts` — a Node script that downloads the [SpamAssassin public corpus](https://spamassassin.apache.org/old/publiccorpus/) (`easy_ham`, `hard_ham`, `spam`; SHA-256 pinned), takes the first 100 messages of each group, decodes them with `mailparser` into a bounded text (From, Subject, List-Unsubscribe present, Precedence, first 500 chars of body), sends every email to each model via the Worker, and prints a table:

```
model       | ham flagged @0.5 | spam caught @0.5 | ham flagged @0.9 | spam caught @0.9 | unsure | errors | avg ms
```

"Ham flagged" (legitimate mail marked as spam) is the number to minimise. Gemma gives a categorical verdict, so its score is 0 or 1 and both thresholds read the same; Clef returns a probability. Explicit Gemma `unsure` verdicts count as unsure; malformed outputs and failed requests count as errors, not as valid classifications. The denominators include all messages, so check the unsure/error columns before comparing rates. `results.json` also holds one row per (message, model) with the message's corpus filename and score — never the email text.

## Results (2026-10-05)

100 messages from each corpus group (200 ham, 100 spam), with no errors or unsure verdicts:

| Model      | Ham flagged @0.5 | Spam caught @0.5 | Ham flagged @0.9 | Spam caught @0.9 | Avg inference ms |
| ---------- | ---------------: | ---------------: | ---------------: | ---------------: | ---------------: |
| Gemma      |           70/200 |          100/100 |           70/200 |          100/100 |              587 |
| Clef       |           34/200 |          100/100 |            5/200 |           99/100 |              383 |
| Clef-Flash |           29/200 |          100/100 |            9/200 |          100/100 |              152 |

These are results on this historical corpus, not a recommendation for a production spam threshold. See `results.json` for per-message scores.

## Running it yourself

Requirements: Node.js 22.18+, `tar` with bzip2 support, `openssl`, and a Cloudflare account with Workers AI enabled (inference is billed to your account).

```sh
npm ci
printf 'BENCHMARK_TOKEN=%s\n' "$(openssl rand -hex 32)" > .dev.vars
chmod 600 .dev.vars    # ignored by git; keep this token private
export BENCHMARK_TOKEN="$(cut -d= -f2 .dev.vars)"
npx wrangler login      # if not already logged in
export CLOUDFLARE_ACCOUNT_ID=<your-account-id>  # choose the account to bill
npm run dev             # local Worker on http://127.0.0.1:8787
PER_GROUP=1 npm run benchmark  # in another terminal, export BENCHMARK_TOKEN there too
```

To deploy the authenticated Worker, run `npx wrangler deploy --secrets-file .dev.vars` with the same account selected. Pass its URL to the CLI with `WORKER_URL=https://<worker>.<subdomain>.workers.dev npm run benchmark`. The secret is uploaded with the deployment; requests without it cannot invoke the AI binding. To remove the Worker later, use `npx wrangler delete --name spam-detection-benchmark` with the same account selected.

A full run uses ~900 billed inferences. `PER_GROUP=1` makes 9 inferences and also writes `results.json`, so rerun without the override for full results. Concurrency and prompts are constants at the top of `scripts/benchmark.ts`.

## Limitations

- The SpamAssassin corpus is from 2002–2003 and may be in the models' training data; it says little about modern phishing or personal inbox preferences.
- 300 messages is a small sample and the ham/spam balance is artificial.
- Latency is wall-clock time around `env.AI.run()` during one run, not a controlled latency benchmark.
- This project does not pick a production threshold or change any mail delivery.

## License

MIT for the benchmark code. The SpamAssassin corpus is distributed by the Apache SpamAssassin project under its own terms and is downloaded at run time, never committed.
