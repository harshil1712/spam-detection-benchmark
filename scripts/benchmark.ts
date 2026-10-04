// CLI: samples the corpus locally, sends every (sample, model) pair to the
// Worker's /classify endpoint (which calls the Workers AI binding) and writes
// report.json + report.md without any email content.

import { parseArgs } from 'node:util';
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
	MODELS,
	isModelAlias,
	buildModelInput,
	parseModelResult,
	renderEmail,
	createReport,
	renderMarkdown,
	type ModelAlias,
	type Prediction,
} from '../src/lib.ts';
import { sampleCorpus, decodeMessage } from './corpus.ts';

const RETRYABLE = new Set([0, 429, 500, 502, 503, 504]);

export interface Transport {
	workerUrl: string;
	token?: string;
	fetchImpl?: typeof fetch;
	sleep?: (ms: number) => Promise<void>;
}

/** POST one request to the Worker, retrying network errors, timeouts and 429/5xx with backoff. */
export async function classify(
	{ workerUrl, token, fetchImpl = fetch, sleep = (ms) => new Promise((r) => setTimeout(r, ms)) }: Transport,
	model: ModelAlias,
	input: object,
	maxAttempts = 4,
): Promise<{ result?: unknown; latencyMs?: number; error?: string; attempts: number }> {
	for (let attempt = 1; ; attempt++) {
		let status = 0;
		let error: string;
		try {
			const response = await fetchImpl(`${workerUrl}/classify`, {
				method: 'POST',
				headers: { 'content-type': 'application/json', ...(token ? { authorization: `Bearer ${token}` } : {}) },
				body: JSON.stringify({ model, input }),
				signal: AbortSignal.timeout(60_000),
			});
			status = response.status;
			const body = (await response.json().catch(() => ({}))) as { result?: unknown; latencyMs?: number; error?: string };
			if (response.ok) return { result: body.result, latencyMs: body.latencyMs, attempts: attempt };
			error = `HTTP ${status}${body.error ? `: ${body.error}` : ''}`;
		} catch (e) {
			error = e instanceof Error && e.name === 'TimeoutError' ? 'Timed out' : 'Network error';
		}
		if (!RETRYABLE.has(status) || attempt === maxAttempts) return { error, attempts: attempt };
		await sleep(500 * 2 ** (attempt - 1));
	}
}

/** Run `fn` over `items` with at most `limit` in flight, preserving order. */
export async function mapWithConcurrency<T, R>(items: T[], limit: number, fn: (item: T) => Promise<R>): Promise<R[]> {
	const results = new Array<R>(items.length);
	let next = 0;
	await Promise.all(
		Array.from({ length: Math.min(limit, items.length) }, async () => {
			while (next < items.length) {
				const i = next++;
				results[i] = await fn(items[i]);
			}
		}),
	);
	return results;
}

async function main() {
	const { values } = parseArgs({
		options: {
			'sample-per-group': { type: 'string', default: '100' },
			concurrency: { type: 'string', default: '6' },
			models: { type: 'string', default: Object.keys(MODELS).join(',') },
			seed: { type: 'string', default: 'spam-detection-benchmark-v1' },
			'worker-url': { type: 'string', default: process.env.BENCHMARK_WORKER_URL ?? 'http://127.0.0.1:8787' },
		},
	});
	const perGroup = Number(values['sample-per-group']);
	const concurrency = Number(values.concurrency);
	const models = values.models.split(',');
	if (!(perGroup >= 1 && perGroup <= 500) || !(concurrency >= 1 && concurrency <= 20))
		throw new Error('--sample-per-group must be 1-500 and --concurrency 1-20');
	if (!models.every(isModelAlias)) throw new Error(`--models must be a subset of ${Object.keys(MODELS).join(',')}`);
	const transport: Transport = { workerUrl: values['worker-url'].replace(/\/$/, ''), token: process.env.BENCHMARK_WORKER_TOKEN };

	const samples = await sampleCorpus(perGroup, values.seed, console.error);
	const jobs = [];
	for (const sample of samples) {
		const email = renderEmail(await decodeMessage(await readFile(sample.path)));
		for (const model of models) jobs.push({ sample, model, input: buildModelInput(model, email) });
	}
	console.error(`Sending ${jobs.length} requests to ${transport.workerUrl} (concurrency ${concurrency})`);

	let done = 0;
	const predictions = await mapWithConcurrency(jobs, concurrency, async ({ sample, model, input }): Promise<Prediction> => {
		const outcome = await classify(transport, model, input);
		if (++done % 50 === 0 || done === jobs.length) console.error(`  ${done}/${jobs.length}`);
		const parsed = outcome.error ? { score: null, error: outcome.error } : parseModelResult(model, outcome.result);
		return { sampleId: sample.id, model, label: sample.label, latencyMs: outcome.latencyMs, attempts: outcome.attempts, ...parsed };
	});

	const report = createReport({ seed: values.seed, samplePerGroup: perGroup, samples, predictions });
	await writeFile('report.json', JSON.stringify(report, null, 2) + '\n');
	await writeFile('report.md', renderMarkdown(report));
	console.error('Wrote report.json and report.md');
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
	main().catch((error: unknown) => {
		console.error(error instanceof Error ? error.message : String(error));
		process.exit(1);
	});
}
