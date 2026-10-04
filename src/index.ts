// Runs a prepared Workers AI request through the `AI` binding and returns the
// result with the latency measured around `env.AI.run()`. Under `wrangler dev`
// (localhost) no token is needed; a deployed Worker refuses requests until the
// BENCHMARK_TOKEN secret is set and sent as a bearer token.

import { MODELS, isModelAlias } from './lib.ts';

const LOCAL_HOSTS = new Set(['localhost', '127.0.0.1', '[::1]']);
const json = (status: number, body: unknown) => Response.json(body, { status });

export default {
	async fetch(request, env) {
		const url = new URL(request.url);
		if (url.pathname !== '/classify' || request.method !== 'POST') return json(404, { error: 'POST /classify' });
		if (env.BENCHMARK_TOKEN) {
			if (request.headers.get('authorization') !== `Bearer ${env.BENCHMARK_TOKEN}`) return json(401, { error: 'Unauthorized' });
		} else if (!LOCAL_HOSTS.has(url.hostname)) {
			return json(503, { error: 'Set the BENCHMARK_TOKEN secret before using a deployed Worker' });
		}
		const { model, input } = (await request.json().catch(() => ({}))) as { model?: unknown; input?: unknown };
		if (!isModelAlias(model) || typeof input !== 'object' || input === null) return json(400, { error: 'Expected { model, input }' });
		// The generated AiModelList does not include the Clef models yet.
		const ai = env.AI as unknown as { run(model: string, input: object): Promise<unknown> };
		const started = Date.now();
		try {
			const result = await ai.run(MODELS[model], input);
			return json(200, { modelId: MODELS[model], latencyMs: Date.now() - started, result });
		} catch (error) {
			return json(502, { error: error instanceof Error ? error.message : String(error) });
		}
	},
} satisfies ExportedHandler<Env & { BENCHMARK_TOKEN?: string }>;
