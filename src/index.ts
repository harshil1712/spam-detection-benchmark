// Thin authenticated proxy so the benchmark CLI can reach Workers AI through the AI binding.
const MODELS: Record<string, string> = {
	gemma: '@cf/google/gemma-4-26b-a4b-it',
	clef: '@cf/cloudflare/clef',
	'clef-flash': '@cf/cloudflare/clef-flash',
};

export default {
	async fetch(request, env) {
		if (request.method !== 'POST') return new Response('POST { model, input }', { status: 405 });
		if (!env.BENCHMARK_TOKEN || request.headers.get('authorization') !== `Bearer ${env.BENCHMARK_TOKEN}`)
			return new Response('Unauthorized', { status: 401 });
		const { model, input } = await request.json<{ model: string; input: Record<string, unknown> }>();
		if (!MODELS[model]) return new Response(`Unknown model: ${model}`, { status: 400 });
		const started = Date.now();
		const result = await env.AI.run(MODELS[model], input);
		return Response.json({ result, latencyMs: Date.now() - started });
	},
} satisfies ExportedHandler<Env>;
