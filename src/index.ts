// Thin proxy so the benchmark CLI can reach Workers AI through the AI binding.
// Run with `npm run dev`; it is meant for localhost only.
const MODELS: Record<string, string> = {
	gemma: '@cf/google/gemma-4-26b-a4b-it',
	clef: '@cf/cloudflare/clef',
	'clef-flash': '@cf/cloudflare/clef-flash',
};

export default {
	async fetch(request, env) {
		if (request.method !== 'POST') return new Response('POST { model, input }', { status: 405 });
		const { model, input } = await request.json<{ model: string; input: object }>();
		if (!MODELS[model]) return new Response(`Unknown model: ${model}`, { status: 400 });
		// Cast: the generated AiModelList does not include the Clef models yet.
		const ai = env.AI as unknown as { run(model: string, input: object): Promise<unknown> };
		const started = Date.now();
		const result = await ai.run(MODELS[model], input);
		return Response.json({ result, latencyMs: Date.now() - started });
	},
} satisfies ExportedHandler<Env>;
