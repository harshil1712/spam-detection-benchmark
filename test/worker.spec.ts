import { env } from 'cloudflare:test';
import { describe, expect, it } from 'vitest';
import worker, { classify, type BenchEnv } from '../src/index.ts';

// The real AI binding would bill inference, so every test swaps in a stub.
function stubEnv({ fail, token }: { fail?: string; token?: string } = {}) {
	const calls: Array<{ model: string; input: unknown }> = [];
	const ai = {
		run(model: string, input: unknown) {
			calls.push({ model, input });
			if (fail) return Promise.reject(new Error(fail));
			return Promise.resolve({ answers: { spam: { type: 'noul', noul: 0.9 } } });
		},
	};
	const stub = { ...env, AI: ai as unknown as Ai, BENCHMARK_TOKEN: token } as BenchEnv;
	return { env: stub, calls };
}

const IncomingRequest = Request<unknown, IncomingRequestCfProperties>;

const post = (url: string, body: unknown, headers: Record<string, string> = {}) =>
	new IncomingRequest(url, { method: 'POST', headers: { 'content-type': 'application/json', ...headers }, body: JSON.stringify(body) });

const send = (request: Request<unknown, IncomingRequestCfProperties>, benchEnv: BenchEnv) => worker.fetch(request, benchEnv);

describe('classify', () => {
	it('runs the binding with the resolved model ID and measures latency', async () => {
		const { env, calls } = stubEnv();
		const [status, body] = await classify(env, { model: 'clef', input: { model: 'clef', state: 's', questions: {} } });
		expect(status).toBe(200);
		expect(body.ok).toBe(true);
		expect(body.modelId).toBe('@cf/cloudflare/clef');
		expect(calls[0].model).toBe('@cf/cloudflare/clef');
		if (body.ok) {
			expect(typeof body.latencyMs).toBe('number');
			expect(body.result).toEqual({ answers: { spam: { type: 'noul', noul: 0.9 } } });
		}
	});

	it('rejects unknown models and missing input without calling the binding', async () => {
		const { env, calls } = stubEnv();
		expect((await classify(env, { model: 'llama', input: {} }))[0]).toBe(400);
		expect((await classify(env, { model: 'gemma' }))[0]).toBe(400);
		expect((await classify(env, undefined))[0]).toBe(400);
		expect(calls).toHaveLength(0);
	});

	it('reports binding errors as 502 with the message only', async () => {
		const { env } = stubEnv({ fail: 'AiError: 429 rate limited' });
		const [status, body] = await classify(env, { model: 'gemma', input: { messages: [] } });
		expect(status).toBe(502);
		expect(body).toEqual({ ok: false, modelId: '@cf/google/gemma-4-26b-a4b-it', error: 'AiError: 429 rate limited' });
	});
});

describe('fetch handler', () => {
	it('routes, allows local access without a token and reports health', async () => {
		const { env } = stubEnv();
		expect((await send(new IncomingRequest('http://127.0.0.1:8787/classify'), env)).status).toBe(405);
		expect((await send(new IncomingRequest('http://127.0.0.1:8787/nope'), env)).status).toBe(404);
		expect(await (await send(new IncomingRequest('http://127.0.0.1:8787/health'), env)).json()).toEqual({ ok: true, binding: true });
		expect((await send(post('http://localhost:8787/classify', { model: 'clef-flash', input: { state: 'x' } }), env)).status).toBe(200);
		expect((await send(new IncomingRequest('http://localhost:8787/classify', { method: 'POST', body: '{nope' }), env)).status).toBe(400);
	});

	it('requires the BENCHMARK_TOKEN secret once deployed', async () => {
		const url = 'https://bench.example.workers.dev/classify';
		const payload = { model: 'clef', input: {} };
		expect((await send(post(url, payload), stubEnv().env)).status).toBe(503);
		const { env, calls } = stubEnv({ token: 's3cret' });
		expect((await send(post(url, payload), env)).status).toBe(401);
		expect((await send(post(url, payload, { authorization: 'Bearer nope' }), env)).status).toBe(401);
		expect(
			(await send(post('http://localhost:8787/classify', payload), env)).status,
			'token is enforced even locally once configured',
		).toBe(401);
		expect((await send(post(url, payload, { authorization: 'Bearer s3cret' }), env)).status).toBe(200);
		expect(calls).toHaveLength(1);
	});
});
