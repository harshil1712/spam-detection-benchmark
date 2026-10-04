import { env } from 'cloudflare:test';
import { describe, expect, it } from 'vitest';
import worker from '../src/index.ts';

type Env = Parameters<typeof worker.fetch>[1];

// The real AI binding bills inference, so every test stubs it.
function stubEnv({ fail, token }: { fail?: string; token?: string } = {}) {
	const calls: Array<{ model: string; input: unknown }> = [];
	const AI = {
		run: async (model: string, input: unknown) => {
			calls.push({ model, input });
			if (fail) throw new Error(fail);
			return { answers: { spam: { type: 'noul', noul: 0.9 } } };
		},
	};
	return { env: { ...env, AI, BENCHMARK_TOKEN: token } as unknown as Env, calls };
}

const post = (url: string, body: unknown, headers: Record<string, string> = {}) =>
	new Request<unknown, IncomingRequestCfProperties>(url, { method: 'POST', headers, body: JSON.stringify(body) });
const payload = { model: 'clef', input: { state: 'x' } };

describe('worker', () => {
	it('runs the binding with the resolved model ID and reports latency', async () => {
		const { env, calls } = stubEnv();
		const response = await worker.fetch(post('http://localhost:8787/classify', payload), env);
		const body = await response.json<{ modelId: string; latencyMs: number; result: unknown }>();
		expect(response.status).toBe(200);
		expect(body.modelId).toBe('@cf/cloudflare/clef');
		expect(typeof body.latencyMs).toBe('number');
		expect(body.result).toEqual({ answers: { spam: { type: 'noul', noul: 0.9 } } });
		expect(calls).toEqual([{ model: '@cf/cloudflare/clef', input: { state: 'x' } }]);
	});

	it('rejects bad routes and payloads without calling the binding', async () => {
		const { env, calls } = stubEnv();
		const status = async (request: Request<unknown, IncomingRequestCfProperties>) => (await worker.fetch(request, env)).status;
		expect(await status(new Request('http://localhost:8787/classify'))).toBe(404);
		expect(await status(new Request('http://localhost:8787/other', { method: 'POST' }))).toBe(404);
		expect(await status(post('http://localhost:8787/classify', { model: 'llama', input: {} }))).toBe(400);
		expect(await status(post('http://localhost:8787/classify', { model: 'gemma' }))).toBe(400);
		expect(await status(new Request('http://localhost:8787/classify', { method: 'POST', body: '{nope' }))).toBe(400);
		expect(calls).toHaveLength(0);
	});

	it('returns 502 when the binding throws', async () => {
		const response = await worker.fetch(post('http://localhost:8787/classify', payload), stubEnv({ fail: 'AiError: 429' }).env);
		expect(response.status).toBe(502);
		expect(await response.json()).toEqual({ error: 'AiError: 429' });
	});

	it('requires the BENCHMARK_TOKEN secret once deployed', async () => {
		const url = 'https://bench.example.workers.dev/classify';
		expect((await worker.fetch(post(url, payload), stubEnv().env)).status).toBe(503);
		const { env, calls } = stubEnv({ token: 's3cret' });
		expect((await worker.fetch(post(url, payload), env)).status).toBe(401);
		expect((await worker.fetch(post(url, payload, { authorization: 'Bearer nope' }), env)).status).toBe(401);
		expect((await worker.fetch(post('http://localhost:8787/classify', payload), env)).status).toBe(401);
		expect((await worker.fetch(post(url, payload, { authorization: 'Bearer s3cret' }), env)).status).toBe(200);
		expect(calls).toHaveLength(1);
	});
});
