import { mkdtemp, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { classifyWithRetry, mapWithConcurrency, parseCliOptions } from '../scripts/benchmark.ts';
import { decodeMessage, deterministicSample, sha256Hex, verifyArchive } from '../scripts/corpus.ts';

describe('corpus', () => {
	it('deterministicSample is stable for a seed and sensitive to it', () => {
		const items = Array.from({ length: 200 }, (_, i) => `m${String(i).padStart(5, '0')}`);
		const a = deterministicSample(items, 10, 'seed-a');
		expect(deterministicSample([...items].reverse(), 10, 'seed-a')).toEqual(a);
		expect(deterministicSample(items, 10, 'seed-b')).not.toEqual(a);
		expect(new Set(a).size).toBe(10);
		expect(deterministicSample(items, 999, 's')).toHaveLength(200);
	});

	it('verifyArchive rejects checksum mismatches', async () => {
		const dir = await mkdtemp(path.join(tmpdir(), 'bench-'));
		const file = path.join(dir, 'x.tar.bz2');
		await writeFile(file, 'not really an archive');
		await expect(verifyArchive(file, '0'.repeat(64))).rejects.toThrow(/Checksum mismatch/);
		const expected = sha256Hex('not really an archive');
		await expect(verifyArchive(file, expected)).resolves.toBe(expected);
	});

	it('decodeMessage decodes MIME headers/bodies and converts HTML-only mail', async () => {
		const quotedPrintable = [
			'From: =?UTF-8?B?SsO8cmdlbg==?= <j@example.org>',
			'Subject: =?UTF-8?Q?Caf=C3=A9_menu?=',
			'List-Unsubscribe: <mailto:leave@example.org>',
			'Precedence: bulk',
			'Content-Type: text/plain; charset=utf-8',
			'Content-Transfer-Encoding: quoted-printable',
			'',
			'Cr=C3=A8me br=C3=BBl=C3=A9e',
		].join('\r\n');
		const plain = await decodeMessage(quotedPrintable);
		expect(plain.from).toMatch(/Jürgen/);
		expect(plain.subject).toBe('Café menu');
		expect(plain.listUnsubscribe).toBe(true);
		expect(plain.precedence).toBe('bulk');
		expect(plain.body).toMatch(/Crème brûlée/);

		const htmlOnly = [
			'From: a@b.c',
			'Subject: html',
			'Content-Type: text/html',
			'',
			'<html><body><p>Hello <b>there</b></p><script>evil()</script></body></html>',
		].join('\r\n');
		const html = await decodeMessage(htmlOnly);
		expect(html.body).toMatch(/Hello\s+there/);
		expect(html.body).not.toMatch(/evil|<b>/);
		expect(html.listUnsubscribe).toBe(false);
	});
});

describe('classifyWithRetry', () => {
	type Scripted = 'network' | { status: number; body?: unknown };
	const scriptedFetch = (responses: Scripted[], calls: Scripted[]) =>
		(async () => {
			const next = responses.shift()!;
			calls.push(next);
			if (next === 'network') throw new TypeError('fetch failed');
			return new Response(JSON.stringify(next.body ?? { ok: next.status === 200, result: { r: 1 }, latencyMs: 7 }), {
				status: next.status,
			});
		}) as typeof fetch;
	const base = { workerUrl: 'http://w', workerToken: null, timeoutMs: 1000 };
	const job = { model: 'clef' as const, input: {} };

	it('retries retryable failures with backoff', async () => {
		const calls: Scripted[] = [];
		const sleeps: number[] = [];
		const ok = await classifyWithRetry({ ...base, fetchImpl: scriptedFetch(['network', { status: 503 }, { status: 200 }], calls) }, job, {
			sleepImpl: async (ms) => void sleeps.push(ms),
		});
		expect(ok).toEqual({ result: { r: 1 }, latencyMs: 7, attempts: 3 });
		expect(sleeps).toEqual([500, 1000]);
	});

	it('does not retry auth failures', async () => {
		const calls: Scripted[] = [];
		const auth = await classifyWithRetry({ ...base, fetchImpl: scriptedFetch([{ status: 401 }, { status: 200 }], calls) }, job, {
			sleepImpl: async () => {},
		});
		expect(calls).toHaveLength(1);
		expect(auth.error).toMatch(/BENCHMARK_WORKER_TOKEN/);
	});

	it('gives up after maxAttempts and reports the attempt count', async () => {
		const calls: Scripted[] = [];
		const responses: Scripted[] = [{ status: 500 }, { status: 500 }, { status: 500 }, { status: 500 }, { status: 200 }];
		const exhausted = await classifyWithRetry({ ...base, fetchImpl: scriptedFetch(responses, calls) }, job, { sleepImpl: async () => {} });
		expect(calls).toHaveLength(4);
		expect(exhausted.error).toMatch(/HTTP 500/);
		expect(exhausted.attempts).toBe(4);
	});
});

describe('mapWithConcurrency', () => {
	it('bounds in-flight work and preserves order', async () => {
		let inFlight = 0;
		let peak = 0;
		const out = await mapWithConcurrency([5, 1, 3, 2, 4], 2, async (n) => {
			inFlight++;
			peak = Math.max(peak, inFlight);
			await new Promise((r) => setTimeout(r, n));
			inFlight--;
			return n * 10;
		});
		expect(out).toEqual([50, 10, 30, 20, 40]);
		expect(peak).toBe(2);
	});
});

describe('parseCliOptions', () => {
	it('applies defaults and validates arguments', () => {
		const defaults = parseCliOptions([]);
		expect(defaults).toMatchObject({
			samplePerGroup: 100,
			concurrency: 6,
			models: ['gemma', 'clef', 'clef-flash'],
			workerUrl: 'http://127.0.0.1:8787',
		});
		const custom = parseCliOptions(['--sample-per-group', '30', '--models', 'clef,clef', '--worker-url', 'https://x.example/'], {
			BENCHMARK_WORKER_TOKEN: 't',
		});
		expect(custom).toMatchObject({ models: ['clef'], workerUrl: 'https://x.example', workerToken: 't' });
		expect(parseCliOptions(['--help'])).toEqual({ help: true });
		expect(() => parseCliOptions(['--sample-per-group', '0'])).toThrow(/between 1 and 500/);
		expect(() => parseCliOptions(['--sample-per-group', '501'])).toThrow(/between 1 and 500/);
		expect(() => parseCliOptions(['--concurrency', '21'])).toThrow(/between 1 and 20/);
		expect(() => parseCliOptions(['--models', 'llama'])).toThrow(/Unknown model alias/);
		expect(() => parseCliOptions(['--worker-url', 'not a url'])).toThrow(/Invalid worker URL/);
		expect(() => parseCliOptions(['--bogus'])).toThrow();
	});
});
