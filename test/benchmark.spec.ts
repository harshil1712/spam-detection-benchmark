import { mkdtemp, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { classify, mapWithConcurrency } from '../scripts/benchmark.ts';
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
		const file = path.join(await mkdtemp(path.join(tmpdir(), 'bench-')), 'x.tar.bz2');
		await writeFile(file, 'not really an archive');
		await expect(verifyArchive(file, '0'.repeat(64))).rejects.toThrow(/Checksum mismatch/);
		await expect(verifyArchive(file, sha256Hex('not really an archive'))).resolves.toBeUndefined();
	});

	it('decodeMessage decodes MIME and converts HTML-only mail', async () => {
		const plain = await decodeMessage(
			[
				'From: =?UTF-8?B?SsO8cmdlbg==?= <j@example.org>',
				'Subject: =?UTF-8?Q?Caf=C3=A9_menu?=',
				'List-Unsubscribe: <mailto:leave@example.org>',
				'Precedence: bulk',
				'Content-Type: text/plain; charset=utf-8',
				'Content-Transfer-Encoding: quoted-printable',
				'',
				'Cr=C3=A8me br=C3=BBl=C3=A9e',
			].join('\r\n'),
		);
		expect(plain).toMatchObject({ subject: 'Café menu', listUnsubscribe: true, precedence: 'bulk' });
		expect(plain.from).toMatch(/Jürgen/);
		expect(plain.body).toMatch(/Crème brûlée/);

		const html = await decodeMessage(
			['From: a@b.c', 'Subject: html', 'Content-Type: text/html', '', '<p>Hello <b>there</b></p><script>evil()</script>'].join('\r\n'),
		);
		expect(html.body).toMatch(/Hello\s+there/);
		expect(html.body).not.toMatch(/evil|<b>/);
		expect(html.listUnsubscribe).toBe(false);
	});
});

describe('classify', () => {
	type Step = 'network' | number;
	const scripted = (steps: Step[]) => {
		const calls: Step[] = [];
		const fetchImpl = (async () => {
			const step = steps.shift()!;
			calls.push(step);
			if (step === 'network') throw new TypeError('fetch failed');
			return new Response(JSON.stringify(step === 200 ? { result: { r: 1 }, latencyMs: 7 } : { error: 'nope' }), { status: step });
		}) as typeof fetch;
		return { calls, fetchImpl };
	};
	const sleeps: number[] = [];
	const transport = (steps: Step[]) => ({ workerUrl: 'http://w', ...scripted(steps), sleep: async (ms: number) => void sleeps.push(ms) });

	it('retries network errors and 5xx with backoff', async () => {
		const t = transport(['network', 503, 200]);
		expect(await classify(t, 'clef', {})).toEqual({ result: { r: 1 }, latencyMs: 7, attempts: 3 });
		expect(sleeps).toEqual([500, 1000]);
		expect(t.calls).toHaveLength(3);
	});

	it('does not retry auth failures', async () => {
		const t = transport([401, 200]);
		expect(await classify(t, 'clef', {})).toEqual({ error: 'HTTP 401: nope', attempts: 1 });
		expect(t.calls).toHaveLength(1);
	});

	it('gives up after maxAttempts', async () => {
		const t = transport([500, 500, 500, 500, 200]);
		expect(await classify(t, 'clef', {})).toEqual({ error: 'HTTP 500: nope', attempts: 4 });
		expect(t.calls).toHaveLength(4);
	});
});

describe('mapWithConcurrency', () => {
	it('bounds in-flight work and preserves order', async () => {
		let inFlight = 0;
		let peak = 0;
		const out = await mapWithConcurrency([5, 1, 3, 2, 4], 2, async (n) => {
			peak = Math.max(peak, ++inFlight);
			await new Promise((r) => setTimeout(r, n));
			inFlight--;
			return n * 10;
		});
		expect(out).toEqual([50, 10, 30, 20, 40]);
		expect(peak).toBe(2);
	});
});
