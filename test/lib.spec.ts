import { describe, expect, it } from 'vitest';
import {
	buildModelInput,
	computeMetrics,
	createReport,
	parseModelResult,
	renderEmail,
	renderMarkdown,
	type Prediction,
} from '../src/lib.ts';

describe('renderEmail', () => {
	it('bounds and normalises every field', () => {
		const text = renderEmail({ from: 'a'.repeat(1000), subject: '  hello\n\nworld  ', listUnsubscribe: true, body: 'x '.repeat(2000) });
		expect(text).toContain(`From: ${'a'.repeat(320)}\n`);
		expect(text).toContain('Subject: hello world\n');
		expect(text).toContain('List-Unsubscribe header present: yes');
		expect(text).toContain('Precedence: (none)');
		expect(text.split('Body (truncated):\n')[1]).toHaveLength(500);
	});
});

describe('buildModelInput', () => {
	it('produces the documented request shapes', () => {
		const gemma = buildModelInput('gemma', 'EMAIL');
		expect(gemma).toMatchObject({ temperature: 0, max_completion_tokens: 64, chat_template_kwargs: { enable_thinking: false } });
		expect(JSON.stringify(gemma.messages)).toContain('EMAIL');
		expect(buildModelInput('clef-flash', 'EMAIL')).toMatchObject({
			model: 'clef-flash',
			state: 'EMAIL',
			questions: { spam: { type: 'noul' } },
		});
	});
});

describe('parseModelResult', () => {
	const gemma = (content: unknown) => parseModelResult('gemma', { choices: [{ message: { content } }] });
	const clef = (noul: unknown) => parseModelResult('clef', { answers: { spam: { type: 'noul', noul } } });

	it('handles Gemma verdicts, unsure and malformed output', () => {
		expect(gemma('{"verdict":"spam"}')).toEqual({ score: 1, verdict: 'spam' });
		expect(gemma('```json\n{"verdict": "Ham"}\n```')).toEqual({ score: 0, verdict: 'ham' });
		expect(gemma('unsure')).toEqual({ score: null, verdict: 'unsure' });
		expect(gemma('').error).toMatch(/No assistant content/);
		expect(gemma(undefined).error).toMatch(/No assistant content/);
		expect(gemma('probably fine').error).toMatch(/Unrecognised/);
	});

	it('validates Clef noul scores', () => {
		expect(clef(0.42)).toEqual({ score: 0.42 });
		for (const bad of [undefined, null, '0.9', 1.5, -0.1]) expect(clef(bad).error).toMatch(/noul/);
		expect(parseModelResult('clef', null).error).toMatch(/noul/);
	});
});

const p = (model: Prediction['model'], label: Prediction['label'], rest: Partial<Prediction>): Prediction => ({
	sampleId: `${label}/${Math.random()}`,
	model,
	label,
	score: null,
	attempts: 1,
	...rest,
});

describe('computeMetrics', () => {
	it('keeps errors and unsure in denominators and reports the zero-event bound', () => {
		const metrics = computeMetrics([
			p('clef', 'ham', { score: 0.1, latencyMs: 100 }),
			p('clef', 'ham', { score: 0.96, latencyMs: 300 }),
			p('clef', 'ham', { error: 'boom' }),
			p('clef', 'spam', { score: 0.99, latencyMs: 200 }),
			p('clef', 'spam', { error: 'boom' }),
			p('gemma', 'ham', { verdict: 'unsure', latencyMs: 50 }),
			p('gemma', 'spam', { score: 1, verdict: 'spam', latencyMs: 150 }),
		]);
		expect(metrics.clef).toMatchObject({ requests: 5, errors: 2, unsure: 0, meanLatencyMs: 200 });
		const at = (t: number) => metrics.clef.thresholds.find((x) => x.threshold === t)!;
		expect(at(0.5)).toMatchObject({
			hamFlagged: 1,
			hamTotal: 3,
			spamFlagged: 1,
			spamTotal: 2,
			flaggedPrecision: 0.5,
			hamFalsePositiveUpperBound95: null,
		});
		expect(at(0.99)).toMatchObject({ hamFlagged: 0, spamFlagged: 1, flaggedPrecision: 1 });
		expect(at(0.99).hamFalsePositiveUpperBound95).toBeCloseTo(1 - 0.05 ** (1 / 3), 12);
		expect(metrics.gemma).toMatchObject({ requests: 2, errors: 0, unsure: 1 });
		expect(metrics.gemma.thresholds[0]).toMatchObject({ hamFlagged: 0, spamFlagged: 1 });
	});
});

describe('report', () => {
	it('contains metrics and opaque sample IDs but no email content or paths', () => {
		const samples = [
			{ id: 'spam/00001.x', group: 'spam', label: 'spam' as const, contentHash: 'h1', path: '/secret/path' },
			{ id: 'easy_ham/00002.y', group: 'easy_ham', label: 'ham' as const, contentHash: 'h2', path: '/secret/path2' },
		];
		const report = createReport({
			seed: 's',
			samplePerGroup: 1,
			samples,
			predictions: [
				p('clef', 'spam', { sampleId: samples[0].id, score: 0.97, latencyMs: 10 }),
				p('clef', 'ham', { sampleId: samples[1].id, score: 0.03, latencyMs: 12 }),
			],
		});
		expect(JSON.stringify(report)).not.toContain('/secret');
		expect(report.models).toEqual({ clef: '@cf/cloudflare/clef' });
		expect(report.samples[0]).toEqual({ id: 'spam/00001.x', group: 'spam', label: 'spam', contentHash: 'h1' });
		expect(renderMarkdown(report)).toContain('| clef | 0.95 | 0/1 0.00% (95% upper bound 95.00%) | 1/1 100.00% | 100.00% |');
	});
});
