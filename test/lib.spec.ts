import { describe, expect, it } from 'vitest';
import {
	FIELD_LIMITS,
	MODELS,
	buildEmailState,
	buildModelInput,
	computeMetrics,
	createReport,
	exceedsErrorGate,
	parseModelResult,
	renderMarkdown,
	zeroEventUpperBound,
	type Prediction,
} from '../src/lib.ts';

describe('email state', () => {
	it('bounds and normalises every field', () => {
		const state = buildEmailState({
			from: 'a'.repeat(1000),
			subject: '  hello\n\nworld  ',
			listUnsubscribe: true,
			precedence: 'b'.repeat(1000),
			body: 'x '.repeat(2000),
		});
		expect(state.from).toHaveLength(FIELD_LIMITS.from);
		expect(state.subject).toBe('hello world');
		expect(state.listUnsubscribe).toBe('yes');
		expect(state.precedence).toHaveLength(FIELD_LIMITS.precedence);
		expect(state.body).toHaveLength(FIELD_LIMITS.body);
		expect(buildEmailState({}).listUnsubscribe).toBe('no');
	});
});

describe('buildModelInput', () => {
	it('produces the documented request shapes', () => {
		const state = buildEmailState({ from: 'x@y.z', subject: 's', body: 'b' });
		const gemma = buildModelInput('gemma', state);
		expect(gemma).toMatchObject({ temperature: 0, max_completion_tokens: 64, chat_template_kwargs: { enable_thinking: false } });
		expect(gemma.messages).toHaveLength(2);
		const clef = buildModelInput('clef-flash', state);
		expect(clef.model).toBe('clef-flash');
		expect(clef.questions).toMatchObject({ spam: { type: 'noul' } });
		expect(clef.state).toMatch(/From: x@y.z/);
		expect(() => buildModelInput('nope', state)).toThrow(/Unknown model/);
	});
});

describe('parseModelResult', () => {
	it('handles Gemma verdicts, unsure, reasoning and malformed output', () => {
		const ok = (content: string) => parseModelResult('gemma', { choices: [{ message: { content, reasoning_content: 'spam spam spam' } }] });
		expect(ok('{"verdict":"spam"}')).toEqual({ score: 1, verdict: 'spam' });
		expect(ok('```json\n{"verdict": "Ham"}\n```')).toEqual({ score: 0, verdict: 'ham' });
		expect(ok('unsure')).toEqual({ score: null, verdict: 'unsure' });
		expect(parseModelResult('gemma', { response: 'ham' })).toEqual({ score: 0, verdict: 'ham' });
		expect(ok('').error).toMatch(/Empty/);
		expect(ok('I think it is probably fine').error).toMatch(/Unrecognised/);
		expect(parseModelResult('gemma', { choices: [{ message: { reasoning_content: 'spam' } }] }).error).toMatch(/No assistant content/);
	});

	it('validates Clef noul scores', () => {
		const clef = (noul: unknown) => parseModelResult('clef', { answers: { spam: { type: 'noul', noul } } });
		expect(clef(0.42)).toEqual({ score: 0.42, verdict: null });
		expect(clef(undefined).error).toMatch(/Missing/);
		expect(clef(null).error).toMatch(/Missing/);
		expect(clef('0.9').error).toMatch(/Missing/);
		expect(clef(1.5).error).toMatch(/outside/);
		expect(parseModelResult('clef', {}).error).toMatch(/Missing/);
	});
});

describe('computeMetrics', () => {
	it('keeps errors and unsure in denominators and reports the zero-event bound', () => {
		const predictions: Prediction[] = [
			{ model: 'clef', label: 'ham', score: 0.1, latencyMs: 100 },
			{ model: 'clef', label: 'ham', score: 0.96, latencyMs: 300 },
			{ model: 'clef', label: 'ham', score: null, error: 'boom' },
			{ model: 'clef', label: 'spam', score: 0.99, latencyMs: 200 },
			{ model: 'clef', label: 'spam', score: null, error: 'boom' },
			{ model: 'gemma', label: 'ham', score: null, verdict: 'unsure', latencyMs: 50 },
			{ model: 'gemma', label: 'spam', score: 1, verdict: 'spam', latencyMs: 150 },
		];
		const metrics = computeMetrics(predictions, [0.5, 0.99]);
		const clef = metrics.clef;
		expect(clef.errors).toBe(2);
		expect(clef.errorRate).toBe(0.4);
		expect(clef.meanLatencyMs).toBe(200);
		expect(clef.thresholds.map((t) => [t.hamFlagged, t.hamTotal, t.spamFlagged, t.spamTotal, t.flaggedPrecision])).toEqual([
			[1, 3, 1, 2, 0.5],
			[0, 3, 1, 2, 1],
		]);
		expect(clef.thresholds[0].hamFalsePositiveUpperBound95).toBeNull();
		expect(clef.thresholds[1].hamFalsePositiveUpperBound95).toBeCloseTo(zeroEventUpperBound(3)!, 12);
		expect(zeroEventUpperBound(60)).toBeCloseTo(0.0487, 3);
		const gemma = metrics.gemma;
		expect(gemma.unscored).toBe(1);
		expect(gemma.errors).toBe(0);
		expect(gemma.thresholds[0].hamFlagged).toBe(0);
		expect(gemma.thresholds[0].spamRecall).toBe(1);
		expect(exceedsErrorGate(metrics)).toEqual(['clef']);
		const single = computeMetrics([{ model: 'clef', label: 'ham', score: 0.1, latencyMs: 1 }], [0.5]);
		expect(single.clef.thresholds[0].flaggedPrecision).toBeNull();
		expect(single.clef.thresholds[0].spamRecall).toBeNull();
	});
});

describe('report', () => {
	it('omits email content and secrets', () => {
		const samples = [
			{ id: 'spam/00001.x', group: 'spam', label: 'spam', contentHash: 'h1', path: '/secret/path' },
			{ id: 'easy_ham/00002.y', group: 'easy_ham', label: 'ham', contentHash: 'h2', path: '/secret/path2' },
		] as const;
		const predictions: Prediction[] = [
			{ sampleId: samples[0].id, model: 'clef', label: 'spam', score: 0.97, latencyMs: 10, attempts: 1 },
			{ sampleId: samples[1].id, model: 'clef', label: 'ham', score: 0.03, latencyMs: 12, attempts: 1 },
		];
		const report = createReport({
			predictions,
			samples: [...samples],
			corpus: {
				source: 'https://example.org/',
				archives: [],
				groups: [{ name: 'spam' }, { name: 'easy_ham' }],
				samplePerGroup: 1,
				seed: 's',
				uniqueContentCounts: {},
			},
			transport: { name: 'workers-ai-binding', detail: 'test' },
			hashes: { 'src/lib.ts': 'abc' },
			run: { generatedAt: '2026-01-01T00:00:00.000Z', models: ['clef'], node: 'v24' },
		});
		const json = JSON.stringify(report);
		for (const forbidden of ['/secret/path', 'VIAGRA', 'accountId', 'account_id', 'Bearer', 'Authorization'])
			expect(json).not.toContain(forbidden);
		expect(Object.keys(report.models)).toEqual(['clef']);
		expect(report.requestTemplates.clef.modelId).toBe(MODELS.clef.id);
		expect(report.predictions).toHaveLength(2);
		expect(report.samples[0]).not.toHaveProperty('path');
		const md = renderMarkdown(report);
		expect(md).toMatch(/\| clef \| 0\.95 \| 0\/1 \(0\.00%\) — 95% upper bound 95\.00% \| 1\/1 \(100\.00%\) \| 100\.00% \|/);
		expect(md).toMatch(/Transport: workers-ai-binding/);
	});
});
