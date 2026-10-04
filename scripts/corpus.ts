// Download, verify, decode and deterministically sample the Apache
// SpamAssassin public corpus. Node-only (uses the filesystem and `tar`).

import { createHash } from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { mkdir, readdir, readFile, writeFile, access } from 'node:fs/promises';
import path from 'node:path';
import { simpleParser } from 'mailparser';
import { buildEmailState, type EmailFields, type EmailState, type Label } from '../src/lib.ts';

const execFileAsync = promisify(execFile);

export const CORPUS_BASE_URL = 'https://spamassassin.apache.org/old/publiccorpus/';

export interface CorpusArchive {
	name: string;
	label: Label;
	file: string;
	directory: string;
	sha256: string;
}

export const CORPUS_ARCHIVES: CorpusArchive[] = [
	{
		name: 'easy_ham',
		label: 'ham',
		file: '20030228_easy_ham.tar.bz2',
		directory: 'easy_ham',
		sha256: '2b7b65904bcfcc31d2b5f51946f2d261370b257402cbbd62930b46ab83367438',
	},
	{
		name: 'hard_ham',
		label: 'ham',
		file: '20030228_hard_ham.tar.bz2',
		directory: 'hard_ham',
		sha256: 'ce2ce67880643dbde65ea7f85bffbfe4417349c4bd80b6b0de56262ae6b0a9c9',
	},
	{
		name: 'spam',
		label: 'spam',
		file: '20030228_spam.tar.bz2',
		directory: 'spam',
		sha256: 'c08debc32413804949a866be45ef78195cec2cbafd1da744ed76cdb860589743',
	},
];

export const DEFAULT_CACHE_DIR = path.resolve('.cache');

export interface CorpusGroup extends CorpusArchive {
	dir: string;
	files: string[];
	count: number;
}

export interface Corpus {
	cacheDir: string;
	groups: CorpusGroup[];
}

export interface Sample {
	id: string;
	group: string;
	label: Label;
	contentHash: string;
	path: string;
}

export function sha256Hex(data: Buffer | string): string {
	return createHash('sha256').update(data).digest('hex');
}

async function exists(file: string): Promise<boolean> {
	try {
		await access(file);
		return true;
	} catch {
		return false;
	}
}

export async function verifyArchive(file: string, expectedSha256: string): Promise<string> {
	const actual = sha256Hex(await readFile(file));
	if (actual !== expectedSha256) {
		throw new Error(`Checksum mismatch for ${path.basename(file)}: expected ${expectedSha256}, got ${actual}`);
	}
	return actual;
}

async function download(url: string, destination: string): Promise<void> {
	const response = await fetch(url);
	if (!response.ok) throw new Error(`Download failed (${response.status}) for ${url}`);
	await writeFile(destination, Buffer.from(await response.arrayBuffer()));
}

/**
 * Ensure every archive is downloaded, checksum-verified and extracted under
 * `cacheDir`. Returns per-group metadata including message counts.
 */
export async function prepareCorpus({ cacheDir = DEFAULT_CACHE_DIR, log = (_: string) => {} } = {}): Promise<Corpus> {
	const archiveDir = path.join(cacheDir, 'archives');
	const extractDir = path.join(cacheDir, 'extracted');
	await mkdir(archiveDir, { recursive: true });
	await mkdir(extractDir, { recursive: true });
	const groups: CorpusGroup[] = [];
	for (const archive of CORPUS_ARCHIVES) {
		const archivePath = path.join(archiveDir, archive.file);
		if (!(await exists(archivePath))) {
			log(`Downloading ${archive.file}`);
			await download(CORPUS_BASE_URL + archive.file, archivePath);
		}
		await verifyArchive(archivePath, archive.sha256);
		const groupDir = path.join(extractDir, archive.directory);
		if (!(await exists(groupDir))) {
			log(`Extracting ${archive.file}`);
			await execFileAsync('tar', ['-xjf', archivePath, '-C', extractDir]);
		}
		const files = (await readdir(groupDir)).filter((f) => /^\d{5}\./.test(f)).sort();
		groups.push({ ...archive, dir: groupDir, files, count: files.length });
	}
	return { cacheDir, groups };
}

// Deterministic PRNG (mulberry32) seeded from a string, so the same seed and
// corpus always yield the same sample regardless of platform.
export function seededRandom(seed: string): () => number {
	let a = 0;
	const digest = createHash('sha256').update(seed).digest();
	for (let i = 0; i < 4; i++) a = (a << 8) | digest[i];
	return function next() {
		a |= 0;
		a = (a + 0x6d2b79f5) | 0;
		let t = Math.imul(a ^ (a >>> 15), 1 | a);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

export function deterministicSample(items: string[], count: number, seed: string): string[] {
	const sorted = [...items].sort();
	const rand = seededRandom(seed);
	for (let i = sorted.length - 1; i > 0; i--) {
		const j = Math.floor(rand() * (i + 1));
		[sorted[i], sorted[j]] = [sorted[j], sorted[i]];
	}
	return sorted.slice(0, Math.min(count, sorted.length));
}

/**
 * Pick `perGroup` messages from each corpus group. Sample IDs are
 * `<group>/<corpus filename>`; the content hash covers the raw message bytes.
 */
export async function sampleCorpus(corpus: Corpus, { perGroup, seed }: { perGroup: number; seed: string }): Promise<Sample[]> {
	const samples: Sample[] = [];
	for (const group of corpus.groups) {
		const chosen = deterministicSample(group.files, perGroup, `${seed}:${group.name}`);
		for (const file of chosen) {
			const filePath = path.join(group.dir, file);
			const raw = await readFile(filePath);
			samples.push({ id: `${group.name}/${file}`, group: group.name, label: group.label, contentHash: sha256Hex(raw), path: filePath });
		}
	}
	return samples;
}

function htmlToText(html: string): string {
	return html
		.replace(/<(script|style)[\s\S]*?<\/\1>/gi, ' ')
		.replace(/<br\s*\/?>|<\/p>|<\/div>|<\/tr>|<\/li>/gi, '\n')
		.replace(/<[^>]+>/g, ' ')
		.replace(/&nbsp;/gi, ' ')
		.replace(/&amp;/gi, '&')
		.replace(/&lt;/gi, '<')
		.replace(/&gt;/gi, '>')
		.replace(/&quot;/gi, '"')
		.replace(/&#39;/gi, "'");
}

/** Decode a raw RFC 822 message into the fields the benchmark uses. */
export async function decodeMessage(raw: Buffer | string): Promise<EmailFields> {
	const parsed = await simpleParser(raw, { skipImageLinks: true, skipTextToHtml: true });
	const precedence = parsed.headers.get('precedence');
	// mailparser folds List-* headers into a single "list" object.
	const list = parsed.headers.get('list') as { unsubscribe?: unknown } | undefined;
	return {
		from: parsed.from?.text ?? '',
		subject: parsed.subject ?? '',
		listUnsubscribe: list?.unsubscribe != null,
		precedence: typeof precedence === 'string' ? precedence : Array.isArray(precedence) ? precedence.join(', ') : '',
		body: parsed.text?.trim() ? parsed.text : parsed.html ? htmlToText(parsed.html) : '',
	};
}

export async function loadEmailState(sample: Sample): Promise<EmailState> {
	const raw = await readFile(sample.path);
	return buildEmailState(await decodeMessage(raw));
}

/** Count distinct content hashes per group (duplicates are kept, not removed). */
export function uniqueContentCounts(samples: Sample[]): Record<string, number> {
	const counts: Record<string, number> = {};
	for (const group of new Set(samples.map((s) => s.group))) {
		counts[group] = new Set(samples.filter((s) => s.group === group).map((s) => s.contentHash)).size;
	}
	return counts;
}
