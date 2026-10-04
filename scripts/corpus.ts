// Download, verify, decode and deterministically sample the Apache SpamAssassin public corpus.

import { createHash } from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { mkdir, readdir, readFile, writeFile, access } from 'node:fs/promises';
import path from 'node:path';
import { simpleParser } from 'mailparser';
import type { EmailFields, Label, Sample } from '../src/lib.ts';

const BASE_URL = 'https://spamassassin.apache.org/old/publiccorpus/';
const CACHE_DIR = path.resolve('.cache');

const ARCHIVES: Array<{ group: string; label: Label; file: string; sha256: string }> = [
	{
		group: 'easy_ham',
		label: 'ham',
		file: '20030228_easy_ham.tar.bz2',
		sha256: '2b7b65904bcfcc31d2b5f51946f2d261370b257402cbbd62930b46ab83367438',
	},
	{
		group: 'hard_ham',
		label: 'ham',
		file: '20030228_hard_ham.tar.bz2',
		sha256: 'ce2ce67880643dbde65ea7f85bffbfe4417349c4bd80b6b0de56262ae6b0a9c9',
	},
	{
		group: 'spam',
		label: 'spam',
		file: '20030228_spam.tar.bz2',
		sha256: 'c08debc32413804949a866be45ef78195cec2cbafd1da744ed76cdb860589743',
	},
];

const sha256Hex = (data: Buffer | string) => createHash('sha256').update(data).digest('hex');

const exists = (file: string) =>
	access(file).then(
		() => true,
		() => false,
	);

async function verifyArchive(file: string, expected: string): Promise<void> {
	const actual = sha256Hex(await readFile(file));
	if (actual !== expected) throw new Error(`Checksum mismatch for ${path.basename(file)}: expected ${expected}, got ${actual}`);
}

/** Same seed + corpus => same sample, independent of directory order. */
function deterministicSample(files: string[], count: number, seed: string): string[] {
	return files
		.map((file) => ({ file, key: sha256Hex(`${seed}:${file}`) }))
		.sort((a, b) => a.key.localeCompare(b.key))
		.slice(0, count)
		.map((x) => x.file);
}

/** Download, verify and extract the archives (cached in .cache/), then pick `perGroup` messages from each. */
export async function sampleCorpus(perGroup: number, seed: string, log: (line: string) => void): Promise<Array<Sample & { path: string }>> {
	const samples: Array<Sample & { path: string }> = [];
	await mkdir(path.join(CACHE_DIR, 'extracted'), { recursive: true });
	for (const archive of ARCHIVES) {
		const archivePath = path.join(CACHE_DIR, archive.file);
		if (!(await exists(archivePath))) {
			log(`Downloading ${archive.file}`);
			const response = await fetch(BASE_URL + archive.file);
			if (!response.ok) throw new Error(`Download failed (${response.status}) for ${archive.file}`);
			await writeFile(archivePath, Buffer.from(await response.arrayBuffer()));
		}
		await verifyArchive(archivePath, archive.sha256);
		const dir = path.join(CACHE_DIR, 'extracted', archive.group);
		if (!(await exists(dir))) {
			log(`Extracting ${archive.file}`);
			await promisify(execFile)('tar', ['-xjf', archivePath, '-C', path.dirname(dir)]);
		}
		const files = (await readdir(dir)).filter((f) => /^\d{5}\./.test(f));
		for (const file of deterministicSample(files, perGroup, `${seed}:${archive.group}`)) {
			const filePath = path.join(dir, file);
			samples.push({
				id: `${archive.group}/${file}`,
				group: archive.group,
				label: archive.label,
				contentHash: sha256Hex(await readFile(filePath)),
				path: filePath,
			});
		}
	}
	return samples;
}

const htmlToText = (html: string) =>
	html
		.replace(/<(script|style)[\s\S]*?<\/\1>/gi, ' ')
		.replace(/<[^>]+>/g, ' ')
		.replace(/&nbsp;/gi, ' ')
		.replace(/&amp;/gi, '&')
		.replace(/&lt;/gi, '<')
		.replace(/&gt;/gi, '>')
		.replace(/&quot;/gi, '"');

/** Decode a raw RFC 822 message into the fields the benchmark uses. */
export async function decodeMessage(raw: Buffer | string): Promise<EmailFields> {
	const parsed = await simpleParser(raw, { skipImageLinks: true, skipTextToHtml: true });
	const precedence = parsed.headers.get('precedence');
	// mailparser folds List-* headers into one "list" object.
	const list = parsed.headers.get('list') as { unsubscribe?: unknown } | undefined;
	return {
		from: parsed.from?.text,
		subject: parsed.subject,
		listUnsubscribe: list?.unsubscribe != null,
		precedence: typeof precedence === 'string' ? precedence : undefined,
		body: parsed.text?.trim() ? parsed.text : parsed.html ? htmlToText(parsed.html) : '',
	};
}
