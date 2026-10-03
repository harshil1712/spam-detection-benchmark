// Download, verify, decode and deterministically sample the Apache
// SpamAssassin public corpus. Node-only (uses the filesystem and `tar`).

import { createHash } from "node:crypto";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { mkdir, readdir, readFile, writeFile, access } from "node:fs/promises";
import path from "node:path";
import { simpleParser } from "mailparser";
import { buildEmailState } from "./lib.mjs";

const execFileAsync = promisify(execFile);

export const CORPUS_BASE_URL = "https://spamassassin.apache.org/old/publiccorpus/";

export const CORPUS_ARCHIVES = Object.freeze([
  Object.freeze({
    name: "easy_ham",
    label: "ham",
    file: "20030228_easy_ham.tar.bz2",
    directory: "easy_ham",
    sha256: "2b7b65904bcfcc31d2b5f51946f2d261370b257402cbbd62930b46ab83367438",
  }),
  Object.freeze({
    name: "hard_ham",
    label: "ham",
    file: "20030228_hard_ham.tar.bz2",
    directory: "hard_ham",
    sha256: "ce2ce67880643dbde65ea7f85bffbfe4417349c4bd80b6b0de56262ae6b0a9c9",
  }),
  Object.freeze({
    name: "spam",
    label: "spam",
    file: "20030228_spam.tar.bz2",
    directory: "spam",
    sha256: "c08debc32413804949a866be45ef78195cec2cbafd1da744ed76cdb860589743",
  }),
]);

export const DEFAULT_CACHE_DIR = path.resolve(".cache");

export function sha256Hex(buffer) {
  return createHash("sha256").update(buffer).digest("hex");
}

async function exists(file) {
  try {
    await access(file);
    return true;
  } catch {
    return false;
  }
}

export async function verifyArchive(file, expectedSha256) {
  const actual = sha256Hex(await readFile(file));
  if (actual !== expectedSha256) {
    throw new Error(`Checksum mismatch for ${path.basename(file)}: expected ${expectedSha256}, got ${actual}`);
  }
  return actual;
}

async function download(url, destination) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Download failed (${response.status}) for ${url}`);
  await writeFile(destination, Buffer.from(await response.arrayBuffer()));
}

/**
 * Ensure every archive is downloaded, checksum-verified and extracted under
 * `cacheDir`. Returns per-group metadata including message counts.
 */
export async function prepareCorpus({ cacheDir = DEFAULT_CACHE_DIR, log = () => {} } = {}) {
  const archiveDir = path.join(cacheDir, "archives");
  const extractDir = path.join(cacheDir, "extracted");
  await mkdir(archiveDir, { recursive: true });
  await mkdir(extractDir, { recursive: true });
  const groups = [];
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
      await execFileAsync("tar", ["-xjf", archivePath, "-C", extractDir]);
    }
    const files = (await readdir(groupDir)).filter((f) => /^\d{5}\./.test(f)).sort();
    groups.push({ ...archive, dir: groupDir, files, count: files.length });
  }
  return { cacheDir, groups };
}

// Deterministic PRNG (mulberry32) seeded from a string, so the same seed and
// corpus always yield the same sample regardless of platform.
export function seededRandom(seed) {
  let a = 0;
  const digest = createHash("sha256").update(String(seed)).digest();
  for (let i = 0; i < 4; i++) a = (a << 8) | digest[i];
  return function next() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function deterministicSample(items, count, seed) {
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
export async function sampleCorpus(corpus, { perGroup, seed }) {
  const samples = [];
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

function htmlToText(html) {
  return String(html)
    .replace(/<(script|style)[\s\S]*?<\/\1>/gi, " ")
    .replace(/<br\s*\/?>|<\/p>|<\/div>|<\/tr>|<\/li>/gi, "\n")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'");
}

/** Decode a raw RFC 822 message into the fields the benchmark uses. */
export async function decodeMessage(raw) {
  const parsed = await simpleParser(raw, { skipImageLinks: true, skipTextToHtml: true });
  const header = (name) => {
    const value = parsed.headers.get(name);
    if (value == null) return "";
    if (typeof value === "string") return value;
    if (Array.isArray(value)) return value.map((v) => (typeof v === "string" ? v : v?.text ?? "")).join(", ");
    return value.text ?? value.value ?? String(value);
  };
  const body = parsed.text && parsed.text.trim() ? parsed.text : parsed.html ? htmlToText(parsed.html) : "";
  return {
    from: parsed.from?.text ?? header("from"),
    subject: parsed.subject ?? "",
    listUnsubscribe: parsed.headers.get("list")?.unsubscribe != null || parsed.headers.has("list-unsubscribe"),
    precedence: header("precedence"),
    body,
  };
}

export async function loadEmailState(sample) {
  const raw = await readFile(sample.path);
  return buildEmailState(await decodeMessage(raw));
}

/** Count distinct content hashes per group (duplicates are kept, not removed). */
export function uniqueContentCounts(samples) {
  const counts = {};
  for (const group of new Set(samples.map((s) => s.group))) {
    counts[group] = new Set(samples.filter((s) => s.group === group).map((s) => s.contentHash)).size;
  }
  return counts;
}
