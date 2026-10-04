import { cloudflareTest } from '@cloudflare/vitest-plugin';
import { defineConfig } from 'vitest/config';

// The Worker is tested inside workerd; the Node CLI and corpus code run in Node.
export default defineConfig({
	test: {
		projects: [
			{
				plugins: [
					cloudflareTest({
						wrangler: { configPath: './wrangler.jsonc' },
						// Tests stub env.AI; never open a remote proxy session (would need Cloudflare credentials and bill inference).
						remoteBindings: false,
					}),
				],
				test: { name: 'worker', include: ['test/worker.spec.ts'] },
			},
			{
				test: { name: 'node', include: ['test/lib.spec.ts', 'test/benchmark.spec.ts'] },
			},
		],
	},
});
