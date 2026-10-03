import js from "@eslint/js";
import globals from "globals";

export default [
  { ignores: ["node_modules/", ".cache/", ".wrangler/"] },
  js.configs.recommended,
  {
    files: ["**/*.mjs"],
    languageOptions: { ecmaVersion: 2024, sourceType: "module", globals: { ...globals.nodeBuiltin, ...globals.serviceworker } },
    rules: { "no-unused-vars": ["error", { argsIgnorePattern: "^_" }] },
  },
];
