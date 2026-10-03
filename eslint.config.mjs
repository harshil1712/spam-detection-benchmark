import js from "@eslint/js";
import globals from "globals";

export default [
  { ignores: ["node_modules/", ".cache/", ".wrangler/"] },
  js.configs.recommended,
  {
    files: ["scripts/**/*.mjs", "eslint.config.mjs"],
    languageOptions: { ecmaVersion: 2024, sourceType: "module", globals: globals.nodeBuiltin },
  },
  {
    files: ["src/**/*.mjs"],
    languageOptions: { ecmaVersion: 2024, sourceType: "module", globals: globals.serviceworker },
  },
  {
    files: ["**/*.test.mjs"],
    languageOptions: { globals: { ...globals.nodeBuiltin } },
  },
  { rules: { "no-unused-vars": ["error", { argsIgnorePattern: "^_" }] } },
];
