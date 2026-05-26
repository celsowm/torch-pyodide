#!/usr/bin/env node
// Generates python/torch/_version.py from python/pyproject.toml so that the
// Python package and the bundled JS runtime always expose the same version.
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const pyprojectPath = resolve(root, "python/pyproject.toml");
const pythonVersionPath = resolve(root, "python/torch/_version.py");
const runtimeVersionPath = resolve(root, "runtime/src/version.ts");

const pyproject = readFileSync(pyprojectPath, "utf8");
const match = pyproject.match(/^\s*version\s*=\s*"([^"]+)"/m);
if (!match) {
  console.error(`[sync_torch_version] could not find version in ${pyprojectPath}`);
  process.exit(1);
}
const version = match[1];

const pythonContent =
  "# Auto-generated from python/pyproject.toml by scripts/sync_torch_version.mjs.\n" +
  "# Do not edit manually.\n" +
  `__version__ = "${version}"\n`;

const runtimeContent =
  "// Auto-generated from python/pyproject.toml by scripts/sync_torch_version.mjs.\n" +
  "// Do not edit manually.\n" +
  `export const TORCH_PYODIDE_VERSION = "${version}" as const;\n`;

writeFileSync(pythonVersionPath, pythonContent);
writeFileSync(runtimeVersionPath, runtimeContent);
console.log(`[sync_torch_version] wrote ${pythonVersionPath} -> __version__ = "${version}"`);
console.log(`[sync_torch_version] wrote ${runtimeVersionPath} -> TORCH_PYODIDE_VERSION = "${version}"`);
