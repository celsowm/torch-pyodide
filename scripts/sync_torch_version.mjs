#!/usr/bin/env node
// Generates python/torch/_version.py from python/pyproject.toml so that the
// Python package and the bundled JS runtime always expose the same version.
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const pyprojectPath = resolve(root, "python/pyproject.toml");
const versionPath = resolve(root, "python/torch/_version.py");

const pyproject = readFileSync(pyprojectPath, "utf8");
const match = pyproject.match(/^\s*version\s*=\s*"([^"]+)"/m);
if (!match) {
  console.error(`[sync_torch_version] could not find version in ${pyprojectPath}`);
  process.exit(1);
}
const version = match[1];

const content =
  "# Auto-generated from python/pyproject.toml by scripts/sync_torch_version.mjs.\n" +
  "# Do not edit manually.\n" +
  `__version__ = "${version}"\n`;

writeFileSync(versionPath, content);
console.log(`[sync_torch_version] wrote ${versionPath} -> __version__ = "${version}"`);
