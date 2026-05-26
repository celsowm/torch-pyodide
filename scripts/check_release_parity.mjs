#!/usr/bin/env node
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

function readVersionFromPyproject(pathname) {
  const content = readFileSync(pathname, "utf8");
  const match = content.match(/^\s*version\s*=\s*"([^"]+)"/m);
  if (!match) {
    throw new Error(`Could not find version in ${pathname}`);
  }
  return match[1];
}

function readPythonVersion(pathname) {
  const content = readFileSync(pathname, "utf8");
  const match = content.match(/^__version__\s*=\s*"([^"]+)"/m);
  if (!match) {
    throw new Error(`Could not find __version__ in ${pathname}`);
  }
  return match[1];
}

function readRuntimeVersion(pathname) {
  const content = readFileSync(pathname, "utf8");
  const match = content.match(/^export const TORCH_PYODIDE_VERSION = "([^"]+)"/m);
  if (!match) {
    throw new Error(`Could not find TORCH_PYODIDE_VERSION in ${pathname}`);
  }
  return match[1];
}

function resolveTagVersion() {
  if (process.env.GITHUB_REF_TYPE === "tag" && process.env.GITHUB_REF_NAME) {
    return process.env.GITHUB_REF_NAME.replace(/^v/, "");
  }
  const ref = process.env.GITHUB_REF ?? "";
  const match = ref.match(/^refs\/tags\/v(.+)$/);
  if (match) {
    return match[1];
  }
  return null;
}

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const pyprojectPath = resolve(root, "python/pyproject.toml");
const pythonVersionPath = resolve(root, "python/torch/_version.py");
const runtimeVersionPath = resolve(root, "runtime/src/version.ts");

const pyprojectVersion = readVersionFromPyproject(pyprojectPath);
const pythonVersion = readPythonVersion(pythonVersionPath);
const runtimeVersion = readRuntimeVersion(runtimeVersionPath);

if (pythonVersion !== pyprojectVersion) {
  throw new Error(
    `Version mismatch: python/torch/_version.py=${pythonVersion} but python/pyproject.toml=${pyprojectVersion}. Run npm run sync:version.`,
  );
}

if (runtimeVersion !== pyprojectVersion) {
  throw new Error(
    `Version mismatch: runtime/src/version.ts=${runtimeVersion} but python/pyproject.toml=${pyprojectVersion}. Run npm run sync:version.`,
  );
}

const tagVersion = resolveTagVersion();
if (tagVersion && tagVersion !== pyprojectVersion) {
  throw new Error(
    `Tag/version mismatch: git tag v${tagVersion} but python/pyproject.toml has ${pyprojectVersion}.`,
  );
}

console.log(
  `[check_release_parity] OK (pyproject=${pyprojectVersion}, python=${pythonVersion}, runtime=${runtimeVersion}${tagVersion ? `, tag=${tagVersion}` : ""})`,
);
