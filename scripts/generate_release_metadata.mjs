#!/usr/bin/env node
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync, copyFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve, join } from "node:path";

function readVersionFromPyproject(pathname) {
  const content = readFileSync(pathname, "utf8");
  const match = content.match(/^\s*version\s*=\s*"([^"]+)"/m);
  if (!match) {
    throw new Error(`Could not find version in ${pathname}`);
  }
  return match[1];
}

function sha256Hex(buffer) {
  return createHash("sha256").update(buffer).digest("hex");
}

function normalizeBaseUrl(url) {
  return url.replace(/\/+$/, "");
}

function ensureDir(pathname) {
  mkdirSync(pathname, { recursive: true });
}

function buildUrl(baseUrl, relativePath) {
  const normalizedBase = normalizeBaseUrl(baseUrl);
  const normalizedRelative = relativePath.replace(/^\/+/, "");
  return `${normalizedBase}/${normalizedRelative}`;
}

async function sleep(ms) {
  await new Promise((resolveSleep) => setTimeout(resolveSleep, ms));
}

async function resolveWheelFromPyPI(packageName, version, maxAttempts, retryDelayMs) {
  const url = `https://pypi.org/pypi/${packageName}/${version}/json`;
  let lastError = null;

  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status} from ${url}`);
      }
      const payload = await response.json();
      const files = Array.isArray(payload.urls) ? payload.urls : [];
      const wheel =
        files.find((item) => item.packagetype === "bdist_wheel" && item.filename?.endsWith(".whl")) ??
        null;
      if (!wheel?.url) {
        throw new Error(`No wheel found in PyPI metadata for ${packageName} ${version}`);
      }
      return {
        wheelUrl: wheel.url,
        filename: wheel.filename ?? null,
      };
    } catch (error) {
      lastError = error;
      const isLastAttempt = attempt === maxAttempts;
      if (isLastAttempt) {
        break;
      }
      console.warn(
        `[generate_release_metadata] PyPI metadata not ready (attempt ${attempt}/${maxAttempts}): ${String(error)}. Retrying in ${retryDelayMs}ms.`,
      );
      await sleep(retryDelayMs);
    }
  }

  throw new Error(
    `[generate_release_metadata] Failed to resolve wheel from PyPI after ${maxAttempts} attempts: ${String(lastError)}`,
  );
}

async function downloadBytes(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} while downloading ${url}`);
  }
  const buffer = Buffer.from(await response.arrayBuffer());
  return buffer;
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} while fetching JSON ${url}`);
  }
  return response.json();
}

function isValidVersionEntry(value) {
  return typeof value === "string" && value.trim().length > 0;
}

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const pyprojectPath = resolve(root, "python/pyproject.toml");
const runtimeDistFile = resolve(root, "runtime/dist-distribution/runtime.mjs");
const outDir = resolve(root, process.env.RELEASE_METADATA_OUT_DIR ?? "runtime/dist-channel");

const packageName = process.env.PYPI_PACKAGE_NAME ?? "torch-pyodide";
const pagesBaseUrl =
  process.env.RELEASE_PAGES_BASE_URL ?? "https://celsowm.github.io/torch-pyodide";
const maxAttempts = Number(process.env.PYPI_MAX_ATTEMPTS ?? "20");
const retryDelayMs = Number(process.env.PYPI_RETRY_DELAY_MS ?? "15000");
const preservePriorVersions = process.env.RELEASE_PRESERVE_PRIOR_VERSIONS === "1";

const version = readVersionFromPyproject(pyprojectPath);
const runtimeBytes = readFileSync(runtimeDistFile);
const runtimeSha256 = sha256Hex(runtimeBytes);

const runtimeRelativePath = `runtime/${version}/runtime.mjs`;
const manifestRelativePath = `manifests/${version}.json`;
const latestRelativePath = "latest.json";

const runtimeOutputPath = join(outDir, runtimeRelativePath);
const manifestOutputPath = join(outDir, manifestRelativePath);
const latestOutputPath = join(outDir, latestRelativePath);
const releaseAssetsDir = join(outDir, "release-assets");

ensureDir(dirname(runtimeOutputPath));
ensureDir(dirname(manifestOutputPath));
ensureDir(dirname(latestOutputPath));
ensureDir(releaseAssetsDir);

copyFileSync(runtimeDistFile, runtimeOutputPath);

let knownVersions = [];
if (preservePriorVersions) {
  const versionsUrl = buildUrl(pagesBaseUrl, "versions.json");
  try {
    const existing = await fetchJson(versionsUrl);
    if (Array.isArray(existing.versions)) {
      knownVersions = existing.versions.filter(isValidVersionEntry);
    }
  } catch (error) {
    console.warn(
      `[generate_release_metadata] Could not load existing versions from ${versionsUrl}. Continuing with current version only. Cause: ${String(error)}`,
    );
  }

  for (const previousVersion of knownVersions) {
    if (previousVersion === version) {
      continue;
    }
    try {
      const previousManifestUrl = buildUrl(pagesBaseUrl, `manifests/${previousVersion}.json`);
      const previousManifest = await fetchJson(previousManifestUrl);
      const previousRuntimeBytes = await downloadBytes(previousManifest.runtimeUrl);
      const previousRuntimeOutputPath = join(outDir, `runtime/${previousVersion}/runtime.mjs`);
      const previousManifestOutputPath = join(outDir, `manifests/${previousVersion}.json`);
      ensureDir(dirname(previousRuntimeOutputPath));
      ensureDir(dirname(previousManifestOutputPath));
      writeFileSync(previousRuntimeOutputPath, previousRuntimeBytes);
      writeFileSync(previousManifestOutputPath, `${JSON.stringify(previousManifest, null, 2)}\n`);
    } catch (error) {
      console.warn(
        `[generate_release_metadata] Skipped preserving ${previousVersion}: ${String(error)}`,
      );
    }
  }
}

const { wheelUrl, filename } = await resolveWheelFromPyPI(
  packageName,
  version,
  maxAttempts,
  retryDelayMs,
);
const wheelBytes = await downloadBytes(wheelUrl);
const wheelSha256 = sha256Hex(wheelBytes);

const manifest = {
  torchVersion: version,
  runtimeUrl: buildUrl(pagesBaseUrl, runtimeRelativePath),
  wheelUrl,
  runtimeSha256,
  wheelSha256,
};

writeFileSync(manifestOutputPath, `${JSON.stringify(manifest, null, 2)}\n`);
writeFileSync(latestOutputPath, `${JSON.stringify(manifest, null, 2)}\n`);
const mergedVersions = Array.from(new Set([...knownVersions, version])).sort((a, b) =>
  a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" }),
);
writeFileSync(join(outDir, "versions.json"), `${JSON.stringify({ versions: mergedVersions }, null, 2)}\n`);
writeFileSync(join(releaseAssetsDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
writeFileSync(
  join(releaseAssetsDir, `manifest-${version}.json`),
  `${JSON.stringify(manifest, null, 2)}\n`,
);
copyFileSync(runtimeDistFile, join(releaseAssetsDir, "runtime.mjs"));
copyFileSync(runtimeDistFile, join(releaseAssetsDir, `runtime-${version}.mjs`));

console.log(`[generate_release_metadata] version=${version}`);
console.log(`[generate_release_metadata] runtimeSha256=${runtimeSha256}`);
console.log(`[generate_release_metadata] wheel=${filename ?? wheelUrl}`);
console.log(`[generate_release_metadata] wheelSha256=${wheelSha256}`);
console.log(`[generate_release_metadata] wrote ${runtimeOutputPath}`);
console.log(`[generate_release_metadata] wrote ${manifestOutputPath}`);
console.log(`[generate_release_metadata] wrote ${latestOutputPath}`);
console.log(`[generate_release_metadata] wrote ${join(outDir, "versions.json")}`);
