#!/usr/bin/env node
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

function sha256Hex(buffer) {
  return createHash("sha256").update(buffer).digest("hex");
}

function normalizeManifest(manifest) {
  const required = [
    "torchVersion",
    "runtimeUrl",
    "wheelUrl",
    "runtimeSha256",
    "wheelSha256",
  ];
  for (const key of required) {
    if (!manifest[key] || typeof manifest[key] !== "string") {
      throw new Error(`Manifest missing or invalid "${key}"`);
    }
  }
  return manifest;
}

async function fetchBytes(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} while fetching ${url}`);
  }
  return Buffer.from(await response.arrayBuffer());
}

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const manifestPath = resolve(root, process.env.RELEASE_MANIFEST_PATH ?? "runtime/dist-channel/latest.json");
const manifest = normalizeManifest(JSON.parse(readFileSync(manifestPath, "utf8")));
const runtimeLocalPath = process.env.RELEASE_RUNTIME_LOCAL_PATH
  ? resolve(root, process.env.RELEASE_RUNTIME_LOCAL_PATH)
  : null;

const wheelBytes = await fetchBytes(manifest.wheelUrl);
const runtimeBytes = runtimeLocalPath ? readFileSync(runtimeLocalPath) : await fetchBytes(manifest.runtimeUrl);

const runtimeSha256 = sha256Hex(runtimeBytes);
const wheelSha256 = sha256Hex(wheelBytes);

if (runtimeSha256 !== manifest.runtimeSha256) {
  throw new Error(
    `runtimeSha256 mismatch: manifest=${manifest.runtimeSha256} computed=${runtimeSha256}`,
  );
}

if (wheelSha256 !== manifest.wheelSha256) {
  throw new Error(`wheelSha256 mismatch: manifest=${manifest.wheelSha256} computed=${wheelSha256}`);
}

console.log(`[verify_release_metadata_integrity] OK for torchVersion=${manifest.torchVersion}`);
