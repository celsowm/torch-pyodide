// Validates local fallback package assembly in runtime/demo/shared.ts.
// Ensures every top-level python/torch/*.py file is imported as ?raw and
// written into /home/pyodide/torch via FS.writeFile.
// Run: node scripts/validate-local-fallback.mjs

import { readdirSync, readFileSync } from "node:fs";
import { resolve, dirname, relative } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..", "..");
const torchDir = resolve(ROOT, "python", "torch");
const sharedTsPath = resolve(ROOT, "runtime", "demo", "shared.ts");

function collectPythonFiles(dir) {
  const entries = readdirSync(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const entryPath = resolve(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...collectPythonFiles(entryPath));
      continue;
    }
    if (entry.isFile() && entry.name.endsWith(".py")) {
      files.push(relative(torchDir, entryPath).replaceAll("\\", "/"));
    }
  }
  return files;
}

const pythonFiles = collectPythonFiles(torchDir).sort();

const sharedContent = readFileSync(sharedTsPath, "utf-8");

const importedFiles = new Set();
const importRe = /import\s+\w+\s+from\s+"..\/..\/python\/torch\/([^"?]+)\?raw";/g;
let m;
while ((m = importRe.exec(sharedContent)) !== null) {
  importedFiles.add(m[1]);
}

const writtenFiles = new Set();
const writeRe = /FS\.writeFile\("\/home\/pyodide\/torch\/([^"]+)",\s*\w+\);/g;
while ((m = writeRe.exec(sharedContent)) !== null) {
  writtenFiles.add(m[1]);
}

const errors = [];

for (const pyFile of pythonFiles) {
  if (!importedFiles.has(pyFile)) {
    errors.push(`Missing ?raw import for python/torch/${pyFile}`);
  }
  if (!writtenFiles.has(pyFile)) {
    errors.push(`Missing FS.writeFile for /home/pyodide/torch/${pyFile}`);
  }
}

for (const imported of importedFiles) {
  if (!pythonFiles.includes(imported)) {
    errors.push(`Import references non-existent file: python/torch/${imported}`);
  }
}

for (const written of writtenFiles) {
  if (!pythonFiles.includes(written)) {
    errors.push(`FS.writeFile references non-existent file: /home/pyodide/torch/${written}`);
  }
}

if (errors.length > 0) {
  console.error(`\n❌ Local fallback validation failed — ${errors.length} error(s):\n`);
  for (const error of errors) {
    console.error(`  ${error}`);
  }
  console.error("");
  process.exit(1);
}

console.log(`✅ Local fallback validated: ${pythonFiles.length} python/torch files are imported and written.`);
