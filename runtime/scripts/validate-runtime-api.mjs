// Validates that all runtime methods called from the Python layer exist
// on the TorchPyodideRuntime class in runtime.ts.
// Run: node scripts/validate-runtime-api.mjs
// Exits with code 1 on mismatch.

import { readFileSync, existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..", "..");

// ---------------------------------------------------------------------------
// 1. Extract all runtime.methodName() calls from Python files
// ---------------------------------------------------------------------------
const pythonDir = resolve(ROOT, "python/torch");
const pythonFiles = [
  "_tensor.py",
  "_runtime.py",
  "__init__.py",
  "cuda.py",
];

const pythonMethods = new Set();
const pythonCallRe = /runtime\.(\w+)\s*\(/g;

for (const pyFile of pythonFiles) {
  const filePath = resolve(pythonDir, pyFile);
  if (!existsSync(filePath)) {
    console.error(`  File not found: ${filePath}`);
    continue;
  }
  const content = readFileSync(filePath, "utf-8");
  let m;
  while ((m = pythonCallRe.exec(content)) !== null) {
    pythonMethods.add(m[1]);
  }
}

// Add methods called via attribute access (e.g. self._id for property-like access)
// Also add any methods used indirectly
// Also check for runtime.device etc property accesses

const pythonProps = new Set();
const pythonPropRe = /runtime\.(\w+)\s*(?!\()/g;
for (const pyFile of pythonFiles) {
  const filePath = resolve(pythonDir, pyFile);
  if (!existsSync(filePath)) continue;
  const content = readFileSync(filePath, "utf-8");
  let m;
  while ((m = pythonPropRe.exec(content)) !== null) {
    // If it's a method call, already caught above; otherwise treat as property
    if (!pythonMethods.has(m[1])) {
      pythonProps.add(m[1]);
    }
  }
}

// ---------------------------------------------------------------------------
// 2. Extract all methods from TorchPyodideRuntime class in runtime.ts
// ---------------------------------------------------------------------------
const runtimeTsPath = resolve(ROOT, "runtime/src/runtime.ts");
const runtimeContent = readFileSync(runtimeTsPath, "utf-8");

// Match async methodName(param...): type { ... or methodName(param...) { ...
// Only public methods (not prefixed with # or private)
const methodsRe = /^\s{2}async\s+(\w+)\s*\([^)]*\)\s*:/gm;
const tsMethods = new Set();
let mM;
while ((mM = methodsRe.exec(runtimeContent)) !== null) {
  tsMethods.add(mM[1]);
}

// Also match non-async methods
const syncMethodsRe = /^\s{2}(?:async\s+)?(\w+)\s*\([^)]*\)\s*(?::\s*\w+(?:\[\])?(?:\s*\|\s*\w+(?:\[\])?)*)?\s*{/gm;
while ((mM = syncMethodsRe.exec(runtimeContent)) !== null) {
  tsMethods.add(mM[1]);
}

// Filter out private methods (starting with # or prefixed with private)
const privateMethods = /^\s{2}(?:private|#)\s+\w+/gm;
while ((mM = privateMethods.exec(runtimeContent)) !== null) {
  const name = mM[0].trim().split(/\s+/)[1];
  tsMethods.delete(name);
}

// ---------------------------------------------------------------------------
// 3. Validate
// ---------------------------------------------------------------------------
const errors = [];

// Every Python-called method must exist in the TS runtime
for (const pyMethod of pythonMethods) {
  if (!tsMethods.has(pyMethod)) {
    errors.push(`  Missing runtime method: "${pyMethod}" (called from Python but not defined in runtime.ts)`);
  }
}

if (errors.length > 0) {
  console.error(`\n❌ Runtime API validation failed — ${errors.length} error(s):\n`);
  for (const err of errors) {
    console.error(err);
  }
  console.error("\n");
  process.exit(1);
} else {
  console.log(`✅ All ${pythonMethods.size} runtime methods called from Python are defined in runtime.ts.`);
}
