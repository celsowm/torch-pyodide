// Validates that TypeScript source files use only identifiers that are explicitly
// imported or are well-known globals. Catches cases like:
//   - BufferUsage used in arithmeticOps.ts but not imported from ./utils.js
//   - REDUCE_DIM_SHADER used in reductionOps.ts but not imported
//
// Strategy: for each *._SHADER constant and BufferUsage, check if the name
// appears in the file body outside of import statements.
//
// Run: node scripts/validate-imports.mjs
// Exits with code 1 on mismatch.

import { readFileSync, existsSync, readdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..", "src");
const OPS_DIR = resolve(ROOT, "ops");
const RUNTIME_TS = resolve(ROOT, "runtime.ts");

const filesToCheck = [
  RUNTIME_TS,
  ...(existsSync(OPS_DIR)
    ? readdirSync(OPS_DIR).filter((f) => f.endsWith(".ts")).map((f) => resolve(OPS_DIR, f))
    : []),
].filter(existsSync);

// ---------------------------------------------------------------------------
// 1. Parse all import statements in a file
// ---------------------------------------------------------------------------
function parseImports(content) {
  const imports = new Set();

  // import { A, B as C } from "..."
  const namedRe = /import\s+\{([^}]+)\}\s+from\s+/g;
  let m;
  while ((m = namedRe.exec(content)) !== null) {
    for (const spec of m[1].split(",")) {
      const trimmed = spec.trim();
      if (trimmed.startsWith("type ")) continue;
      // Handle "A as B" -> track B
      const asMatch = trimmed.match(/(\w+)\s+as\s+(\w+)/);
      imports.add(asMatch ? asMatch[2] : trimmed);
    }
  }

  // import X from "..."
  const defaultRe = /import\s+(\w+)\s+from\s+/g;
  while ((m = defaultRe.exec(content)) !== null) imports.add(m[1]);

  // import * as X from "..."
  const starRe = /import\s+\*\s+as\s+(\w+)\s+from\s+/g;
  while ((m = starRe.exec(content)) !== null) imports.add(m[1]);

  return imports;
}

// ---------------------------------------------------------------------------
// 2. For each file, look for usage of known exported names that should be
//    imported. We specifically look for these patterns:
//      - UPPER_CASE constant (BufferUsage, REDUCE_DIM_SHADER, etc.)
//      - Known WebGPU symbols
// ---------------------------------------------------------------------------
const KNOWN_CONSTANTS = [
  "BufferUsage",
  // All _SHADER constants (we'll detect dynamically)
];

const errors = [];

for (const filePath of filesToCheck) {
  const content = readFileSync(filePath, "utf-8");
  const imports = parseImports(content);

  // Find all imports ending with _SHADER (both the import names and their aliases)
  // We don't need to enumerate them; we just check usage vs imports.

  // Remove import/export/comment lines, but keep track of line numbers for errors
  const lines = content.split("\n");
  const bodyLines = [];
  for (const line of lines) {
    const trimmed = line.trim();
    if (
      trimmed.startsWith("import ") ||
      trimmed.startsWith("export ") ||
      trimmed.startsWith("//") ||
      trimmed.startsWith("/*") ||
      trimmed.startsWith("*") ||
      trimmed.startsWith("} from")
    ) {
      continue;
    }
    bodyLines.push(line);
  }
  const body = bodyLines.join("\n");

  // Clean comments and strings from body
  const cleaned = body
    .replace(/\/\/.*$/gm, "")
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/`(?:[^`\\]|\\.)*`/g, "``")
    .replace(/'(?:[^'\\]|\\.)*'/g, "''")
    .replace(/"(?:[^"\\]|\\.)*"/g, '""');

  // Pattern 1: UPPER_CASE constant used in expressions (as value, not member access)
  const wordRe = /\b([A-Z][A-Z_0-9]+)\b/g;
  let m;
  while ((m = wordRe.exec(cleaned)) !== null) {
    const name = m[1];

    // Skip very short names (likely typegpu internal)
    if (name === "S" || name === "R" || name === "T" || name === "F" || name === "N") continue;

    // Skip known globals
    if (
      name === "GPUBufferUsage" || name === "GPUMapMode" ||
      name === "GPUDevice" || name === "GPUAdapter" ||
      name === "GPUQueue" || name === "GPUErrorFilter" ||
      name === "RED" || name === "GREEN" || name === "BLUE" || name === "ALPHA"
    ) continue;

    // Skip if it's defined locally
    const localDefRe = new RegExp(
      `\\b(const|let|var|function|class|interface|type|enum)\\s+${name}\\b`,
    );
    if (localDefRe.test(body)) continue;

    // Skip if it's imported
    if (imports.has(name)) continue;

    // Only flag if it looks like a shared constant (BufferUsage or *._SHADER)
    if (name === "BufferUsage" || name.endsWith("_SHADER")) {
      errors.push(
        `  ${relativePath(filePath)}: "${name}" is used but not imported`
      );
    }
  }
}

function relativePath(absPath) {
  return absPath.startsWith(ROOT) ? absPath.substring(ROOT.length + 1) : absPath;
}

if (errors.length > 0) {
  console.error(`\n❌ Import validation failed — ${errors.length} error(s):\n`);
  for (const err of errors) {
    console.error(err);
  }
  console.error("\n");
  process.exit(1);
} else {
  console.log("✅ All imports validated successfully (no dangling references found).");
}
