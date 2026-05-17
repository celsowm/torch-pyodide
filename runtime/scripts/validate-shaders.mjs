// Validates that all WGSL entrypoints used in ops/*.ts exist in the shader files.
// Run: node scripts/validate-shaders.mjs
// Exits with code 1 on mismatch.

import { readFileSync, readdirSync, existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const SRC = resolve(__dirname, "..", "src");

// ---------------------------------------------------------------------------
// 1. Scan all .wgsl files and index their entrypoints
// ---------------------------------------------------------------------------
const shadersDir = resolve(SRC, "vendor/torchjs/shaders");

/** Map from import constant name (e.g. "UNARY_SHADER") to list of entrypoints found in that .wgsl */
const availableEntrypoints = {};

for (const file of readdirSync(shadersDir)) {
  if (!file.endsWith(".wgsl")) continue;
  const content = readFileSync(resolve(shadersDir, file), "utf-8");
  const entrypoints = [];
  // Match `fn ENTRYPOINT_NAME(` at the start of a compute shader
  const re = /@compute[\s\S]*?fn\s+(\w+)\s*\(/g;
  let mEntry;
  while ((mEntry = re.exec(content)) !== null) {
    entrypoints.push(mEntry[1]);
  }
  // The import constant name is the uppercase version of the file base name + _SHADER
  // e.g. unary.wgsl -> UNARY_SHADER, leaky_relu.wgsl -> LEAKY_RELU_SHADER
  const baseName = file.replace(/\.wgsl$/, "").toUpperCase().replace(/-/g, "_") + "_SHADER";
  availableEntrypoints[baseName] = entrypoints;
}

// ---------------------------------------------------------------------------
// 2. Map from import constant name to variable name used in getOrCreatePipeline calls
//    We need to know which .wgsl file corresponds to which import in the ops code.
//    We parse the imports in each ops file to create a mapping.
// ---------------------------------------------------------------------------
// Read the barrel file to see which shaders are re-exported as what
// Actually easier: read the actual shader index to know which import name -> which .wgsl

const shaderIndexPath = resolve(SRC, "vendor/torchjs/shaders/index.ts");
const shaderIndexContent = readFileSync(shaderIndexPath, "utf-8");

// Build map: import name -> shader constant name (e.g. "UNARY_SHADER" -> "unary.wgsl")
const importToShader = {};
const importRe = /import\s+(\w+)\s+from\s+['"]\.\/([\w-]+)\.wgsl['"]/g;
let mImport;
while ((mImport = importRe.exec(shaderIndexContent)) !== null) {
  importToShader[mImport[1]] = mImport[2] + ".wgsl";
}

// ---------------------------------------------------------------------------
// 3. Scan ops/*.ts and runtime.ts for getOrCreatePipeline calls
// ---------------------------------------------------------------------------
const opsDir = resolve(SRC, "ops");
const runtimeFile = resolve(SRC, "runtime.ts");

const filesToScan = [
  ...(existsSync(opsDir) ? readdirSync(opsDir).filter(f => f.endsWith(".ts")) : []).map(f => resolve(opsDir, f)),
  runtimeFile,
];

const errors = [];

for (const filePath of filesToScan) {
  const content = readFileSync(filePath, "utf-8");
  const lines = content.split("\n");

  // Which shader imports are used in this file?
  const localShaderVars = {};
  for (const line of lines) {
    // Match: import { ..., UNARY_SHADER, ... } from ...
    const importMatch = line.match(/\{\s*([\w,\s]+)\s*\}/);
    if (!importMatch) continue;
    const vars = importMatch[1].split(",").map(v => v.trim());
    for (const v of vars) {
      if (importToShader[v]) {
        // Find which local variable holds this shader (same name usually)
        localShaderVars[v] = importToShader[v];
      }
    }
  }

  // For each getOrCreatePipeline(SHADER_VAR, "entrypoint") call, validate
  const callRe = /getOrCreatePipeline\((\w+),\s*["']([\w_]+)["']\)/g;
  let mCall;
  while ((mCall = callRe.exec(content)) !== null) {
    const shaderVar = mCall[1];
    const usedEntrypoint = mCall[2];

    // Determine which .wgsl this shaderVar refers to
    // First check localShaderVars, then check if it's a known import
    const wgslFile = localShaderVars[shaderVar] || importToShader[shaderVar];
    if (!wgslFile) {
      // Maybe the shader is used directly (e.g. LEAKY_RELU_SHADER)
      // Try the importToShader map directly
      if (!importToShader[shaderVar]) {
        errors.push(`  ${relativePath(filePath)}: cannot determine which .wgsl "${shaderVar}" refers to`);
        continue;
      }
    }

    const actualWgsl = importToShader[shaderVar];
    if (!actualWgsl) {
      errors.push(`  ${relativePath(filePath)}: shader "${shaderVar}" not found in shader index`);
      continue;
    }

    const shaderConstant = actualWgsl.replace(/\.wgsl$/, "").toUpperCase().replace(/-/g, "_") + "_SHADER";
    const available = availableEntrypoints[shaderConstant];
    if (!available) {
      errors.push(`  ${relativePath(filePath)}: no entrypoints found for shader "${actualWgsl}" (constant ${shaderConstant})`);
      continue;
    }

    if (!available.includes(usedEntrypoint)) {
      errors.push(
        `  ${relativePath(filePath)}: entrypoint "${usedEntrypoint}" not found in ${actualWgsl}. ` +
        `Available: [${available.join(", ")}]`
      );
    }
  }
}

function relativePath(absPath) {
  return absPath.startsWith(SRC) ? absPath.substring(SRC.length + 1) : absPath;
}

// ---------------------------------------------------------------------------
// 4. Report
// ---------------------------------------------------------------------------
if (errors.length > 0) {
  console.error(`\n❌ Shader entrypoint validation failed — ${errors.length} error(s):\n`);
  for (const err of errors) {
    console.error(err);
  }
  console.error("\n");
  process.exit(1);
} else {
  console.log("✅ All shader entrypoints validated successfully.");
}
