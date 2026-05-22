import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const PYODIDE_VERSION = "0.29.4";
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const distTarget = path.join(repoRoot, "external", "pyodide", "dist");
const vitePublicTarget = path.join(repoRoot, "runtime", "public", "pyodide");

function run(command, args, cwd) {
  const result = spawnSync(command, args, { cwd, stdio: "inherit", shell: true });
  if (result.status !== 0) {
    throw new Error(`Command failed: ${command} ${args.join(" ")}`);
  }
}

async function main() {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "torch-pyodide-"));
  try {
    const npmBin = process.platform === "win32" ? "npm.cmd" : "npm";
    run(npmBin, ["init", "-y"], tempDir);
    run(npmBin, ["install", "--no-package-lock", `pyodide@${PYODIDE_VERSION}`], tempDir);

    const sourceDist = path.join(tempDir, "node_modules", "pyodide");
    await fs.rm(distTarget, { force: true, recursive: true });
    await fs.mkdir(path.dirname(distTarget), { recursive: true });
    await fs.cp(sourceDist, distTarget, { recursive: true });

    await fs.rm(vitePublicTarget, { force: true, recursive: true });
    await fs.mkdir(path.dirname(vitePublicTarget), { recursive: true });
    await fs.cp(sourceDist, vitePublicTarget, { recursive: true });

    console.log(`Pyodide ${PYODIDE_VERSION} installed into:`);
    console.log(`- ${distTarget}`);
    console.log(`- ${vitePublicTarget}`);
  } finally {
    await fs.rm(tempDir, { force: true, recursive: true });
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
