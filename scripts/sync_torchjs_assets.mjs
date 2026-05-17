import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const manifestPath = path.join(repoRoot, "scripts", "sync-manifest.json");

async function main() {
  const raw = await fs.readFile(manifestPath, "utf8");
  const manifest = JSON.parse(raw);

  const sourceRoot = manifest.sourceRoot;
  if (!sourceRoot) {
    throw new Error("sync-manifest.json requires sourceRoot");
  }

  for (const entry of manifest.files) {
    const source = path.resolve(sourceRoot, entry.from);
    const target = path.resolve(repoRoot, entry.to);
    await fs.mkdir(path.dirname(target), { recursive: true });
    await fs.copyFile(source, target);
    console.log(`synced: ${entry.from} -> ${entry.to}`);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
