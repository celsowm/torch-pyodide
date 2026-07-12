import { expect, test, Page } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { writeFileSync } from "node:fs";

const testDir = path.dirname(fileURLToPath(import.meta.url));
const runtimeRoot = path.resolve(testDir, "..");

async function waitForPlaygroundReady(page: Page): Promise<void> {
  await page.goto("/playground/?force_fallback=1", { timeout: 180000 });
  await page.context().clearCookies();
  try {
    await page.evaluate(async () => {
      if (indexedDB && indexedDB.databases) {
        const dbs = await indexedDB.databases();
        await Promise.all(dbs.map((d) => new Promise((res) => {
          const req = indexedDB.deleteDatabase(d.name as string);
          req.onsuccess = req.onerror = req.onblocked = () => res(null);
        })));
      }
    });
  } catch { /* ignore */ }
  await page.waitForFunction(
    () => {
      const meta = document.getElementById("meta");
      return Boolean(meta?.textContent?.startsWith("Ready."));
    },
    { timeout: 300000 },
  );
}

async function runSelectedExample(page: Page, exampleId: string, timeoutMs = 120000): Promise<string> {
  const runButton = page.locator("#run");
  await expect(runButton).toBeEnabled({ timeout: 300000 });
  await page.locator("#output").evaluate((node) => { node.textContent = ""; });
  await page.locator("#example-select").selectOption(exampleId);
  await runButton.click();
  await page.waitForFunction(
    () => {
      const run = document.getElementById("run") as HTMLButtonElement | null;
      const output = document.getElementById("output");
      const text = output?.textContent ?? "";
      return Boolean(!run?.disabled && text.length > 0 && text !== "Running...");
    },
    { timeout: timeoutMs },
  );
  return page.locator("#output").innerText();
}

test("distributions example runs and matches log_prob parity @webgpu", async ({ browser }) => {
  const page = await browser.newPage();
  const consoleFailures: string[] = [];
  page.on("console", (msg) => { if (msg.type() === "error") consoleFailures.push(msg.text()); });
  page.on("pageerror", (e) => consoleFailures.push(e.message));

  await waitForPlaygroundReady(page);
  const output = await runSelectedExample(page, "distributions");
  await page.close();

  writeFileSync("C:/Temp/dist_out.txt", output);
  const trimmed = output.trim();
  if (!trimmed.startsWith("{")) {
    console.log("RAW_OUTPUT", trimmed.slice(0, 4000));
    throw new Error("example did not emit JSON");
  }
  const sanitized = trimmed.replace(/\bNaN\b/g, "null").replace(/\bInfinity\b/g, "null");
  const parsed = JSON.parse(sanitized);
  console.log("PER_DIFFS", JSON.stringify(parsed.per));
  console.log("RSAMPLE", JSON.stringify(parsed.rsample));
  expect(parsed.ok).toBe(true);
  expect(parsed.max_diff).toBeLessThan(1e-4);
  expect(consoleFailures).toEqual([]);
});
