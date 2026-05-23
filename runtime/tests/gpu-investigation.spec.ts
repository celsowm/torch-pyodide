import { expect, test, Page } from "@playwright/test";

type ExampleMeta = {
  id: string;
  label: string;
  file: string;
};

type Catalog = {
  examples: ExampleMeta[];
};

type ExampleRun = {
  id: string;
  label: string;
  elapsedMs: number;
  failed: boolean;
  outputSnippet: string;
  warnings: string[];
  errors: string[];
  pageErrors: string[];
};

const TARGET_IDS = [
  "padding_modes",
  "training_sequential_mlp",
  "training_classification",
];

const ignoredConsoleFragments = [
  "Failed to load resource: the server responded with a status of 404",
  "favicon.ico",
  "[torch-pyodide] Using bundled local fallback",
  "[torch-pyodide] WebGPU active:",
  "torch version installed:",
  "TORCH VERSION:",
];

function isIgnoredConsoleMessage(text: string): boolean {
  return ignoredConsoleFragments.some((fragment) => text.includes(fragment));
}

async function waitForPlaygroundReady(page: Page): Promise<void> {
  await page.goto("/playground/", { timeout: 180000 });
  await page.waitForFunction(
    () => {
      const meta = document.getElementById("meta");
      return Boolean(meta?.textContent?.startsWith("Ready."));
    },
    { timeout: 300000 },
  );
}

async function runSelectedExample(page: Page): Promise<{ output: string; elapsedMs: number }> {
  const startedAt = Date.now();
  await page.locator("#output").evaluate((node) => {
    node.textContent = "";
  });
  await page.locator("#run").click();
  await page.waitForFunction(
    () => {
      const output = document.getElementById("output");
      return Boolean(output?.textContent && output.textContent.length > 0);
    },
    { timeout: 600000 },
  );
  const output = await page.locator("#output").innerText();
  const elapsedMs = Date.now() - startedAt;
  return { output, elapsedMs };
}

test.describe.serial("GPU investigation @webgpu @investigate", () => {
  let page: Page;
  let examplesById = new Map<string, ExampleMeta>();

  test.beforeAll(async ({ browser }) => {
    page = await browser.newPage();
    await waitForPlaygroundReady(page);

    const catalog = (await page.evaluate(async () => {
      const response = await fetch("/examples.json", { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`Failed to load examples catalog: HTTP ${response.status}`);
      }
      return response.json();
    })) as Catalog;

    for (const ex of catalog.examples) {
      examplesById.set(ex.id, ex);
    }
  });

  test.afterAll(async () => {
    await page?.close();
  });

  test("collect timings and warnings for target examples", async () => {
    test.setTimeout(30 * 60 * 1000);

    const missing = TARGET_IDS.filter((id) => !examplesById.has(id));
    expect(missing, `Missing example ids in catalog: ${missing.join(", ")}`).toEqual([]);

    const runs: ExampleRun[] = [];

    for (const id of TARGET_IDS) {
      const meta = examplesById.get(id)!;
      const warnings: string[] = [];
      const errors: string[] = [];
      const pageErrors: string[] = [];

      const onConsole = (msg: { type: () => string; text: () => string }) => {
        const type = msg.type();
        const text = msg.text();
        if (isIgnoredConsoleMessage(text)) return;
        if (type === "warning") warnings.push(text);
        if (type === "error") errors.push(text);
      };
      const onPageError = (err: Error) => {
        pageErrors.push(err.message);
      };

      page.on("console", onConsole);
      page.on("pageerror", onPageError);
      await page.locator("#example-select").selectOption(id);
      await expect(page.locator("#example-select")).toHaveValue(id);

      const { output, elapsedMs } = await runSelectedExample(page);
      page.off("console", onConsole);
      page.off("pageerror", onPageError);

      const failed =
        output.startsWith("ERROR") ||
        output.includes("Traceback") ||
        output.includes("PythonError") ||
        output.includes("Failed to get WebGPU adapter");

      runs.push({
        id,
        label: meta.label,
        elapsedMs,
        failed,
        outputSnippet: output.slice(0, 500),
        warnings,
        errors,
        pageErrors,
      });
    }

    const ordered = [...runs].sort((a, b) => b.elapsedMs - a.elapsedMs);
    const summary = ordered
      .map((r) => {
        return `${r.id} (${r.label}) | ${(r.elapsedMs / 1000).toFixed(2)}s | failed=${r.failed} | warnings=${r.warnings.length} | errors=${r.errors.length} | pageErrors=${r.pageErrors.length}`;
      })
      .join("\n");
    console.log(`\nGPU investigation summary:\n${summary}\n`);

    const warningSet = new Map<string, number>();
    for (const run of runs) {
      for (const w of run.warnings) {
        warningSet.set(w, (warningSet.get(w) ?? 0) + 1);
      }
      for (const e of run.errors) {
        warningSet.set(e, (warningSet.get(e) ?? 0) + 1);
      }
      for (const pe of run.pageErrors) {
        warningSet.set(pe, (warningSet.get(pe) ?? 0) + 1);
      }
    }

    if (warningSet.size > 0) {
      const warningSummary = [...warningSet.entries()]
        .sort((a, b) => b[1] - a[1])
        .map(([msg, count]) => `${count}x ${msg}`)
        .join("\n");
      console.log(`Unique warnings/errors:\n${warningSummary}\n`);
    } else {
      console.log("Unique warnings/errors:\n(none)\n");
    }

    const failedRuns = runs.filter((r) => r.failed || r.errors.length > 0 || r.pageErrors.length > 0);
    expect(
      failedRuns.map((r) => `${r.id}: failed=${r.failed}, errors=${r.errors.length}, pageErrors=${r.pageErrors.length}`),
      failedRuns.map((r) => `${r.id}\n${r.outputSnippet}`).join("\n\n---\n\n"),
    ).toEqual([]);
  });
});

