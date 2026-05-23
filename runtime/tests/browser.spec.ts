import { expect, test, Page } from "@playwright/test";

type ExampleMeta = {
  id: string;
  label: string;
  file: string;
};

type Catalog = {
  examples: ExampleMeta[];
};

const ignoredConsoleFragments = [
  "Failed to load resource: the server responded with a status of 404",
  "favicon.ico",
  "powerPreference option is currently ignored",
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

test.describe.serial("playground examples @webgpu", () => {
  let page: Page;
  let examples: ExampleMeta[];
  const consoleFailures: string[] = [];

  test.beforeAll(async ({ browser }) => {
    page = await browser.newPage();
    page.on("console", (msg) => {
      if (msg.type() !== "error") {
        return;
      }
      const text = msg.text();
      if (!isIgnoredConsoleMessage(text)) {
        consoleFailures.push(`[console.${msg.type()}] ${text}`);
      }
    });
    page.on("pageerror", (error) => {
      consoleFailures.push(`[pageerror] ${error.message}`);
    });

    await waitForPlaygroundReady(page);
    const catalog = await page.evaluate(async () => {
      const response = await fetch("/examples.json", { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`Failed to load examples catalog: HTTP ${response.status}`);
      }
      return response.json();
    }) as Catalog;
    examples = catalog.examples;
    expect(examples.length).toBeGreaterThan(0);
  });

  test.afterAll(async () => {
    await page?.close();
  });

  test("catalog contains every selectable option", async () => {
    const optionIds = await page.locator("#example-select option").evaluateAll((options) =>
      options.map((option) => (option as HTMLOptionElement).value),
    );
    expect(optionIds).toEqual(examples.map((example) => example.id));
  });

  test("all playground examples run without Python or WebGPU errors", async () => {
    test.setTimeout(30 * 60 * 1000);
    const failures: string[] = [];
    const timings: Array<{ id: string; label: string; elapsedMs: number }> = [];

    for (const example of examples) {
      consoleFailures.length = 0;
      await page.locator("#example-select").selectOption(example.id);
      await expect(page.locator("#example-select")).toHaveValue(example.id);

      const runResult = await runSelectedExample(page);
      const outputText = runResult.output;
      timings.push({ id: example.id, label: example.label, elapsedMs: runResult.elapsedMs });
      const outputFailed =
        outputText.startsWith("ERROR") ||
        outputText.includes("Traceback") ||
        outputText.includes("PythonError") ||
        outputText.includes("Failed to get WebGPU adapter");

      if (outputFailed || consoleFailures.length > 0) {
        failures.push(
          [
            `${example.id} (${example.label})`,
            outputFailed ? outputText.slice(0, 1000) : "",
            ...consoleFailures,
          ]
            .filter(Boolean)
            .join("\n"),
        );
      }
    }

    const slowest = [...timings]
      .sort((a, b) => b.elapsedMs - a.elapsedMs)
      .slice(0, 10);
    const timingSummary = slowest
      .map((t) => `${t.id} (${t.label}): ${(t.elapsedMs / 1000).toFixed(2)}s`)
      .join("\n");
    test.info().annotations.push({
      type: "timings",
      description: timingSummary,
    });
    console.log(`\nTop 10 slowest playground examples:\n${timingSummary}\n`);

    expect(failures, failures.join("\n\n---\n\n")).toEqual([]);
  });
});
