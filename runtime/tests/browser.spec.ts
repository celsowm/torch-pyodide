import { expect, test, Page } from "@playwright/test";
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

type ExampleMeta = {
  id: string;
  label: string;
  file: string;
};

type Catalog = {
  examples: ExampleMeta[];
};

type DeterministicParityOutput = {
  loss_before: number;
  loss_after: number;
  generated_ids: number[];
  generated_text: string;
  last_logits: number[];
};

const testDir = path.dirname(fileURLToPath(import.meta.url));
const runtimeRoot = path.resolve(testDir, "..");
const repoRoot = path.resolve(runtimeRoot, "..");

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

function parseJsonOutput<T>(output: string): T {
  return JSON.parse(output.trim()) as T;
}

function runExampleWithRealTorch(exampleFile: string): { output?: DeterministicParityOutput; skipReason?: string } {
  const python = process.env.PYTHON ?? "python";
  const examplePath = path.join(runtimeRoot, "playground", "public", "examples", exampleFile);
  const env = { ...process.env };
  delete env.PYTHONPATH;
  const result = spawnSync(python, [examplePath], {
    cwd: repoRoot,
    env,
    encoding: "utf-8",
    timeout: 120000,
  });

  const combinedError = `${result.stderr ?? ""}\n${result.stdout ?? ""}`;
  if (result.error) {
    if ((result.error as { code?: string }).code === "ENOENT") {
      return { skipReason: `Python executable not found: ${python}` };
    }
    throw result.error;
  }
  if (result.status !== 0) {
    if (
      combinedError.includes("ModuleNotFoundError: No module named 'torch'") ||
      combinedError.includes("ImportError: No module named torch")
    ) {
      return { skipReason: "PyTorch real is not installed in the local Python environment." };
    }
    throw new Error(`Real PyTorch example failed with exit code ${result.status}:\n${combinedError}`);
  }

  return { output: parseJsonOutput<DeterministicParityOutput>(result.stdout) };
}

async function waitForPlaygroundReady(page: Page): Promise<void> {
  await page.goto("/playground/?force_fallback=1", { timeout: 180000 });
  await page.waitForFunction(
    () => {
      const meta = document.getElementById("meta");
      return Boolean(meta?.textContent?.startsWith("Ready."));
    },
    { timeout: 300000 },
  );
}

async function runSelectedExample(
  page: Page,
  exampleId: string,
  timeoutMs: number = 45000,
): Promise<{ output: string; elapsedMs: number }> {
  const startedAt = Date.now();
  const runButton = page.locator("#run");
  await expect(runButton).toBeEnabled({ timeout: 300000 });
  await page.locator("#output").evaluate((node) => {
    node.textContent = "";
  });
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
  const elapsedMs = Date.now() - startedAt;
  if (elapsedMs > timeoutMs) {
    throw new Error(`Example "${exampleId}" exceeded timeout (${timeoutMs}ms).`);
  }
  const output = await page.locator("#output").innerText();
  return { output, elapsedMs };
}

test.describe.serial("playground examples @webgpu", () => {
  let page: Page;
  let examples: ExampleMeta[];
  const consoleFailures: string[] = [];

  test.beforeAll(async ({ browser }) => {
    page = await browser.newPage();
    page.on("console", (msg) => {
      const text = msg.text();
      if (!isIgnoredConsoleMessage(text)) {
        console.log(`[browser.${msg.type()}] ${text}`);
      }
      if (msg.type() === "error") {
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

  test("cat expand where backward returns expected gradients", async () => {
    consoleFailures.length = 0;
    await page.locator("#example-select").selectOption("autograd_cat_expand_where");
    await expect(page.locator("#example-select")).toHaveValue("autograd_cat_expand_where");

    const { output } = await runSelectedExample(page, "autograd_cat_expand_where");

    expect(output).toMatch(/grad a:\s*\[4(?:\.0)?, 4(?:\.0)?, 4(?:\.0)?\]/);
    expect(output).toMatch(/grad b:\s*\[1(?:\.0)?, 1(?:\.0)?, 1(?:\.0)?\]/);
    expect(output).toMatch(/grad x:\s*\[1(?:\.0)?, 0(?:\.0)?, 1(?:\.0)?\]/);
    expect(output).toMatch(/grad y:\s*\[0(?:\.0)?, 1(?:\.0)?, 0(?:\.0)?\]/);
    expect(consoleFailures).toEqual([]);
  });

  test("reduction broadcast where bool and scatter backward match expected gradients", async () => {
    consoleFailures.length = 0;
    await page.locator("#example-select").selectOption("autograd_reduction_broadcast_scatter");
    await expect(page.locator("#example-select")).toHaveValue("autograd_reduction_broadcast_scatter");

    const { output } = await runSelectedExample(page, "autograd_reduction_broadcast_scatter");

    expect(output).toMatch(/grad sum dim:\s*\[\[1(?:\.0)?, 1(?:\.0)?, 1(?:\.0)?\], \[1(?:\.0)?, 1(?:\.0)?, 1(?:\.0)?\]\]/);
    expect(output).toMatch(/grad mean dim:\s*\[\[0\.5, 0\.5, 0\.5\], \[0\.5, 0\.5, 0\.5\]\]/);
    expect(output).toMatch(/grad broadcast row:\s*\[\[2(?:\.0)?, 2(?:\.0)?, 2(?:\.0)?\]\]/);
    expect(output).toMatch(/grad broadcast col:\s*\[\[4(?:\.0)?\], \[4(?:\.0)?\], \[4(?:\.0)?\]\]/);
    expect(output).toMatch(/grad broadcast scalar:\s*6(?:\.0)?/);
    expect(output).toMatch(/grad where bool x:\s*\[1(?:\.0)?, 0(?:\.0)?, 1(?:\.0)?\]/);
    expect(output).toMatch(/grad where bool y:\s*\[0(?:\.0)?, 1(?:\.0)?, 0(?:\.0)?\]/);
    expect(output).toMatch(/grad scatter base:\s*\[0(?:\.0)?, 1(?:\.0)?, 0(?:\.0)?\]/);
    expect(output).toMatch(/grad scatter src:\s*\[1(?:\.0)?, 1(?:\.0)?\]/);
    expect(consoleFailures).toEqual([]);
  });

  test("single parameter SGD step stays finite and updates weight", async () => {
    consoleFailures.length = 0;
    await page.locator("#example-select").selectOption("autograd_sgd_single_param");
    await expect(page.locator("#example-select")).toHaveValue("autograd_sgd_single_param");

    const { output } = await runSelectedExample(page, "autograd_sgd_single_param");

    expect(output).not.toMatch(/nan|NaN|inf|Infinity/);
    expect(output).toMatch(/loss:\s*4(?:\.0)?/);
    expect(output).toMatch(/updated_w:\s*0\.4(?:0+)?/);
    expect(consoleFailures).toEqual([]);
  });

  test("embedding linear language-model smoke returns expected shapes", async () => {
    consoleFailures.length = 0;
    await page.locator("#example-select").selectOption("nn_embedding_language_model_smoke");
    await expect(page.locator("#example-select")).toHaveValue("nn_embedding_language_model_smoke");

    const { output } = await runSelectedExample(page, "nn_embedding_language_model_smoke");

    expect(output).not.toMatch(/nan|NaN|inf|Infinity|Traceback|ERROR/);
    expect(output).toContain("H: (2, 4, 8)");
    expect(output).toContain("logits: (2, 4, 100)");
    expect(output).toContain("flat: (8, 100) (8,)");
    expect(output).toContain("next: (2, 100)");
    expect(output).toContain("loss: ()");
    expect(consoleFailures).toEqual([]);
  });

  test("tiny bigram language model trains and generates finite text", async () => {
    consoleFailures.length = 0;
    await page.locator("#example-select").selectOption("nn_tiny_bigram_lm_training");
    await expect(page.locator("#example-select")).toHaveValue("nn_tiny_bigram_lm_training");

    const { output } = await runSelectedExample(page, "nn_tiny_bigram_lm_training", 120000);

    expect(output).not.toMatch(/nan|NaN|inf|Infinity|Traceback|ERROR/);
    const before = output.match(/loss before:\s*([0-9.]+)/);
    const after = output.match(/loss after:\s*([0-9.]+)/);
    expect(before).not.toBeNull();
    expect(after).not.toBeNull();
    expect(Number(after?.[1])).toBeLessThan(Number(before?.[1]));
    expect(output).toMatch(/generated:\s*I(?:\s+(?:\.|AI|I|like|pytorch)){6}/);
    expect(consoleFailures).toEqual([]);
  });

  test("tiny bigram deterministic parity matches real PyTorch", async () => {
    const reference = runExampleWithRealTorch("nn_tiny_bigram_lm_deterministic.py");
    if (reference.skipReason) {
      test.skip(true, reference.skipReason);
    }
    expect(reference.output).toBeDefined();

    consoleFailures.length = 0;
    await page.locator("#example-select").selectOption("nn_tiny_bigram_lm_deterministic");
    await expect(page.locator("#example-select")).toHaveValue("nn_tiny_bigram_lm_deterministic");

    const { output } = await runSelectedExample(page, "nn_tiny_bigram_lm_deterministic", 120000);
    expect(output).not.toMatch(/nan|NaN|inf|Infinity|Traceback|ERROR/);

    const actual = parseJsonOutput<DeterministicParityOutput>(output);
    const expected = reference.output!;

    expect(actual.generated_ids).toEqual(expected.generated_ids);
    expect(actual.generated_text).toBe(expected.generated_text);
    expect(actual.loss_before).toBeCloseTo(expected.loss_before, 4);
    expect(actual.loss_after).toBeCloseTo(expected.loss_after, 4);
    expect(actual.last_logits).toHaveLength(expected.last_logits.length);
    for (let i = 0; i < expected.last_logits.length; i += 1) {
      expect(actual.last_logits[i]).toBeCloseTo(expected.last_logits[i]!, 4);
    }
    expect(consoleFailures).toEqual([]);
  });

  test("autoregressive argmax generation extends token context", async () => {
    consoleFailures.length = 0;
    await page.locator("#example-select").selectOption("nn_autoregressive_argmax_generation");
    await expect(page.locator("#example-select")).toHaveValue("nn_autoregressive_argmax_generation");

    const { output } = await runSelectedExample(page, "nn_autoregressive_argmax_generation");

    expect(output).not.toMatch(/nan|NaN|inf|Infinity|Traceback|ERROR/);
    const generated = output.match(/generated:\s*(\[\[.*\]\])/);
    expect(generated).not.toBeNull();
    const tokens = JSON.parse(generated?.[1] ?? "[]") as number[][];
    expect(tokens).toHaveLength(1);
    expect(tokens[0].slice(0, 3)).toEqual([5, 11, 7]);
    expect(tokens[0].length).toBeGreaterThanOrEqual(4);
    expect(tokens[0].length).toBeLessThanOrEqual(8);
    for (const token of tokens[0]) {
      expect(Number.isInteger(token)).toBe(true);
      expect(token).toBeGreaterThanOrEqual(0);
      expect(token).toBeLessThan(32);
    }
    expect(consoleFailures).toEqual([]);
  });

  test("adamw token classifier trains with inferred view shape", async () => {
    consoleFailures.length = 0;
    await page.locator("#example-select").selectOption("nn_adamw_token_classifier");
    await expect(page.locator("#example-select")).toHaveValue("nn_adamw_token_classifier");

    const { output } = await runSelectedExample(page, "nn_adamw_token_classifier");

    expect(output).not.toMatch(/nan|NaN|inf|Infinity|Traceback|ERROR/);
    const losses = [...output.matchAll(/step=(\d+) loss=([0-9.]+)/g)].map((match) => ({
      step: Number(match[1]),
      loss: Number(match[2]),
    }));
    expect(losses).toHaveLength(4);
    expect(losses.map((entry) => entry.step)).toEqual([0, 1, 2, 3]);
    expect(losses[3].loss).toBeLessThan(losses[0].loss);
    const pred = output.match(/next_token_pred:\s*([0-9.]+)/);
    expect(pred).not.toBeNull();
    const token = Number(pred?.[1]);
    expect(Number.isInteger(token)).toBe(true);
    expect(token).toBeGreaterThanOrEqual(0);
    expect(token).toBeLessThan(8);
    expect(consoleFailures).toEqual([]);
  });

  test("cross entropy backward prints gradient values", async () => {
    consoleFailures.length = 0;
    await page.locator("#example-select").selectOption("autograd_cross_entropy_grad_repr");
    await expect(page.locator("#example-select")).toHaveValue("autograd_cross_entropy_grad_repr");

    const { output } = await runSelectedExample(page, "autograd_cross_entropy_grad_repr");

    expect(output).toMatch(/loss:\s*0\.24131/);
    expect(output).toMatch(/grad:\s*tensor\(\[\[-0\.2144, 0\.1753, 0\.0391\]\]\)/);
    expect(output).not.toContain("Tensor(_id=");
    expect(consoleFailures).toEqual([]);
  });

  test("all playground examples run without Python or WebGPU errors", async () => {
    test.setTimeout(30 * 60 * 1000);
    const failures: string[] = [];
    const timings: Array<{ id: string; label: string; elapsedMs: number }> = [];

    for (const [index, example] of examples.entries()) {
      consoleFailures.length = 0;
      console.log(`[browser examples] ${index + 1}/${examples.length}: ${example.id}`);
      await page.locator("#example-select").selectOption(example.id);
      await expect(page.locator("#example-select")).toHaveValue(example.id);

      let runResult: { output: string; elapsedMs: number };
      try {
        runResult = await runSelectedExample(page, example.id);
      } catch (error) {
        failures.push(`${example.id} (${example.label})\n${String(error)}`);
        break;
      }
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

  test("dtype aliases behave like PyTorch names in browser runtime", async () => {
    await page.locator("#example-select").selectOption("dtype_aliases");
    await expect(page.locator("#example-select")).toHaveValue("dtype_aliases");
    await runSelectedExample(page, "dtype_aliases");
    await page.waitForFunction(() => {
      const output = document.getElementById("output")?.textContent ?? "";
      return output.trim().startsWith("{");
    });
    const outputText = await page.locator("#output").innerText();
    const output = JSON.parse(outputText) as Record<string, unknown>;

    expect(output.long_alias).toBe("int64");
    expect(output.bool_alias).toBe("bool");
    expect(output.double_alias).toBe("float64");
    expect(output.half_alias).toBe("float16");
    expect(output.int_alias).toBe("int32");
    expect(output.float_alias).toBe("float32");
    expect(output.short_alias).toBe("int16");
    expect(output.char_alias).toBe("int8");
    expect(output.byte_alias).toBe("uint8");

    expect(output.idx_dtype).toBe("int64");
    expect(output.roundtrip_dtype).toBe("int64");
    expect(output.mask_dtype).toBe("bool");
    expect(output.selected).toEqual([1, 3]);
  });
});
