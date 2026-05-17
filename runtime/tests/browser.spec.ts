import { expect, test } from "@playwright/test";

test("mvp demo runs tensor + matmul + reductions in pyodide @webgpu", async ({ page }) => {
  await page.goto("/demo/index.html");

  await page.waitForFunction(() => Boolean((window as any).__torchMvpStatus), null, {
    timeout: 120000
  });

  const status = await page.evaluate(() => (window as any).__torchMvpStatus);
  expect(status.ok).toBe(true);
  expect(status.result.ok).toBe(true);
  expect(status.result.sum).toBe(28);
  expect(status.result.mean).toBe(7);
  expect(["published", "local-dev"]).toContain(status.installMode);
});

test("demo falls back to local-dev when published install is forced to fail", async ({ page }) => {
  await page.goto("/demo/index.html?force_fallback=1");

  await page.waitForFunction(() => Boolean((window as any).__torchMvpStatus), null, {
    timeout: 120000
  });

  const status = await page.evaluate(() => (window as any).__torchMvpStatus);
  expect(status.installMode).toBe("local-dev");
  expect(String(status.installDetail)).toContain("Forced published install failure");
  if (!status.ok) {
    expect(String(status.error)).toContain("Failed to request WebGPU adapter");
  }
});

test("playground runs default code and prints json output", async ({ page }) => {
  await page.goto("/playground/index.html");
  await page.waitForSelector("#run:not([disabled])", { timeout: 120000 });
  await page.click("#run");
  await page.waitForFunction(() => {
    const output = document.querySelector("#output");
    if (!output || !output.textContent) {
      return false;
    }
    return output.textContent.includes('"shape"') || output.textContent.includes("Failed to request WebGPU adapter");
  });

  const outputText = await page.locator("#output").innerText();
  if (outputText.includes("Failed to request WebGPU adapter")) {
    expect(outputText).toContain("Failed to request WebGPU adapter");
  } else {
    expect(outputText).toContain('"shape"');
    expect(outputText).toContain('"values"');
  }
  expect(outputText).not.toContain("dict' object has no attribute 'id'");
});

test("runtime returns explicit error when webgpu is unavailable", async ({ page }) => {
  await page.goto("/demo/index.html");
  const error = await page.evaluate(async () => {
    const mod = await import("/src/runtime.ts");
    const runtime = new mod.TorchPyodideRuntime();
    try {
      await runtime.init(null);
      return "";
    } catch (err) {
      return String(err);
    }
  });
  expect(error).toContain("WebGPU unavailable");
});

test("run_sync entrypoint error is explicit when executed from synchronous runPython", async ({ page }) => {
  await page.goto("/demo/index.html");
  await page.waitForFunction(() => Boolean((window as any).__torchMvpStatus), null, {
    timeout: 120000
  });
  const status = await page.evaluate(() => (window as any).__torchMvpStatus);
  expect(String(status.runSyncError)).toContain("Cannot stack switch");
});
