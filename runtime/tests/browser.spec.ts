import { expect, test } from "@playwright/test";

test("mvp demo runs tensor + matmul + reductions in pyodide", async ({ page }) => {
  await page.goto("/demo/index.html");

  await page.waitForFunction(() => Boolean((window as any).__torchMvpStatus), null, {
    timeout: 120000
  });

  const status = await page.evaluate(() => (window as any).__torchMvpStatus);
  if (status.ok) {
    expect(status.result.ok).toBe(true);
    expect(status.result.sum).toBe(28);
    expect(status.result.mean).toBe(7);
  } else {
    expect(String(status.error)).toContain("Failed to request WebGPU adapter");
  }
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
