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
  expect(typeof status.result.cuda_available).toBe("boolean");
  expect(typeof status.result.cuda_device_count).toBe("number");
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
    expect(String(status.error)).toContain("Failed to get WebGPU adapter");
  }
});

test("playground loads examples dropdown and runs selected code", async ({ page }) => {
  await page.goto("/playground/index.html");
  await page.waitForSelector("#run:not([disabled])", { timeout: 120000 });
  const optionCount = await page.locator("#example-select option").count();
  expect(optionCount).toBeGreaterThanOrEqual(7);
  await page.click("#run");
  await page.waitForFunction(() => {
    const output = document.querySelector("#output");
    if (!output || !output.textContent) {
      return false;
    }
    return (
      output.textContent.includes('"shape"') || output.textContent.includes("Failed to get WebGPU adapter")
    );
  });

  const outputText = await page.locator("#output").innerText();
  if (outputText.includes("Failed to get WebGPU adapter")) {
    expect(outputText).toContain("Failed to get WebGPU adapter");
  } else {
    expect(outputText).toContain('"shape"');
    expect(outputText).toContain('"values"');
  }
  expect(outputText).not.toContain("dict' object has no attribute 'id'");
});

test("playground switches example immediately and reset restores selected example", async ({ page }) => {
  await page.goto("/playground/index.html");
  await page.waitForSelector("#run:not([disabled])", { timeout: 120000 });

  const editor = page.locator(".cm-content");
  await page.selectOption("#example-select", "reshape_transpose");
  await expect(editor).toContainText("transpose");

  await editor.click();
  await page.keyboard.press("Control+a");
  await page.keyboard.type("print('edited')");
  await expect(editor).toContainText("edited");

  await page.click("#reset");
  await expect(editor).toContainText("reshape");
  await expect(editor).not.toContainText("edited");
});

test("playground shows explicit error when examples catalog fails to load", async ({ page }) => {
  await page.route(/.*examples.*\.json.*/, async (route) => {
    await route.fulfill({
      status: 500,
      contentType: "application/json",
      body: JSON.stringify({ error: "forced failure" })
    });
  });

  await page.goto("/playground/index.html");
  await page.waitForSelector("#meta");
  await expect(page.locator("#meta")).toContainText("Failed to initialize playground:");
  await expect(page.locator("#run")).toBeDisabled();
  await expect(page.locator("#reset")).toBeDisabled();
  await expect(page.locator("#example-select")).toBeDisabled();
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
