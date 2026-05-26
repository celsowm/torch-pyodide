#!/usr/bin/env node
import { chromium } from "playwright";

const baseUrl = process.env.MANIFEST_BASE_URL ?? "http://127.0.0.1:4180";
const pyodideIndexUrl = process.env.PYODIDE_INDEX_URL ?? "https://cdn.jsdelivr.net/pyodide/v0.29.4/full/";
const forceRuntimeBaseUrl = process.env.MANIFEST_FORCE_RUNTIME_BASE_URL ?? "";

async function waitForEndpoint(url, maxAttempts = 20, delayMs = 500) {
  let lastError = null;
  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      const response = await fetch(url, { method: "GET" });
      if (response.ok) {
        return;
      }
      lastError = new Error(`HTTP ${response.status}`);
    } catch (error) {
      lastError = error;
    }
    await new Promise((resolveSleep) => setTimeout(resolveSleep, delayMs));
  }
  throw new Error(`Failed to reach ${url}: ${String(lastError)}`);
}

const browser = await chromium.launch({
  channel: "chromium",
  headless: true,
  args: [
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan,UseSkiaRenderer",
  ],
});

try {
  await waitForEndpoint(`${baseUrl}/latest.json`);
  const page = await browser.newPage();
  page.setDefaultTimeout(240000);
  await page.goto(`${baseUrl}/`, { waitUntil: "load" });

  const smokeResult = await page.evaluate(
    async ({ baseUrlValue, pyodideIndexUrlValue, forceRuntimeBaseUrlValue }) => {
      const manifestResponse = await fetch(`${baseUrlValue}/latest.json`, { cache: "no-store" });
      if (!manifestResponse.ok) {
        throw new Error(`Failed to load latest.json (HTTP ${manifestResponse.status})`);
      }
      const manifest = await manifestResponse.json();
      if (forceRuntimeBaseUrlValue) {
        manifest.runtimeUrl = `${forceRuntimeBaseUrlValue.replace(/\/+$/, "")}/runtime/${manifest.torchVersion}/runtime.mjs`;
      }

      const runtimeModule = await import(/* @vite-ignore */ manifest.runtimeUrl);
      runtimeModule.installTorchRuntime(globalThis);

      const pyodideModule = await import(/* @vite-ignore */ `${pyodideIndexUrlValue}pyodide.mjs`);
      const pyodide = await pyodideModule.loadPyodide({ indexURL: pyodideIndexUrlValue });

      await pyodide.loadPackage("micropip");
      await pyodide.runPythonAsync(`
import micropip
await micropip.install("${manifest.wheelUrl}", reinstall=True)
`);

      const value = await pyodide.runPythonAsync(`
import torch
x = torch.tensor([1.0, 2.0, 3.0])
float(x.sum().item())
`);

      return {
        torchVersion: manifest.torchVersion,
        sumValue: Number(value),
      };
    },
    {
      baseUrlValue: baseUrl,
      pyodideIndexUrlValue: pyodideIndexUrl,
      forceRuntimeBaseUrlValue: forceRuntimeBaseUrl,
    },
  );

  if (smokeResult.sumValue !== 6) {
    throw new Error(`Unexpected tensor sum from smoke test: ${smokeResult.sumValue}`);
  }
  console.log(`[smoke_manifest_client] OK torch=${smokeResult.torchVersion} sum=${smokeResult.sumValue}`);
} finally {
  await browser.close();
}
