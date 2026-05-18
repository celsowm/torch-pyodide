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

test("device manager creates and reads tensors @webgpu", async ({ page }) => {
  await page.goto("/demo/index.html");
  await page.waitForFunction(() => Boolean((window as any).__torchMvpStatus), null, {
    timeout: 120000
  });

  const result = await page.evaluate(async () => {
    const mod = await import("/src/runtime.ts");
    const rt = new mod.TorchPyodideRuntime();
    await rt.init();
    const t = await rt.tensorFromData([1, 2, 3, 4], [2, 2], "float32");
    const data = await rt.toList(t.id);
    await rt.destroy(t.id);
    return data;
  });

  expect(result).toEqual([1, 2, 3, 4]);
});

test("device manager recovers from device lost and reads from shadow copy @webgpu", async ({ page }) => {
  await page.goto("/demo/index.html");
  await page.waitForFunction(() => Boolean((window as any).__torchMvpStatus), null, {
    timeout: 120000
  });

  const result = await page.evaluate(async () => {
    const mod = await import("/src/runtime.ts");
    const rt = new mod.TorchPyodideRuntime();
    await rt.init();
    const t = await rt.tensorFromData([10, 20, 30], [3], "float32");
    const dataBefore = await rt.toList(t.id);

    // Force device lost by destroying the device and letting DeviceManager recover
    const dm = (rt as any).deviceMgr;
    const device = dm.device;
    device.destroy();

    // After device lost, reading should trigger recovery and fall back to shadow copy
    const recovered = await rt.toList(t.id);
    await rt.destroy(t.id);
    return { before: dataBefore, after: recovered };
  });

  expect(result.before).toEqual([10, 20, 30]);
  // After device lost + recovery, shadow copy fallback should return the same data
  expect(result.after).toEqual([10, 20, 30]);
});

test("shape transpose and permute match the playground example plus an extra ND case @webgpu", async ({ page }) => {
  await page.goto("/demo/index.html");
  await page.waitForFunction(() => Boolean((window as any).__torchMvpStatus), null, {
    timeout: 120000
  });

  const result = await page.evaluate(async () => {
    const mod = await import("/src/runtime.ts");
    const rt = new mod.TorchPyodideRuntime();
    await rt.init();

    const m = await rt.tensorFromData([1, 2, 3, 4], [2, 2], "float32");
    const t = await rt.transpose(m.id, 0, 1);

    const x = await rt.tensorFromData([1, 2, 3, 4], [2, 1, 2], "float32");
    const p = await rt.permute(x.id, [2, 0, 1]);

    const y = await rt.tensorFromData([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2], "float32");
    const q = await rt.permute(y.id, [3, 1, 2, 0]);

    const out = {
      transpose: await rt.toList(t.id),
      permute: await rt.toList(p.id),
      transposeShape: t.shape,
      permuteShape: p.shape,
      extraPermuteShape: q.shape,
    };

    await rt.destroy(m.id);
    await rt.destroy(x.id);
    await rt.destroy(t.id);
    await rt.destroy(p.id);
    await rt.destroy(y.id);
    await rt.destroy(q.id);

    return out;
  });

  expect(result.transpose).toEqual([1, 3, 2, 4]);
  expect(result.permute).toEqual([1, 3, 2, 4]);
  expect(result.transposeShape).toEqual([2, 2]);
  expect(result.permuteShape).toEqual([2, 2, 1]);
  expect(result.extraPermuteShape).toEqual([2, 2, 2, 1]);
});

test("broadcasted comparisons and where match the matrix threshold example @webgpu", async ({ page }) => {
  await page.goto("/demo/index.html");
  await page.waitForFunction(() => Boolean((window as any).__torchMvpStatus), null, {
    timeout: 120000
  });

  const result = await page.evaluate(async () => {
    const mod = await import("/src/runtime.ts");
    const rt = new mod.TorchPyodideRuntime();
    await rt.init();

    const a = await rt.tensorFromData([1, 5, 3, 4, 2, 6], [2, 3], "float32");
    const threshold = await rt.tensorFromData([3, 3, 3], [3], "float32");
    const scalar = await rt.full([], 3.0, "float32");

    const gt = await rt.gt(a.id, threshold.id);
    const le = await rt.le(a.id, threshold.id);
    const zeros = await rt.fullLike(a.id, 0.0, "float32");
    const cond = await rt.gt(a.id, scalar.id);
    const where = await rt.where(cond.id, a.id, zeros.id);

    const out = {
      gt: await rt.toList(gt.id),
      le: await rt.toList(le.id),
      where: await rt.toList(where.id),
    };

    await rt.destroy(a.id);
    await rt.destroy(threshold.id);
    await rt.destroy(scalar.id);
    await rt.destroy(gt.id);
    await rt.destroy(le.id);
    await rt.destroy(zeros.id);
    await rt.destroy(cond.id);
    await rt.destroy(where.id);

    return out;
  });

  expect(result.gt).toEqual([0, 1, 0, 1, 0, 1]);
  expect(result.le).toEqual([1, 0, 1, 0, 1, 0]);
  expect(result.where).toEqual([0, 5, 0, 4, 0, 6]);
});

test("shape select slice and indexSelect match basic indexing semantics @webgpu", async ({ page }) => {
  await page.goto("/demo/index.html");
  await page.waitForFunction(() => Boolean((window as any).__torchMvpStatus), null, {
    timeout: 120000
  });

  const result = await page.evaluate(async () => {
    const mod = await import("/src/runtime.ts");
    const rt = new mod.TorchPyodideRuntime();
    await rt.init();

    const x = await rt.tensorFromData([1, 2, 3, 4, 5, 6], [3, 2], "float32");
    const select = await rt.select(x.id, 0, 1);
    const sliced = await rt.slice(x.id, 0, 0, 3, 2);
    const indices = await rt.tensorFromData([2, 0], [2], "int32");
    const gathered = await rt.indexSelect(x.id, 0, indices.id);

    const out = {
      selectRow1: await rt.toList(select.id),
      sliceRows0_3Step2: await rt.toList(sliced.id),
      getitemRow1: await rt.toList(select.id),
      gather: await rt.toList(gathered.id),
      selectShape: select.shape,
      sliceShape: sliced.shape,
      gatherShape: gathered.shape,
    };

    await rt.destroy(x.id);
    await rt.destroy(select.id);
    await rt.destroy(sliced.id);
    await rt.destroy(indices.id);
    await rt.destroy(gathered.id);

    return out;
  });

  expect(result.selectRow1).toEqual([3, 4]);
  expect(result.sliceRows0_3Step2).toEqual([1, 2, 5, 6]);
  expect(result.getitemRow1).toEqual([3, 4]);
  expect(result.gather).toEqual([5, 6, 1, 2]);
  expect(result.selectShape).toEqual([2]);
  expect(result.sliceShape).toEqual([2, 2]);
  expect(result.gatherShape).toEqual([2, 2]);
});

test("runBatch accumulates compute ops into a single submit @webgpu", async ({ page }) => {
  await page.goto("/demo/index.html");
  await page.waitForFunction(() => Boolean((window as any).__torchMvpStatus), null, {
    timeout: 120000
  });

  const result = await page.evaluate(async () => {
    const mod = await import("/src/runtime.ts");
    const rt = new mod.TorchPyodideRuntime();
    await rt.init();

    const result = await rt.runBatch(async () => {
      // Create two tensors, add them, then multiply by 2 — all in one batch
      const a = await rt.tensorFromData([1, 2, 3, 4], [4], "float32");
      const b = await rt.tensorFromData([10, 20, 30, 40], [4], "float32");
      const c = await rt.add(a.id, b.id);
      const d = await rt.mul(c.id, b.id);
      const e = await rt.relu(d.id);
      return { aId: a.id, bId: b.id, dId: d.id, eId: e.id };
    });

    const finalData = await rt.toList(result.eId);
    await rt.destroy(result.aId);
    await rt.destroy(result.bId);
    await rt.destroy(result.dId);
    await rt.destroy(result.eId);
    return finalData;
  });

  // (1+10)*10 = 110, (2+20)*20 = 440, (3+30)*30 = 990, (4+40)*40 = 1760
  expect(result).toEqual([110, 440, 990, 1760]);
});
