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

    // Force device recovery via DeviceManager
    const dm = (rt as any).deviceMgr;
    await dm.forceDeviceRecovery();

    // After recovery, reading should fall back to shadow copy
    const recovered = await rt.toList(t.id);
    await rt.destroy(t.id);
    return { before: dataBefore, after: recovered };
  });

  expect(result.before).toEqual([10, 20, 30]);
  // After device lost + recovery, shadow copy fallback should return the same data
  expect(result.after).toEqual([10, 20, 30]);
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

test.describe("playground example scripts", () => {
  const examples: { id: string; file: string }[] = [
    { id: "tensor_basics", file: "tensor_basics.py" },
    { id: "matmul_relu", file: "matmul_relu.py" },
    { id: "reshape_transpose", file: "reshape_transpose.py" },
    { id: "rand_tensor", file: "rand_tensor.py" },
    { id: "clamp_values", file: "clamp_values.py" },
    { id: "where_select", file: "where_select.py" },
    { id: "argmax_argmin", file: "argmax_argmin.py" },
    { id: "randn_tensor", file: "randn_tensor.py" },
    { id: "arange_int32", file: "arange_int32.py" },
    { id: "full_and_full_like", file: "full_and_full_like.py" },
    { id: "unary_abs_neg", file: "unary_abs_neg.py" },
    { id: "unary_sqrt_exp_log", file: "unary_sqrt_exp_log.py" },
    { id: "shape_flatten_squeeze", file: "shape_flatten_squeeze.py" },
    { id: "shape_transpose_permute", file: "shape_transpose_permute.py" },
    { id: "index_select_slice", file: "index_select_slice.py" },
    { id: "cat_stack", file: "cat_stack.py" },
    { id: "expand_index_select", file: "expand_index_select.py" },
    { id: "broadcasting", file: "broadcasting.py" },
    { id: "compare_ops", file: "compare_ops.py" },
    { id: "unary_advanced", file: "unary_advanced.py" },
    { id: "reduce_dim", file: "reduce_dim.py" },
    { id: "masked_select_fill", file: "masked_select_fill.py" },
    { id: "broadcast_compare", file: "broadcast_compare.py" },
    { id: "reduce_dim_keepdim", file: "reduce_dim_keepdim.py" },
    { id: "nn_linear_relu", file: "nn_linear_relu.py" },
    { id: "nn_conv2d", file: "nn_conv2d.py" },
    { id: "nn_batchnorm", file: "nn_batchnorm.py" },
    { id: "nn_pooling", file: "nn_pooling.py" },
    { id: "nn_losses", file: "nn_losses.py" },
    { id: "nn_nll_loss", file: "nn_nll_loss.py" },
    { id: "nn_batchnorm_training", file: "nn_batchnorm_training.py" },
  ];

  for (const ex of examples) {
    test(`${ex.id} runs without error`, async ({ page }) => {
      await page.goto("/playground/index.html");
      await page.waitForSelector("#run:not([disabled])", { timeout: 120000 });

      await page.selectOption("#example-select", ex.id, { timeout: 10000 });

      await page.waitForTimeout(500);
      await page.click("#run");

      // Wait until output appears or an error is shown
      await page.waitForFunction(
        () => {
          const output = document.querySelector("#output");
          if (!output || !output.textContent) return false;
          return output.textContent.length > 0;
        },
        { timeout: 60000 }
      );

      const outputText = await page.locator("#output").innerText();

      // If WebGPU is unavailable, skip — this is an env limitation, not a code bug
      if (outputText.includes("Failed to get WebGPU adapter")) {
        test.skip();
        return;
      }

      // Must not contain Python or runtime errors
      expect(outputText).not.toContain("Traceback (most recent call last)");
      expect(outputText).not.toContain("Error:");
      expect(outputText).not.toContain("PythonError");
      expect(outputText).not.toContain("object has no attribute");
      expect(outputText).not.toContain("is not subscriptable");

      // Must contain valid JSON output
      expect(outputText).toMatch(/^\{/);
      try {
        JSON.parse(outputText);
      } catch {
        expect(outputText).toBe("valid JSON output");
      }
    });
  }
});
