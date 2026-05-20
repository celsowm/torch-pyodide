import { expect, test, Page } from "@playwright/test";

const examples: { id: string; code: string }[] = [
  { id: "tensor_basics", code: `import json\nimport torch\n\na = torch.tensor([[1.0, 2.0], [3.0, 4.0]])\nb = torch.ones((2, 2))\nc = a.add(b).mul(torch.tensor([[2.0, 2.0], [2.0, 2.0]]))\nout = {\n  "shape": list(c.shape),\n  "values": c.tolist(),\n  "sum": c.sum().tolist(),\n  "mean": c.mean().tolist(),\n  "cuda_available": torch.cuda.is_available(),\n  "cuda_device_count": torch.cuda.device_count(),\n}\njson.dumps(out)` },
  { id: "matmul_relu", code: `import json\nimport torch\n\nx = torch.tensor([[1.0, -2.0], [3.0, -4.0]])\nw = torch.tensor([[0.5, 1.0], [1.5, -1.0]])\ny = x.matmul(w).relu()\nout = {\n  "shape": list(y.shape),\n  "values": y.tolist(),\n  "sum": y.sum().tolist(),\n}\njson.dumps(out)` },
  { id: "reshape_transpose", code: `import json\nimport torch\n\na = torch.tensor([[1.0, 2.0], [3.0, 4.0]])\nt = a.T\nr = t.reshape((4,))\nout = {\n  "transpose": t.tolist(),\n  "reshape": r.tolist(),\n}\njson.dumps(out)` },
  { id: "rand_tensor", code: `import json\nimport torch\n\nx = torch.rand((2, 3))\nout = {\n  "shape": list(x.shape),\n  "values": x.tolist(),\n}\njson.dumps(out)` },
  { id: "clamp_values", code: `import json\nimport torch\n\nx = torch.tensor([[-1.0, 0.2, 0.8], [1.5, 2.0, -0.3]])\ny = torch.clamp(x, 0.0, 1.0)\nout = {\n  "input": x.tolist(),\n  "clamped": y.tolist(),\n}\njson.dumps(out)` },
  { id: "where_select", code: `import json\nimport torch\n\ncond = torch.tensor([[True, False], [False, True]])\nx = torch.tensor([[10.0, 20.0], [30.0, 40.0]])\ny = torch.tensor([[1.0, 2.0], [3.0, 4.0]])\nout_tensor = torch.where(cond, x, y)\nout = {\n  "condition": cond.tolist(),\n  "x": x.tolist(),\n  "y": y.tolist(),\n  "where": out_tensor.tolist(),\n}\njson.dumps(out)` },
  { id: "argmax_argmin", code: `import json\nimport torch\n\nx = torch.tensor([[1.0, 3.0, 2.0], [9.0, 0.5, 7.0]])\nmax_idx = torch.argmax(x)\nmin_idx = torch.argmin(x)\nout = {\n  "input": x.tolist(),\n  "argmax": max_idx.tolist(),\n  "argmin": min_idx.tolist(),\n}\njson.dumps(out)` },
  { id: "randn_tensor", code: `import json\nimport torch\n\nx = torch.randn((2, 4))\nout = {\n  "shape": list(x.shape),\n  "values": x.tolist(),\n}\njson.dumps(out)` },
  { id: "arange_int32", code: `import json\nimport torch\n\nx = torch.arange(1, 10, 2, dtype=torch.int32)\nout = {\n  "dtype": str(x.dtype),\n  "values": x.tolist(),\n}\njson.dumps(out)` },
  { id: "full_and_full_like", code: `import json\nimport torch\n\nbase = torch.tensor([[1.0, 2.0], [3.0, 4.0]])\na = torch.full((2, 2), 7.0, dtype=torch.int32)\nb = torch.full_like(base, 9.0)\nout = {\n  "full": a.tolist(),\n  "full_dtype": str(a.dtype),\n  "full_like": b.tolist(),\n  "full_like_dtype": str(b.dtype),\n}\njson.dumps(out)` },
  { id: "unary_abs_neg", code: `import json\nimport torch\n\nx = torch.tensor([[-1.0, 2.0], [-3.0, 4.0]])\nout = {\n  "input": x.tolist(),\n  "abs": torch.abs(x).tolist(),\n  "neg": torch.neg(x).tolist(),\n}\njson.dumps(out)` },
  { id: "unary_sqrt_exp_log", code: `import json\nimport torch\n\nx = torch.tensor([[1.0, 4.0], [9.0, 16.0]])\ny = torch.tensor([[0.0, 1.0], [2.0, 0.0]])\nz = torch.tensor([[1.0, 2.718281828], [7.389056099, 1.0]])\nout = {\n  "sqrt": torch.sqrt(x).tolist(),\n  "exp": torch.exp(y).tolist(),\n  "log": torch.log(z).tolist(),\n}\njson.dumps(out)` },
  { id: "shape_flatten_squeeze", code: `import json\nimport torch\n\nx = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])\nout = {\n  "input_shape": list(x.shape),\n  "flatten_1_2": torch.flatten(x, 1, 2).tolist(),\n  "squeeze_all_shape": list(torch.squeeze(x).shape),\n  "unsqueeze_dim0_shape": list(torch.unsqueeze(torch.tensor([1.0, 2.0]), 0).shape),\n}\njson.dumps(out)` },
  { id: "shape_transpose_permute", code: `import json\nimport torch\n\nx = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])\nt = torch.transpose(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), 0, 1)\np = torch.permute(x, (2, 0, 1))\nout = {\n  "transpose": t.tolist(),\n  "permute_shape": list(p.shape),\n}\njson.dumps(out)` },
  { id: "index_select_slice", code: `import json\nimport torch\n\nx = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])\nout = {\n  "select_row_1": torch.select(x, 0, 1).tolist(),\n  "slice_rows_0_3_step2": x[0:3:2].tolist(),\n  "getitem_row_1": x[1].tolist(),\n}\njson.dumps(out)` },
  { id: "cat_stack", code: `import json\nimport torch\n\na = torch.tensor([[1.0, 2.0], [3.0, 4.0]])\nb = torch.tensor([[5.0, 6.0], [7.0, 8.0]])\ncat_rows = torch.cat([a, b], dim=0)\ncat_cols = torch.cat([a, b], dim=1)\nstacked = torch.stack([a, b], dim=0)\nout = {\n  "cat_dim0": cat_rows.tolist(),\n  "cat_dim0_shape": list(cat_rows.shape),\n  "cat_dim1": cat_cols.tolist(),\n  "cat_dim1_shape": list(cat_cols.shape),\n  "stack_dim0": stacked.tolist(),\n  "stack_dim0_shape": list(stacked.shape),\n}\njson.dumps(out)` },
  { id: "expand_index_select", code: `import json\nimport torch\n\nx = torch.tensor([[1.0, 2.0], [3.0, 4.0]])\ne = x.expand(3, 2, 2)\nindices = torch.tensor([0, 1], dtype=torch.int32)\ns = torch.index_select(x, 0, indices)\nout = {\n  "input": x.tolist(),\n  "expand_shape": list(e.shape),\n  "expand": e.tolist(),\n  "index_select": s.tolist(),\n}\njson.dumps(out)` },
  { id: "broadcasting", code: `import json\nimport torch\n\na = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\nb = torch.tensor([10.0, 20.0, 30.0])\nout = {\n  "a + b (broadcast)": a.add(b).tolist(),\n  "a * 2.0 (scalar broadcast)": a.mul(2.0).tolist(),\n  "shape_a": list(a.shape),\n  "shape_b": list(b.shape),\n}\njson.dumps(out)` },
  { id: "compare_ops", code: `import json\nimport torch\n\na = torch.tensor([[1.0, 5.0], [3.0, 2.0]])\nb = torch.tensor([[1.0, 3.0], [3.0, 4.0]])\nout = {\n  "a": a.tolist(),\n  "b": b.tolist(),\n  "eq": torch.eq(a, b).tolist(),\n  "ne": torch.ne(a, b).tolist(),\n  "lt": torch.lt(a, b).tolist(),\n  "le": torch.le(a, b).tolist(),\n  "gt": torch.gt(a, b).tolist(),\n  "ge": torch.ge(a, b).tolist(),\n}\njson.dumps(out)` },
  { id: "unary_advanced", code: `import json\nimport torch\nimport torch.nn.functional as F\n\nx = torch.tensor([[-1.0, 0.5, 1.0], [2.0, -2.0, 0.5]])\nout = {\n  "input": x.tolist(),\n  "sigmoid": torch.sigmoid(x).tolist(),\n  "tanh": torch.tanh(x).tolist(),\n  "sin": torch.sin(x).tolist(),\n  "cos": torch.cos(x).tolist(),\n  "gelu": F.gelu(x).tolist(),\n  "silu": F.silu(x).tolist(),\n  "leaky_relu": F.leaky_relu(x, 0.01).tolist(),\n  "floor": torch.floor(x).tolist(),\n  "ceil": torch.ceil(x).tolist(),\n  "round": torch.round(x).tolist(),\n  "reciprocal": torch.reciprocal(x).tolist(),\n  "square": torch.square(x).tolist(),\n}\njson.dumps(out)` },
  { id: "reduce_dim", code: `import json\nimport torch\n\nx = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\nout = {\n  "input": x.tolist(),\n  "sum(0)": torch.sum(x, 0).tolist(),\n  "sum(1)": torch.sum(x, 1).tolist(),\n  "mean(0)": torch.mean(x, 0).tolist(),\n  "mean(1)": torch.mean(x, 1).tolist(),\n  "prod": torch.prod(x).tolist(),\n  "min": torch.min(x).tolist(),\n  "max": torch.max(x).tolist(),\n}\njson.dumps(out)` },
  { id: "masked_select_fill", code: `import json\nimport torch\n\nx = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\nmask = torch.tensor([[True, False, True], [False, True, False]])\nfilled = torch.masked_fill(x, mask, 99.0)\nselected = torch.masked_select(x, mask)\nout = {\n  "input": x.tolist(),\n  "mask": mask.tolist(),\n  "masked_fill": filled.tolist(),\n  "masked_select": selected.tolist(),\n}\njson.dumps(out)` },
  { id: "broadcast_compare", code: `import json\nimport torch\n\na = torch.tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])\nthreshold = torch.tensor([3.0, 3.0, 3.0])\nout = {\n  "a": a.tolist(),\n  "threshold": threshold.tolist(),\n  "a > threshold": torch.gt(a, threshold).tolist(),\n  "a <= threshold": torch.le(a, threshold).tolist(),\n  "where(a > 3, a, 0)": torch.where(torch.gt(a, torch.tensor(3.0)), a, torch.zeros_like(a)).tolist(),\n}\njson.dumps(out)` },
  { id: "reduce_dim_keepdim", code: `import json\nimport torch\n\nx = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\nout = {\n  "input_shape": list(x.shape),\n  "sum(0, keepdim)": torch.sum(x, 0, True).tolist(),\n  "sum(0, keepdim)_shape": list(torch.sum(x, 0, True).shape),\n  "mean(1, keepdim)": torch.mean(x, 1, True).tolist(),\n  "mean(1, keepdim)_shape": list(torch.mean(x, 1, True).shape),\n}\njson.dumps(out)` },
  { id: "nn_linear_relu", code: `import json\nimport torch\nimport torch.nn as nn\n\nmodel = nn.Sequential(\n    nn.Linear(4, 8),\n    nn.ReLU(),\n    nn.Linear(8, 2),\n)\nx = torch.randn((3, 4))\ny = model(x)\nout = {"shape": list(y.shape), "values": y.tolist()}\njson.dumps(out)` },
  { id: "nn_conv2d", code: `import json\nimport torch\nimport torch.nn as nn\n\nconv = nn.Conv2d(1, 4, 3)\nx = torch.randn((2, 1, 6, 6))\ny = conv(x)\nout = {"shape": list(y.shape), "sum": y.sum().tolist()}\njson.dumps(out)` },
  { id: "nn_batchnorm", code: `import json\nimport torch\nimport torch.nn as nn\n\nbn = nn.BatchNorm1d(4)\nx = torch.randn((3, 4))\ny = bn(x)\nout = {"shape": list(y.shape), "values": y.tolist()}\njson.dumps(out)` },
  { id: "nn_pooling", code: `import json\nimport torch\nimport torch.nn as nn\n\nx = torch.randn((1, 2, 8, 8))\nmax_pool = nn.MaxPool2d(2)\navg_pool = nn.AvgPool2d(2)\nout = {\n    "max_pool_shape": list(max_pool(x).shape),\n    "avg_pool_shape": list(avg_pool(x).shape),\n}\njson.dumps(out)` },
  { id: "nn_losses", code: `import json\nimport torch\nimport torch.nn.functional as F\n\nlogits = torch.randn((2, 3))\ntarget = torch.tensor([0, 2])\nloss = F.cross_entropy(logits, target)\nmse = F.mse_loss(logits, torch.zeros((2, 3)))\nout = {\n    "cross_entropy": loss.tolist(),\n    "mse_loss": mse.tolist(),\n}\njson.dumps(out)` },
  { id: "nn_nll_loss", code: `import json\nimport torch\nimport torch.nn.functional as F\n\nlogits = torch.randn((2, 4))\ntarget = torch.tensor([1, 3])\nlog_probs = F.log_softmax(logits, dim=-1)\nnll = F.nll_loss(log_probs, target, reduction="mean")\nce = F.cross_entropy(logits, target, reduction="mean")\nnll_none = F.nll_loss(log_probs, target, reduction="none")\nout = {\n    "log_probs": [[round(v, 4) for v in row] for row in log_probs.tolist()],\n    "nll_loss_mean": round(nll.item(), 4),\n    "cross_entropy_mean": round(ce.item(), 4),\n    "nll_loss_none": [round(v, 4) for v in nll_none.tolist()],\n}\njson.dumps(out)` },
  { id: "nn_batchnorm_training", code: `import json\nimport torch\nimport torch.nn as nn\n\nbn = nn.BatchNorm1d(4)\nbn.weight = torch.ones((4,))\nbn.bias = torch.zeros((4,))\n\nx = torch.tensor([[1.0, 2.0, 3.0, 4.0],\n                  [5.0, 6.0, 7.0, 8.0],\n                  [9.0, 10.0, 11.0, 12.0]])\n\nbn.train()\ny_train = bn(x)\n\nbn.eval()\ny_eval = bn(x)\n\nrunning_mean = bn.running_mean.tolist()\nrunning_var = bn.running_var.tolist()\n\nout = {\n    "x_shape": list(x.shape),\n    "output_train": [[round(v, 4) for v in row] for row in y_train.tolist()],\n    "output_eval": [[round(v, 4) for v in row] for row in y_eval.tolist()],\n    "running_mean": [round(v, 4) for v in running_mean],\n    "running_var": [round(v, 4) for v in running_var],\n}\njson.dumps(out)` },
  { id: "autograd_select", code: `import json\nimport torch\n\ntorch.manual_seed(42)\nx = torch.randn((3, 4), requires_grad=True)\nrow = x.select(0, 1)\nloss = row.abs().sum()\nloss.backward()\nout = {\n    "x_shape": list(x.shape),\n    "row_shape": list(row.shape),\n    "loss": loss.tolist(),\n    "x.grad_nonzero": bool((x.grad != 0).any()) if x.grad is not None else False,\n    "x.grad_shape": list(x.grad.shape) if x.grad is not None else None,\n}\njson.dumps(out)` },
  { id: "autograd_index_select", code: `import json\nimport torch\n\ntorch.manual_seed(42)\nx = torch.randn((5, 3), requires_grad=True)\nidx = torch.tensor([0, 2, 4], dtype=torch.int32)\nselected = torch.index_select(x, 0, idx)\nloss = selected.sum()\nloss.backward()\nout = {\n    "x_shape": list(x.shape),\n    "selected_shape": list(selected.shape),\n    "loss": loss.tolist(),\n    "x.grad_nonzero": bool((x.grad != 0).any()) if x.grad is not None else False,\n    "x.grad_shape": list(x.grad.shape) if x.grad is not None else None,\n}\njson.dumps(out)` },
  { id: "autograd_masked_select", code: `import json\nimport torch\n\ntorch.manual_seed(42)\nx = torch.randn((3, 4), requires_grad=True)\nmask = torch.tensor([[True, False, True, False], [False, True, False, True], [True, True, False, False]])\nselected = torch.masked_select(x, mask)\nloss = selected.sum()\nloss.backward()\nout = {\n    "x_shape": list(x.shape),\n    "selected_shape": list(selected.shape),\n    "loss": loss.tolist(),\n    "x.grad_nonzero": bool((x.grad != 0).any()) if x.grad is not None else False,\n    "x.grad_shape": list(x.grad.shape) if x.grad is not None else None,\n}\njson.dumps(out)` },
  { id: "autograd_max", code: `import json\nimport torch\n\ntorch.manual_seed(42)\nx = torch.randn((2, 5), requires_grad=True)\ny = x.max()\ny.backward()\nout = {\n    "x_shape": list(x.shape),\n    "max_val": y.tolist(),\n    "x.grad_nonzero": bool((x.grad != 0).any()) if x.grad is not None else False,\n    "x.grad_shape": list(x.grad.shape) if x.grad is not None else None,\n}\njson.dumps(out)` },
  { id: "autograd_min", code: `import json\nimport torch\n\ntorch.manual_seed(42)\nx = torch.randn((2, 5), requires_grad=True)\ny = x.min()\ny.backward()\nout = {\n    "x_shape": list(x.shape),\n    "min_val": y.tolist(),\n    "x.grad_nonzero": bool((x.grad != 0).any()) if x.grad is not None else False,\n    "x.grad_shape": list(x.grad.shape) if x.grad is not None else None,\n}\njson.dumps(out)` },
  { id: "gather", code: `import json\nimport torch\n\nx = torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])\nidx = torch.tensor([[0, 1], [2, 0]])\nout = {\n    "input_shape": list(x.shape),\n    "gather_dim0": torch.gather(x, 0, idx).tolist(),\n    "gather_dim1": torch.gather(x, 1, idx).tolist(),\n    "indices_shape": list(idx.shape),\n}\njson.dumps(out)` },
  { id: "sort", code: `import json\nimport torch\n\nx = torch.tensor([[3.0, 1.0, 4.0], [1.0, 5.0, 9.0]])\nv_asc, i_asc = torch.sort(x, dim=1, descending=False)\nv_desc, i_desc = torch.sort(x, dim=1, descending=True)\nout = {\n    "input": x.tolist(),\n    "sort_asc_values": v_asc.tolist(),\n    "sort_asc_indices": i_asc.tolist(),\n    "sort_desc_values": v_desc.tolist(),\n    "sort_desc_indices": i_desc.tolist(),\n}\njson.dumps(out)` },
  { id: "topk", code: `import json\nimport torch\n\nx = torch.tensor([[3.0, 1.0, 4.0, 1.0, 5.0, 9.0], [2.0, 7.0, 1.0, 8.0, 2.0, 8.0]])\nv, i = torch.topk(x, 3, dim=1, largest=True)\nout = {\n    "input": x.tolist(),\n    "top3_values": v.tolist(),\n    "top3_indices": i.tolist(),\n}\njson.dumps(out)` },
  { id: "optim_sgd", code: `import json\nimport torch\nimport torch.nn as nn\nfrom torch.optim import SGD\n\nmodel = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))\nx = torch.randn((3, 4))\ntarget = torch.randn((3, 2))\n\noptimizer = SGD(model.parameters(), lr=0.01)\nloss_fn = nn.MSELoss()\n\nlosses = []\nfor step in range(3):\n    optimizer.zero_grad()\n    out = model(x)\n    loss = loss_fn(out, target)\n    loss.backward()\n    optimizer.step()\n    losses.append(round(loss.item(), 4))\nout = {"losses": losses}\njson.dumps(out)` },
  { id: "optim_adam", code: `import json\nimport torch\nimport torch.nn as nn\nfrom torch.optim import Adam\n\nmodel = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))\nx = torch.randn((3, 4))\ntarget = torch.randn((3, 2))\n\noptimizer = Adam(model.parameters(), lr=0.001)\nloss_fn = nn.MSELoss()\n\nlosses = []\nfor step in range(3):\n    optimizer.zero_grad()\n    out = model(x)\n    loss = loss_fn(out, target)\n    loss.backward()\n    optimizer.step()\n    losses.append(round(loss.item(), 4))\nout = {"losses": losses}\njson.dumps(out)` },
];

test.describe.serial("playground examples @webgpu", () => {
  let page: Page;

  test.beforeAll(async ({ browser }) => {
    page = await browser.newPage();
    await page.goto("/demo/index.html?force_fallback=1", { timeout: 180000 });
    await page.waitForFunction(() => Boolean((window as any).__torchMvpStatus), { timeout: 180000 });
    const status = await page.evaluate(() => (window as any).__torchMvpStatus);
    console.log("[test] __torchMvpStatus:", JSON.stringify(status));
    if (!status.ok) {
      throw new Error("WebGPU bootstrap failed: " + (status.error || "unknown"));
    }
  });

  test.afterAll(async () => {
    await page?.close();
  });

  for (const ex of examples) {
    test(ex.id, async () => {
      test.setTimeout(120000);
      const outputText = await page.evaluate(async (code) => {
        const pyodide = (window as any).__pyodide;
        try {
          return await pyodide.runPythonAsync(code);
        } catch (e) {
          return "[ERROR] " + String(e);
        }
      }, ex.code);

      expect(outputText).not.toContain("[ERROR]");
      expect(outputText).not.toContain("Traceback");
      expect(outputText).not.toContain("PythonError");
      expect(outputText).not.toContain("Failed to get WebGPU adapter");
      const parsed = JSON.parse(
        outputText
          .replace(/\bNaN\b/g, "null")
          .replace(/\bInfinity\b/g, "null")
          .replace(/\b-Infinity\b/g, "null")
      );
      expect(parsed).toBeTruthy();
    });
  }
});
