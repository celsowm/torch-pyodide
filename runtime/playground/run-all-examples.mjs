import { chromium } from "playwright";
import { spawn } from "node:child_process";
import path from "node:path";
import fs from "node:fs";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const runtimeDir = path.resolve(__dirname, "..");
const screenshotDir = path.resolve(__dirname, "screenshots");
const resultsFile = path.resolve(__dirname, "example-results.json");

fs.mkdirSync(screenshotDir, { recursive: true });

const examples = [
  { id: "tensor_basics", label: "Tensor Basics + CUDA Info" },
  { id: "matmul_relu", label: "Matmul + ReLU" },
  { id: "reshape_transpose", label: "Reshape + Transpose" },
  { id: "rand_tensor", label: "Random Tensor (rand)" },
  { id: "clamp_values", label: "Clamp Values" },
  { id: "where_select", label: "Where Selection" },
  { id: "argmax_argmin", label: "Argmax + Argmin" },
  { id: "randn_tensor", label: "Random Normal (randn)" },
  { id: "arange_int32", label: "Arange Int32" },
  { id: "full_and_full_like", label: "Full + Full Like" },
  { id: "unary_abs_neg", label: "Unary Abs + Neg" },
  { id: "unary_sqrt_exp_log", label: "Unary Sqrt + Exp + Log" },
  { id: "shape_flatten_squeeze", label: "Shape Flatten + Squeeze" },
  { id: "shape_transpose_permute", label: "Shape Transpose + Permute" },
  { id: "index_select_slice", label: "Index Select + Slice" },
  { id: "cat_stack", label: "Cat + Stack" },
  { id: "expand_index_select", label: "Expand + Index Select" },
  { id: "broadcasting", label: "Broadcasting" },
  { id: "compare_ops", label: "Compare Ops" },
  { id: "unary_advanced", label: "Unary Advanced" },
  { id: "reduce_dim", label: "Reduction by dim" },
  { id: "masked_select_fill", label: "Masked Select + Fill" },
  { id: "broadcast_compare", label: "Broadcast + Compare" },
  { id: "reduce_dim_keepdim", label: "Reduction keepdim" },
  { id: "nn_linear_relu", label: "nn Linear + ReLU" },
  { id: "nn_conv2d", label: "nn Conv2d" },
  { id: "nn_batchnorm", label: "nn BatchNorm1d" },
  { id: "nn_pooling", label: "nn MaxPool2d + AvgPool2d" },
  { id: "nn_losses", label: "nn Losses" },
  { id: "nn_nll_loss", label: "NLL Loss + Cross Entropy" },
  { id: "nn_batchnorm_training", label: "BatchNorm Training vs Eval" },
  { id: "autograd_basic", label: "Autograd: Backprop" },
  { id: "autograd_conv2d", label: "Autograd: Conv2d Backward" },
  { id: "autograd_nll", label: "Autograd: NLL Loss Backward" },
  { id: "autograd_slice", label: "Autograd: Slice Backward" },
  { id: "autograd_select", label: "Autograd: Select Backward" },
  { id: "autograd_index_select", label: "Autograd: Index Select Backward" },
  { id: "autograd_masked_select", label: "Autograd: Masked Select Backward" },
  { id: "autograd_max", label: "Autograd: Max Backward" },
  { id: "autograd_min", label: "Autograd: Min Backward" },
  { id: "training_linear", label: "Training: Linear Regression (SGD)" },
  { id: "training_classification", label: "Training: Classification (Adam)" },
  { id: "padding_modes", label: "Padding: reflect, replicate, circular" },
  { id: "einsum_multi", label: "Einsum: 3+ operands" },
  { id: "context_managers", label: "Context: no_grad, inference_mode" },
  { id: "distributions", label: "Distributions" },
  { id: "data_loader", label: "DataLoader: Samplers, ConcatDataset" },
  { id: "float16_bfloat16", label: "dtypes: float16, bfloat16" },
  { id: "autograd_activations", label: "Autograd: sigmoid, tanh, gelu..." },
  { id: "autograd_sort_topk", label: "Autograd: Sort + Topk" },
  { id: "autograd_maximum_minimum", label: "Autograd: max/min elementwise" },
  { id: "autograd_cumsum_cumprod", label: "Autograd: cumsum, cumprod, tril..." },
  { id: "autograd_cat_expand_where", label: "Autograd: cat, expand, where" },
];

function startDevServer() {
  return new Promise((resolve, reject) => {
    const proc = spawn(
      "npx",
      ["vite", "--host", "127.0.0.1", "--port", "4173"],
      {
        cwd: runtimeDir,
        stdio: ["ignore", "pipe", "pipe"],
        shell: true,
        windowsHide: false,
      }
    );

    let resolved = false;

    const timeout = setTimeout(() => {
      if (!resolved) {
        resolved = true;
        proc.kill("SIGTERM");
        reject(new Error("Server start timeout (60s)"));
      }
    }, 60000);

    proc.stdout.on("data", (data) => {
      const text = data.toString();
      const stripped = text.replace(/\x1B\[[0-9;]*[a-zA-Z]/g, "");
      process.stdout.write(`[vite] ${stripped.trim()}\n`);
      if (!resolved && stripped.includes("Local:")) {
        resolved = true;
        clearTimeout(timeout);
        // Give it a moment to fully start
        setTimeout(() => resolve(proc), 1000);
      }
    });

    proc.stderr.on("data", (data) => {
      const text = data.toString();
      process.stderr.write(`[vite:err] ${text}`);
    });

    proc.on("error", (err) => {
      if (!resolved) {
        resolved = true;
        clearTimeout(timeout);
        reject(err);
      }
    });

    proc.on("exit", (code) => {
      if (!resolved) {
        resolved = true;
        clearTimeout(timeout);
        reject(new Error(`Server exited with code ${code}`));
      }
    });
  });
}

function stripAnsi(s) {
  return s.replace(/\x1B\[[0-9;]*[a-zA-Z]/g, "");
}

async function run() {
  console.log("Starting Vite dev server...");
  const server = await startDevServer();
  console.log("Server started.");

  const results = [];

  console.log("Launching Chromium (headed)...");
  const browser = await chromium.launch({
    headless: false,
    args: ["--start-maximized"],
  });

  const context = await browser.newContext({
    viewport: { width: 1280, height: 900 },
    ignoreHTTPSErrors: true,
  });

  const page = await context.newPage();

  try {
    console.log("Navigating to playground...");
    await page.goto("http://127.0.0.1:4173/playground/", {
      waitUntil: "networkidle",
      timeout: 30000,
    });

    // Wait for Pyodide to boot
    console.log("Waiting for Pyodide + torch to initialize...");
    await page.waitForFunction(
      () => {
        const meta = document.getElementById("meta");
        return meta && meta.textContent && meta.textContent.startsWith("Ready");
      },
      { timeout: 300000 }
    );
    console.log("Pyodide ready!\n");

    for (const [index, example] of examples.entries()) {
      const exampleNum = index + 1;
      const total = examples.length
      const consoleMsgs = []; // per-example messages

      page.on("console", (msg) => {
        const text = msg.text();
        if (
          text.includes("[HMR]") ||
          text.includes("[vite]") ||
          text.includes("Source map") ||
          text.startsWith("torch version installed")
        ) {
          return;
        }
        consoleMsgs.push({ type: msg.type(), text: text.substring(0, 500) });
      });

      page.on("pageerror", (err) => {
        consoleMsgs.push({ type: "pageerror", text: err.message.substring(0, 500) });
      });

      console.log(`[${exampleNum}/${total}] ${example.label}`);

      try {
        // Select example
        await page.selectOption("#example-select", example.id);
        await page.waitForTimeout(800);

        // Click Run
        await page.click("#run");

        // Wait for output to appear
        await page.waitForFunction(
          () => {
            const el = document.getElementById("output");
            return el && el.textContent && el.textContent.length > 0;
          },
          { timeout: 600000 }
        );

        await page.waitForTimeout(1000);

        const outputText = await page.evaluate(() =>
          document.getElementById("output").textContent
        );

        const screenshotPath = path.join(screenshotDir, `${example.id}.png`);
        await page.screenshot({ path: screenshotPath, fullPage: false });

        const hasError = outputText.startsWith("ERROR") || outputText.includes("Traceback");
        const warnings = consoleMsgs.filter(
          (m) => m.type === "warning" || m.type === "error" || m.type === "pageerror"
        );

        results.push({
          id: example.id,
          label: example.label,
          success: !hasError,
          output: outputText.substring(0, 600),
          consoleMessages: consoleMsgs,
          screenshot: screenshotPath,
          status: hasError ? "ERROR" : "OK",
        });

        if (hasError) {
          console.log(`  ❌ ERROR`);
          console.log(`     ${stripAnsi(outputText).substring(0, 300)}`);
        } else {
          console.log(`  ✅ OK`);
        }
        for (const w of warnings) {
          console.log(`  ⚠ ${w.type}: ${stripAnsi(w.text).substring(0, 200)}`);
        }
      } catch (err) {
        const screenshotPath = path.join(screenshotDir, `${example.id}_error.png`);
        try {
          await page.screenshot({ path: screenshotPath, fullPage: false });
        } catch {}

        results.push({
          id: example.id,
          label: example.label,
          success: false,
          output: `Exception: ${err.message}`,
          consoleMessages: consoleMsgs,
          screenshot: screenshotPath,
          status: "EXCEPTION",
        });

        console.log(`  ❌ EXCEPTION: ${err.message.substring(0, 200)}`);
      }

      // Remove listeners for next iteration
      page.removeAllListeners("console");
      page.removeAllListeners("pageerror");
    }
  } finally {
    fs.writeFileSync(resultsFile, JSON.stringify(results, null, 2), "utf-8");
    console.log(`\nResults saved to ${resultsFile}`);
    await browser.close();
    server.kill("SIGTERM");
    printSummary(results);
  }
}

function printSummary(results) {
  const total = results.length;
  const passed = results.filter((r) => r.success).length;
  const failed = results.filter((r) => !r.success).length;

  console.log("\n" + "=".repeat(70));
  console.log("FINAL SUMMARY");
  console.log("=".repeat(70));
  console.log(`Total: ${total} | Passed: ${passed} | Failed: ${failed}\n`);

  if (failed > 0) {
    console.log("FAILED:");
    console.log("-".repeat(50));
    for (const r of results) {
      if (!r.success) {
        console.log(`❌ ${r.label} (${r.id}) — ${r.status}`);
        const short = stripAnsi(r.output).substring(0, 250);
        console.log(`   ${short}\n`);
      }
    }
  }

  console.log("PASSED:");
  console.log("-".repeat(50));
  for (const r of results) {
    if (r.success) {
      console.log(`✅ ${r.label} (${r.id})`);
    }
  }

  console.log("\nCONSOLE WARNINGS/ERRORS:");
  console.log("-".repeat(50));
  const allWarnings = results.flatMap((r) =>
    r.consoleMessages.filter(
      (m) => m.type === "warning" || m.type === "error" || m.type === "pageerror"
    )
  );
  if (allWarnings.length === 0) {
    console.log("(none)");
  } else {
    const seen = new Set();
    for (const w of allWarnings) {
      const key = `[${w.type}] ${w.text}`;
      if (!seen.has(key)) {
        seen.add(key);
        console.log(`  ⚠ ${key.substring(0, 300)}`);
      }
    }
  }
}

run().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
