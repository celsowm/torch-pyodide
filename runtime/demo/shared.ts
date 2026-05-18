import initPy from "../../python/torch/__init__.py?raw";
import cudaPy from "../../python/torch/cuda.py?raw";
import runtimePy from "../../python/torch/_runtime.py?raw";
import tensorPy from "../../python/torch/_tensor.py?raw";
import nnInitPy from "../../python/torch/nn/__init__.py?raw";
import nnModulesPy from "../../python/torch/nn/modules.py?raw";
import nnFunctionalPy from "../../python/torch/nn/functional.py?raw";
import { installTorchRuntime } from "../src";

type PyodideApi = {
  runPython: (code: string) => unknown;
  runPythonAsync: (code: string) => Promise<unknown>;
  loadPackage: (names: string | string[]) => Promise<unknown>;
  FS: {
    mkdirTree: (path: string) => void;
    writeFile: (path: string, data: string) => void;
  };
};

type LoadPyodideFn = (config: { indexURL: string }) => Promise<PyodideApi>;

async function resolvePyodideIndexURL(): Promise<string> {
  const hostname = globalThis.location?.hostname ?? "";
  const localHost = isLocalhostHost(hostname);
  const local = "/pyodide/";
  if (!localHost) {
    return "https://cdn.jsdelivr.net/pyodide/v0.29.4/full/";
  }
  try {
    const response = await fetch(`${local}pyodide.mjs`, { method: "HEAD" });
    if (response.ok) {
      return local;
    }
  } catch {
    // ignore
  }
  return "https://cdn.jsdelivr.net/pyodide/v0.29.4/full/";
}

async function loadPyodideModule(indexURL: string): Promise<LoadPyodideFn> {
  const module = await import(/* @vite-ignore */ `${indexURL}pyodide.mjs`);
  return module.loadPyodide as LoadPyodideFn;
}

function installLocalTorchPackage(pyodide: PyodideApi): void {
  pyodide.runPython(`
import sys
for name in list(sys.modules):
    if name == "torch" or name.startswith("torch."):
        sys.modules.pop(name, None)
`);
  pyodide.FS.mkdirTree("/home/pyodide/torch");
  pyodide.FS.writeFile("/home/pyodide/torch/__init__.py", initPy);
  pyodide.FS.writeFile("/home/pyodide/torch/cuda.py", cudaPy);
  pyodide.FS.writeFile("/home/pyodide/torch/_runtime.py", runtimePy);
  pyodide.FS.writeFile("/home/pyodide/torch/_tensor.py", tensorPy);
  pyodide.FS.mkdirTree("/home/pyodide/torch/nn");
  pyodide.FS.writeFile("/home/pyodide/torch/nn/__init__.py", nnInitPy);
  pyodide.FS.writeFile("/home/pyodide/torch/nn/modules.py", nnModulesPy);
  pyodide.FS.writeFile("/home/pyodide/torch/nn/functional.py", nnFunctionalPy);
  pyodide.runPython(`
import sys
home = "/home/pyodide"
if home in sys.path:
    sys.path.remove(home)
sys.path.insert(0, home)
`);
}

async function installPublishedTorchPackage(pyodide: PyodideApi): Promise<void> {
  await pyodide.loadPackage("micropip");
  await pyodide.runPythonAsync(`
import micropip
await micropip.install("torch-pyodide")
`);
}

async function verifyInstalledTorch(pyodide: PyodideApi): Promise<void> {
  await pyodide.runPythonAsync(`
import torch
from torch._tensor import _js_meta_to_tuple
tensor_id, shape, dtype = _js_meta_to_tuple({"id": 1, "shape": [2], "dtype": "float32"})
assert tensor_id == 1
assert shape == [2]
assert dtype == "float32"
assert hasattr(torch, "cuda")
assert callable(torch.cuda.is_available)
`);
}

function isLocalhostHost(hostname: string): boolean {
  return hostname === "localhost" || hostname === "127.0.0.1" || hostname === "::1";
}

type InstallMode = "published" | "local-dev";

type BootstrapOptions = {
  forcePublishedFailure?: boolean;
  preferLocalFallbackInProduction?: boolean;
};

function summarizeError(error: unknown): string {
  const compact = String(error).replace(/\s+/g, " ").trim();
  if (compact.length <= 220) {
    return compact;
  }
  return `${compact.slice(0, 217)}...`;
}

export async function bootstrapPyodideTorch(options?: BootstrapOptions) {
  installTorchRuntime(globalThis);
  const indexURL = await resolvePyodideIndexURL();
  const loadPyodide = await loadPyodideModule(indexURL);
  const pyodide = await loadPyodide({ indexURL });
  const hostname = globalThis.location?.hostname ?? "";
  const localHost = isLocalhostHost(hostname);
  const preferLocalFallbackInProduction = options?.preferLocalFallbackInProduction ?? true;

  let installMode: InstallMode = "published";
  let installDetail = "Installed torch-pyodide via micropip from published index.";

  if (!localHost && preferLocalFallbackInProduction) {
    installLocalTorchPackage(pyodide);
    await verifyInstalledTorch(pyodide);
    installMode = "local-dev";
    installDetail = "Using bundled local fallback in production.";
    return { pyodide, indexURL, installMode, installDetail };
  }

  try {
    if (options?.forcePublishedFailure) {
      throw new Error("Forced published install failure for test scenario.");
    }
    await installPublishedTorchPackage(pyodide);
    await verifyInstalledTorch(pyodide);
  } catch (error) {
    const publishedError = summarizeError(error);
    installLocalTorchPackage(pyodide);
    await verifyInstalledTorch(pyodide);
    if (!localHost) {
      installMode = "local-dev";
      installDetail =
        `Published install/verify failed in production; bundled local fallback verified. ` +
        `Cause: ${publishedError}`;
    } else {
      installMode = "local-dev";
      installDetail =
        `Published install/verify failed; local-dev fallback verified. ` +
        `Cause: ${publishedError}`;
    }
  }

  return { pyodide, indexURL, installMode, installDetail };
}

export type { PyodideApi };
