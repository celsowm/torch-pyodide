import initPy from "../../python/torch/__init__.py?raw";
import runtimePy from "../../python/torch/_runtime.py?raw";
import tensorPy from "../../python/torch/_tensor.py?raw";
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
  const local = "/pyodide/";
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
  pyodide.FS.writeFile("/home/pyodide/torch/_runtime.py", runtimePy);
  pyodide.FS.writeFile("/home/pyodide/torch/_tensor.py", tensorPy);
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
torch.init()
a = torch.tensor([1.0, 2.0])
b = torch.ones((2,))
c = a.add(b)
assert c.to_list() == [2.0, 3.0]
`);
}

function isLocalhostHost(hostname: string): boolean {
  return hostname === "localhost" || hostname === "127.0.0.1" || hostname === "::1";
}

type InstallMode = "published" | "local-dev";

type BootstrapOptions = {
  forcePublishedFailure?: boolean;
};

export async function bootstrapPyodideTorch(options?: BootstrapOptions) {
  installTorchRuntime(globalThis);
  const indexURL = await resolvePyodideIndexURL();
  const loadPyodide = await loadPyodideModule(indexURL);
  const pyodide = await loadPyodide({ indexURL });
  const hostname = globalThis.location?.hostname ?? "";
  const localHost = isLocalhostHost(hostname);

  let installMode: InstallMode = "published";
  let installDetail = "Installed torch-pyodide via micropip from published index.";

  try {
    if (options?.forcePublishedFailure) {
      throw new Error("Forced published install failure for test scenario.");
    }
    await installPublishedTorchPackage(pyodide);
    await verifyInstalledTorch(pyodide);
  } catch (error) {
    if (!localHost) {
      installLocalTorchPackage(pyodide);
      installMode = "local-dev";
      installDetail = `Published install/verify failed in production, using bundled local fallback: ${String(error)}`;
    } else {
      installLocalTorchPackage(pyodide);
      installMode = "local-dev";
      installDetail = `Published install/verify failed, using local-dev fallback: ${String(error)}`;
    }
  }

  return { pyodide, indexURL, installMode, installDetail };
}

export type { PyodideApi };
