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
  pyodide.FS.mkdirTree("/home/pyodide/torch");
  pyodide.FS.writeFile("/home/pyodide/torch/__init__.py", initPy);
  pyodide.FS.writeFile("/home/pyodide/torch/_runtime.py", runtimePy);
  pyodide.FS.writeFile("/home/pyodide/torch/_tensor.py", tensorPy);
  pyodide.runPython(`
import sys
if "/home/pyodide" not in sys.path:
    sys.path.insert(0, "/home/pyodide")
`);
}

async function installPublishedTorchPackage(pyodide: PyodideApi): Promise<void> {
  await pyodide.loadPackage("micropip");
  await pyodide.runPythonAsync(`
import micropip
await micropip.install("torch-pyodide")
`);
}

function isLocalhostHost(hostname: string): boolean {
  return hostname === "localhost" || hostname === "127.0.0.1" || hostname === "::1";
}

type InstallMode = "published" | "local-dev";

export async function bootstrapPyodideTorch() {
  installTorchRuntime(globalThis);
  const indexURL = await resolvePyodideIndexURL();
  const loadPyodide = await loadPyodideModule(indexURL);
  const pyodide = await loadPyodide({ indexURL });
  const hostname = globalThis.location?.hostname ?? "";
  const localHost = isLocalhostHost(hostname);

  let installMode: InstallMode = "published";
  let installDetail = "Installed torch-pyodide via micropip from published index.";

  if (localHost) {
    try {
      await installPublishedTorchPackage(pyodide);
    } catch (error) {
      installLocalTorchPackage(pyodide);
      installMode = "local-dev";
      installDetail = `Published install failed, using local-dev fallback: ${String(error)}`;
    }
  } else {
    await installPublishedTorchPackage(pyodide);
  }

  return { pyodide, indexURL, installMode, installDetail };
}

export type { PyodideApi };
