import initPy from "../../python/torch/__init__.py?raw";
import runtimePy from "../../python/torch/_runtime.py?raw";
import tensorPy from "../../python/torch/_tensor.py?raw";
import { installTorchRuntime } from "../src";

type PyodideApi = {
  runPython: (code: string) => unknown;
  runPythonAsync: (code: string) => Promise<unknown>;
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
}

export async function bootstrapPyodideTorch() {
  installTorchRuntime(globalThis);
  const indexURL = await resolvePyodideIndexURL();
  const loadPyodide = await loadPyodideModule(indexURL);
  const pyodide = await loadPyodide({ indexURL });
  installLocalTorchPackage(pyodide);
  return { pyodide, indexURL };
}

export type { PyodideApi };

