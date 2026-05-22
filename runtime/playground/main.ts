import { bootstrapPyodideTorch } from "../demo/shared";
import { EditorState } from "@codemirror/state";
import { EditorView, keymap, lineNumbers } from "@codemirror/view";
import { defaultKeymap, history, historyKeymap } from "@codemirror/commands";
import { python } from "@codemirror/lang-python";
import { oneDark } from "@codemirror/theme-one-dark";

type ExampleMeta = {
  id: string;
  label: string;
  file: string;
};

let selectedMeta: ExampleMeta | null = null;
let metaList: ExampleMeta[] = [];
let codeCache = new Map<string, string>();

const runButton = document.getElementById("run") as HTMLButtonElement;
const resetButton = document.getElementById("reset") as HTMLButtonElement;
const exampleSelect = document.getElementById("example-select") as HTMLSelectElement;
const editorHost = document.getElementById("editor") as HTMLDivElement;
const output = document.getElementById("output") as HTMLElement;
const meta = document.getElementById("meta") as HTMLElement;
const gpuLabel = document.getElementById("gpu-label") as HTMLElement;
let selectedExample: ExampleMeta | null = null;
let examplesById = new Map<string, ExampleMeta>();

function detectWebglRenderer(): string {
  const canvas = document.createElement("canvas");
  const gl =
    (canvas.getContext("webgl2") as WebGLRenderingContext | null) ||
    (canvas.getContext("webgl") as WebGLRenderingContext | null);
  if (!gl) {
    return "";
  }
  const ext = gl.getExtension("WEBGL_debug_renderer_info") as {
    UNMASKED_RENDERER_WEBGL: number;
  } | null;
  if (!ext) {
    return "";
  }
  const renderer = gl.getParameter(ext.UNMASKED_RENDERER_WEBGL);
  return typeof renderer === "string" ? renderer : "";
}

function setGpuLabel(value: string): void {
  gpuLabel.textContent = value || "Unknown GPU";
}

function setOutputState(state: "empty" | "running" | "error" | "ready", text: string): void {
  output.classList.toggle("is-empty", state === "empty");
  output.classList.toggle("is-running", state === "running");
  output.classList.toggle("is-error", state === "error");
  output.textContent = text;
}

const editorState = EditorState.create({
  doc: "",
  extensions: [
    lineNumbers(),
    history(),
    keymap.of([...defaultKeymap, ...historyKeymap]),
    python(),
    oneDark,
    EditorView.theme({
      "&": { height: "100%", minHeight: "0", fontSize: "13px" },
      ".cm-editor": { height: "100%", minHeight: "0" },
      ".cm-scroller": { height: "100%", minHeight: "0", overflow: "auto", fontFamily: '"JetBrains Mono", Consolas, monospace' },
      ".cm-content": { caretColor: "#ffffff" }
    })
  ]
});

const editor = new EditorView({
  state: editorState,
  parent: editorHost
});

function setEditorCode(code: string): void {
  editor.dispatch({
    changes: { from: 0, to: editor.state.doc.length, insert: code }
  });
}

function assetUrl(path: string): string {
  return new URL(path, `${window.location.origin}${import.meta.env.BASE_URL}`).toString();
}

async function fetchCode(file: string): Promise<string> {
  const cached = codeCache.get(file);
  if (cached !== undefined) return cached;
  const fileName = file.substring(file.lastIndexOf("/") + 1);
  const url = assetUrl("examples/" + fileName);
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to fetch example "${file}": HTTP ${response.status}.`);
  }
  const code = await response.text();
  codeCache.set(file, code);
  return code;
}

function assertValidCatalog(raw: unknown): { default?: string; metaList: ExampleMeta[] } {
  if (!raw || typeof raw !== "object") {
    throw new Error("Invalid examples catalog: expected object.");
  }
  const value = raw as { default?: unknown; examples?: unknown };
  if (!Array.isArray(value.examples)) {
    throw new Error("Invalid examples catalog: 'examples' must be an array.");
  }
  metaList = value.examples.map((item, index) => {
    const candidate = item as { id?: unknown; label?: unknown; file?: unknown };
    if (
      !candidate ||
      typeof candidate.id !== "string" ||
      typeof candidate.label !== "string" ||
      typeof candidate.file !== "string"
    ) {
      throw new Error(`Invalid examples catalog: item at index ${index} must include string id/label/file.`);
    }
    return { id: candidate.id, label: candidate.label, file: candidate.file };
  });
  if (metaList.length === 0) {
    throw new Error("Invalid examples catalog: 'examples' cannot be empty.");
  }
  return {
    default: typeof value.default === "string" ? value.default : undefined,
    metaList
  };
}

async function loadCatalog(): Promise<string | undefined> {
  const response = await fetch(assetUrl("examples.json"), { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load examples catalog: HTTP ${response.status}.`);
  }
  const raw = (await response.json()) as unknown;
  const catalog = assertValidCatalog(raw);

  exampleSelect.innerHTML = "";
  for (const meta of catalog.metaList) {
    examplesById.set(meta.id, meta);
    const option = document.createElement("option");
    option.value = meta.id;
    option.textContent = meta.label;
    exampleSelect.appendChild(option);
  }

  const defaultId = catalog.default && examplesById.has(catalog.default) ? catalog.default : catalog.metaList[0].id;
  return defaultId;
}

async function switchToExample(id: string): Promise<void> {
  const meta = examplesById.get(id);
  if (!meta) return;
  selectedMeta = meta;
  const code = await fetchCode(meta.file);
  setEditorCode(code);
}

async function main() {
  try {
    runButton.disabled = true;
    resetButton.disabled = true;
    exampleSelect.disabled = true;
    meta.textContent = "Loading Pyodide + runtime...";
    const { pyodide, indexURL, installMode, installDetail } = await bootstrapPyodideTorch({
      preferLocalFallbackInProduction: true
    });
    const defaultId = await loadCatalog();
    await switchToExample(defaultId!);
    const shortInstallDetail =
      installMode === "published" ? "Published package active." : "Using bundled local fallback.";
    meta.textContent = `Ready. Pyodide: ${indexURL} | mode: ${installMode} | ${shortInstallDetail}`;
    meta.style.display = "none";
    if (installMode !== "published") {
      console.warn(`[torch-pyodide] ${installDetail}`);
    }

    const webglRenderer = detectWebglRenderer();
    if (webglRenderer) {
      setGpuLabel(`${webglRenderer} (WEBGL_debug_renderer_info)`);
    } else {
      try {
        const runtimeGpu = String(
          pyodide.runPython(`
import torch
torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No WebGPU adapter available"
`)
        );
        setGpuLabel(`${runtimeGpu} (torch.cuda)`);
      } catch {
        setGpuLabel("No GPU details available");
      }
    }

    runButton.disabled = false;
    resetButton.disabled = false;
    exampleSelect.disabled = false;
    exampleSelect.onchange = () => {
      const id = exampleSelect.value;
      switchToExample(id);
    };
    runButton.onclick = async () => {
      setOutputState("running", "Running...");
      runButton.disabled = true;
      try {
        const webgpuActive = String(
          pyodide.runPython(`
import torch
str(bool(torch.cuda.is_available()))
`)
        );
        console.log(`[torch-pyodide] WebGPU active: ${webgpuActive}`);

        const code = editor.state.doc.toString();
        pyodide.runPython(`
import io
import sys
_torch_playground_buf = io.StringIO()
_torch_playground_stdout = sys.stdout
sys.stdout = _torch_playground_buf
`);
        await pyodide.runPythonAsync(code);
        const captured = String(
          pyodide.runPython(`
sys.stdout = _torch_playground_stdout
_torch_playground_buf.getvalue()
`)
        );
        setOutputState(captured ? "ready" : "empty", captured || "(no stdout)");
      } catch (error) {
        setOutputState("error", `ERROR:\n${String(error)}`);
      } finally {
        runButton.disabled = false;
      }
    };

    resetButton.onclick = () => {
      if (!selectedMeta) return;
      fetchCode(selectedMeta.file).then(setEditorCode);
    };
  } catch (error) {
    meta.textContent = `Failed to initialize playground: ${String(error)}`;
    setOutputState("error", String(error));
    runButton.disabled = true;
    resetButton.disabled = true;
    exampleSelect.disabled = true;
  }
}

main();
