import { bootstrapPyodideTorch } from "../demo/shared";
import { EditorState } from "@codemirror/state";
import { EditorView, keymap, lineNumbers } from "@codemirror/view";
import { defaultKeymap, history, historyKeymap } from "@codemirror/commands";
import { python } from "@codemirror/lang-python";
import { oneDark } from "@codemirror/theme-one-dark";

type PlaygroundExample = {
  id: string;
  label: string;
  code: string;
};

type ExamplesCatalog = {
  default?: string;
  examples: PlaygroundExample[];
};

const runButton = document.getElementById("run") as HTMLButtonElement;
const resetButton = document.getElementById("reset") as HTMLButtonElement;
const exampleSelect = document.getElementById("example-select") as HTMLSelectElement;
const editorHost = document.getElementById("editor") as HTMLDivElement;
const output = document.getElementById("output") as HTMLElement;
const meta = document.getElementById("meta") as HTMLElement;
const gpuLabel = document.getElementById("gpu-label") as HTMLElement;
let selectedExample: PlaygroundExample | null = null;
let examplesById = new Map<string, PlaygroundExample>();

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
      ".cm-scroller": { fontFamily: '"JetBrains Mono", Consolas, monospace' },
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

function assertValidCatalog(raw: unknown): ExamplesCatalog {
  if (!raw || typeof raw !== "object") {
    throw new Error("Invalid examples catalog: expected object.");
  }
  const value = raw as { default?: unknown; examples?: unknown };
  if (!Array.isArray(value.examples)) {
    throw new Error("Invalid examples catalog: 'examples' must be an array.");
  }
  const parsed: PlaygroundExample[] = value.examples.map((item, index) => {
    const candidate = item as { id?: unknown; label?: unknown; code?: unknown };
    if (
      !candidate ||
      typeof candidate.id !== "string" ||
      typeof candidate.label !== "string" ||
      typeof candidate.code !== "string"
    ) {
      throw new Error(`Invalid examples catalog: item at index ${index} must include string id/label/code.`);
    }
    return { id: candidate.id, label: candidate.label, code: candidate.code };
  });
  if (parsed.length === 0) {
    throw new Error("Invalid examples catalog: 'examples' cannot be empty.");
  }
  return {
    default: typeof value.default === "string" ? value.default : undefined,
    examples: parsed
  };
}

async function loadExamplesCatalog(): Promise<ExamplesCatalog> {
  const response = await fetch("./examples.json", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load examples catalog: HTTP ${response.status}.`);
  }
  const data = (await response.json()) as unknown;
  return assertValidCatalog(data);
}

function initializeExampleSelection(catalog: ExamplesCatalog): void {
  examplesById = new Map(catalog.examples.map((example) => [example.id, example]));
  exampleSelect.innerHTML = "";
  for (const example of catalog.examples) {
    const option = document.createElement("option");
    option.value = example.id;
    option.textContent = example.label;
    exampleSelect.appendChild(option);
  }

  const preferred = catalog.default && examplesById.has(catalog.default) ? catalog.default : catalog.examples[0].id;
  selectedExample = examplesById.get(preferred) ?? catalog.examples[0];
  exampleSelect.value = selectedExample.id;
  setEditorCode(selectedExample.code);
}

async function main() {
  try {
    runButton.disabled = true;
    resetButton.disabled = true;
    exampleSelect.disabled = true;
    meta.textContent = "Loading Pyodide + runtime...";
    const { pyodide, indexURL, installMode, installDetail } = await bootstrapPyodideTorch();
    const catalog = await loadExamplesCatalog();
    initializeExampleSelection(catalog);
    const shortInstallDetail =
      installMode === "published" ? "Published package active." : "Using bundled local fallback.";
    meta.textContent = `Ready. Pyodide: ${indexURL} | mode: ${installMode} | ${shortInstallDetail}`;
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
      const next = examplesById.get(exampleSelect.value);
      if (!next) {
        return;
      }
      selectedExample = next;
      setEditorCode(next.code);
    };
    runButton.onclick = async () => {
      output.textContent = "";
      runButton.disabled = true;
      try {
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
        output.textContent = captured || "(no stdout)";
      } catch (error) {
        output.textContent = `ERROR:\n${String(error)}`;
      } finally {
        runButton.disabled = false;
      }
    };

    resetButton.onclick = () => {
      if (!selectedExample) {
        return;
      }
      setEditorCode(selectedExample.code);
    };
  } catch (error) {
    meta.textContent = `Failed to initialize playground: ${String(error)}`;
    output.textContent = String(error);
    runButton.disabled = true;
    resetButton.disabled = true;
    exampleSelect.disabled = true;
  }
}

main();
