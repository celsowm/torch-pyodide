import { bootstrapPyodideTorch } from "../demo/shared";
import { EditorState } from "@codemirror/state";
import { EditorView, keymap, lineNumbers } from "@codemirror/view";
import { defaultKeymap, history, historyKeymap } from "@codemirror/commands";
import { python } from "@codemirror/lang-python";
import { oneDark } from "@codemirror/theme-one-dark";

const defaultCode = `import json
import torch

torch.init()
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.ones((2, 2))
c = a.add(b).mul(torch.tensor([[2.0, 2.0], [2.0, 2.0]]))
out = {
  "shape": list(c.shape),
  "values": c.to_list(),
  "sum": c.sum().to_list()[0],
  "mean": c.mean().to_list()[0],
}
print(json.dumps(out, indent=2))
`;

const runButton = document.getElementById("run") as HTMLButtonElement;
const resetButton = document.getElementById("reset") as HTMLButtonElement;
const editorHost = document.getElementById("editor") as HTMLDivElement;
const output = document.getElementById("output") as HTMLElement;
const meta = document.getElementById("meta") as HTMLElement;

const editorState = EditorState.create({
  doc: defaultCode,
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

async function main() {
  try {
    runButton.disabled = true;
    meta.textContent = "Loading Pyodide + runtime...";
    const { pyodide, indexURL, installMode, installDetail } = await bootstrapPyodideTorch();
    meta.textContent = `Ready. Pyodide: ${indexURL} | mode: ${installMode} | ${installDetail}`;

    runButton.disabled = false;
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
      editor.dispatch({
        changes: { from: 0, to: editor.state.doc.length, insert: defaultCode }
      });
    };
  } catch (error) {
    meta.textContent = "Failed to initialize playground.";
    output.textContent = String(error);
  }
}

main();
