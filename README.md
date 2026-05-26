# torch-pyodide (MVP)

MVP project for a Python `torch` module running on Pyodide with a WebGPU backend.

## Structure

- `python/`: Python `torch` package (sync API using `pyodide.ffi.run_sync`).
- `runtime/`: JS/WebGPU bridge + demo + browser tests.
- `external/pyodide`: Git submodule of the Pyodide repository.
- `scripts/`: asset sync, dist setup, and optional full build scripts.

## Requirements

- Node 24+
- npm 11+
- Python 3.11+

## Setup

1. Install JS dependencies:

```powershell
npm install
```

2. Sync shaders/helpers from the base project:

```powershell
npm run sync:torchjs
```

3. Download Pyodide runtime artifacts to `external/pyodide/dist` and `runtime/public/pyodide`:

```powershell
npm run setup:pyodide:dist
```

## Run demo

```powershell
npm run demo
```

Open `http://127.0.0.1:4173/demo/index.html`.

## Browser tests

```powershell
npm run test:browser
```

## Browser tests with visible GPU (local)

```powershell
npm run test:browser:gpu
```

## Python tests (local utilities)

```powershell
npm run test:python
```

## PyTorch compatibility report (percentage)

```powershell
npm run compat:report
```

Outputs:
- `python/compatibility/report.json`
- `python/compatibility/report.md`

## Build runtime

```powershell
npm run build:runtime
```

## Build distribution runtime (`runtime.mjs`)

```powershell
npm run build:runtime:distribution
```

## Build wheel locally

```powershell
npm run build:wheel
```

Wheel is generated in `python/dist/`.

## Automated publishing (PyPI + runtime + manifests)

- CI (`.github/workflows/ci.yml`) validates wheel/runtime version parity and builds wheel on every push/PR.
- Pages (`.github/workflows/deploy-pages.yml`) publishes:
  - stable `latest.json`;
  - `manifests/<version>.json`;
  - `runtime/<version>/runtime.mjs`.
- Python release (`.github/workflows/publish-python.yml`) publishes to PyPI on `v*` tags and attaches `wheel`, `runtime.mjs`, and `manifest.json` to GitHub Release.

Stable channel endpoint:

- `https://celsowm.github.io/torch-pyodide/latest.json`
- `https://celsowm.github.io/torch-pyodide/manifests/<version>.json`

## Install with pip

```bash
pip install torch-pyodide
```

## Browser usage with Pyodide + WebGPU

`torch-pyodide` has two parts:

- Python package (wheel), installed in Pyodide with `micropip`;
- JavaScript/WebGPU runtime (`runtime.mjs`), which must be loaded before `import torch`.

Clients do not need to hardcode version or URLs. Recommended flow:

1. fetch `latest.json`;
2. download `runtimeUrl` + `wheelUrl` from the same manifest;
3. validate `runtimeSha256` + `wheelSha256`;
4. install runtime and wheel;
5. optionally remove old wheels from local cache (keep one version).

Minimal example:

```html
<script type="module">
  import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.29.4/full/pyodide.mjs";

  const manifest = await fetch("https://celsowm.github.io/torch-pyodide/latest.json").then((r) => r.json());

  const runtime = await import(manifest.runtimeUrl);
  runtime.installTorchRuntime(globalThis);

  const pyodide = await loadPyodide({
    indexURL: "https://cdn.jsdelivr.net/pyodide/v0.29.4/full/",
  });

  await pyodide.loadPackage("micropip");
  await pyodide.runPythonAsync(`
import micropip
await micropip.install("${manifest.wheelUrl}")
`);

  await pyodide.runPythonAsync(`
import torch

x = torch.tensor([1.0, 2.0, 3.0])
print(torch.cuda.is_available())
print(x.tolist())
`);
</script>
```

Your browser/device must expose WebGPU. If no WebGPU adapter is available, operations such as `torch.tensor(...)` will fail in the runtime.

Manifest schema:

- `torchVersion`
- `runtimeUrl`
- `wheelUrl`
- `runtimeSha256`
- `wheelSha256`

## Full Pyodide build (optional)

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_pyodide_full.ps1
```
