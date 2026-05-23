# AGENTS.md

## Purpose
This repository implements `torch-pyodide`: a PyTorch-compatible API running in the browser via **Pyodide + WebGPU**.

## Quick map
- `python/`: Python `torch` package (public API, autograd, nn, compatibility).
- `runtime/`: TypeScript/WebGPU runtime, WGSL shaders, demo/playground, and Playwright tests.
- `scripts/`: automation for asset/version sync and Pyodide dist setup.
- `external/pyodide/`: upstream Pyodide git submodule (third-party, do not edit by default).

## Scope rules
- Prioritize changes in `python/`, `runtime/`, and `scripts/`.
- **Do not modify `external/pyodide/`** unless the task explicitly asks for it.
- Preserve public `torch` API compatibility; prefer adding tests when behavior changes.

## Setup
Prerequisites:
- Node.js 24+
- npm 11+
- Python 3.11+

Initial setup:
```powershell
npm install
npm run sync:torchjs
npm run setup:pyodide:dist
```

## Main commands
- Runtime demo: `npm run demo`
- Browser tests (with validations): `npm run test:browser`
- Local browser GPU tests: `npm run test:browser:gpu`
- Python tests: `npm run test:python`
- Compatibility report: `npm run compat:report`
- Runtime build: `npm run build:runtime`
- Wheel build: `npm run build:wheel`

## Recommended change workflow
1. Identify the affected layer (`python` vs `runtime`).
2. Make the smallest cohesive change.
3. Run targeted validation first:
   - Runtime: `npm --workspace runtime run validate`
   - Python: `cd python && python -m pytest tests`
4. Run the relevant integration command (`npm run test:browser` or `npm run test:python`).
5. If changing distributed version/API metadata, run `npm run sync:version` (already included in build scripts).

## Useful conventions
- GPU ops and kernels live in `runtime/src/ops` and `runtime/src/vendor/torchjs/shaders`.
- Python `torch` API lives in `python/torch/` (`_tensor*`, `nn/`, `optim/`, `autograd_*` modules).
- Playground examples are in `runtime/playground/public/examples/*.py` and can be used as functional smoke tests.

## Pre-delivery checklist
- Change keeps expected public API behavior.
- Relevant tests passed locally.
- No accidental edits under `external/pyodide/`.
- Docs/comments updated when needed.
