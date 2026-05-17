# torch-pyodide (MVP)

Projeto MVP de um módulo `torch` em Python rodando no Pyodide com backend WebGPU.

## Estrutura

- `python/`: pacote Python `torch` (API síncrona usando `pyodide.ffi.run_sync`).
- `runtime/`: bridge JS/WebGPU + demo + testes browser.
- `external/pyodide`: submódulo Git do repositório Pyodide.
- `scripts/`: sync de assets, setup de dist e build completo opcional.

## Requisitos

- Node 24+
- npm 11+
- Python 3.11+

## Setup

1. Instalar dependências JS:

```powershell
npm install
```

2. Sincronizar shaders/helpers do projeto base:

```powershell
npm run sync:torchjs
```

3. Baixar artefatos do runtime Pyodide para `external/pyodide/dist` e `runtime/public/pyodide`:

```powershell
npm run setup:pyodide:dist
```

## Executar demo

```powershell
npm run demo
```

Abra `http://127.0.0.1:4173/demo/index.html`.

## Testes browser

```powershell
npm run test:browser
```

## Testes Python (utilitários locais)

```powershell
npm run test:python
```

## Build runtime

```powershell
npm run build:runtime
```

## Build completo do Pyodide (opcional)

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_pyodide_full.ps1
```
