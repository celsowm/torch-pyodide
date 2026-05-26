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

## Testes browser com GPU visível (local)

```powershell
npm run test:browser:gpu
```

## Testes Python (utilitários locais)

```powershell
npm run test:python
```

## Compatibilidade com PyTorch (percentual)

```powershell
npm run compat:report
```

Saídas:
- `python/compatibility/report.json`
- `python/compatibility/report.md`

## Build runtime

```powershell
npm run build:runtime
```

## Build runtime de distribuição (`runtime.mjs`)

```powershell
npm run build:runtime:distribution
```

## Build wheel local

```powershell
npm run build:wheel
```

Wheel gerado em `python/dist/`.

## Publicação automática (PyPI + runtime + manifests)

- CI (`.github/workflows/ci.yml`) valida paridade de versão wheel/runtime e gera wheel em push/PR.
- Pages (`.github/workflows/deploy-pages.yml`) publica:
  - `latest.json` estável;
  - `manifests/<versao>.json`;
  - `runtime/<versao>/runtime.mjs`.
- Release Python (`.github/workflows/publish-python.yml`) publica no PyPI em tag `v*` e anexa `wheel`, `runtime.mjs` e `manifest.json` no GitHub Release.

Endpoint estável do canal:

- `https://celsowm.github.io/torch-pyodide/latest.json`
- `https://celsowm.github.io/torch-pyodide/manifests/<versao>.json`

### Instalação com pip

```bash
pip install torch-pyodide
```

### Uso no navegador com Pyodide + WebGPU

`torch-pyodide` tem duas partes:

- pacote Python (`wheel`), instalado no Pyodide com `micropip`;
- runtime JavaScript/WebGPU (`runtime.mjs`), que precisa ser carregado antes de `import torch`.

O cliente **não precisa hardcodar versão/URL**. Fluxo recomendado:

1. buscar `latest.json`;
2. baixar `runtimeUrl` + `wheelUrl` do mesmo manifest;
3. validar `runtimeSha256` + `wheelSha256`;
4. instalar runtime e wheel;
5. opcionalmente remover wheels antigos do seu cache local (manter só 1 versão).

Exemplo mínimo:

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

O navegador/dispositivo precisa disponibilizar WebGPU. Se não houver adapter WebGPU, operações como `torch.tensor(...)` falharão no runtime.

Schema do manifest:

- `torchVersion`
- `runtimeUrl`
- `wheelUrl`
- `runtimeSha256`
- `wheelSha256`

## Build completo do Pyodide (opcional)

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_pyodide_full.ps1
```
