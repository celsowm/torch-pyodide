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

## Build wheel local

```powershell
npm run build:wheel
```

Wheel gerado em `python/dist/`.

## Publicação automática (PyPI + micropip)

- CI (`.github/workflows/ci.yml`) gera wheel em todo push/PR e publica como artifact.
- Release Python (`.github/workflows/publish-python.yml`) publica no PyPI quando você cria tag `v*` (ex.: `v0.0.2`) e anexa o wheel no GitHub Release.

### Instalação com pip

```bash
pip install torch-pyodide
```

### Uso no navegador com Pyodide + WebGPU

`torch-pyodide` tem duas partes:

- o pacote Python `torch`, instalado no Pyodide com `micropip`;
- o runtime JavaScript/WebGPU, que deve ser carregado e registrado em `globalThis` antes de importar `torch`.

Ou seja: `micropip.install("torch-pyodide")` sozinho instala o pacote Python, mas o pacote só executa tensores se `globalThis.__torch_pyodide_runtime__` já tiver sido configurado pelo runtime JS.

Exemplo mínimo:

```html
<script type="module">
  import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.29.4/full/pyodide.mjs";
  import { installTorchRuntime } from "./vendor/torch-pyodide/runtime.mjs";

  installTorchRuntime(globalThis);

  const pyodide = await loadPyodide({
    indexURL: "https://cdn.jsdelivr.net/pyodide/v0.29.4/full/",
  });

  await pyodide.loadPackage("micropip");
  await pyodide.runPythonAsync(`
import micropip
await micropip.install("torch-pyodide")
`);

  await pyodide.runPythonAsync(`
import torch

x = torch.tensor([1.0, 2.0, 3.0])
print(torch.cuda.is_available())
print(x.tolist())
`);
</script>
```

O arquivo `runtime.mjs` deve ser um bundle ESM do runtime deste repositório. Para gerar um bundle local:

```powershell
npm run build:runtime
```

Depois copie ou publique o bundle de runtime junto da sua aplicação. O playground do projeto usa a mesma sequência via `runtime/demo/shared.ts`: instala o runtime JS, instala o pacote Python e só então executa código com `import torch`.

O navegador/dispositivo precisa disponibilizar WebGPU. Se não houver adapter WebGPU, operações como `torch.tensor(...)` falharão no runtime.

### Instalação do pacote Python com micropip

```python
import micropip
await micropip.install("torch-pyodide")
```

Ou direto de um wheel no GitHub Release:

```python
import micropip
await micropip.install("https://github.com/celsowm/torch-pyodide/releases/download/v0.0.2/torch_pyodide-0.0.2-py3-none-any.whl")
```

## Build completo do Pyodide (opcional)

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_pyodide_full.ps1
```
