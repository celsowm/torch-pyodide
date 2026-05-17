import { bootstrapPyodideTorch } from "./shared";

const statusEl = document.getElementById("status");

function setStatus(message: string) {
  if (statusEl) {
    statusEl.textContent = message;
  }
}

async function main() {
  let runSyncError = "";
  let installMode = "unknown";
  let installDetail = "";
  try {
    const params = new URLSearchParams(globalThis.location.search);
    const forceFallback = params.get("force_fallback") === "1";
    const bootstrap = await bootstrapPyodideTorch({
      forcePublishedFailure: forceFallback
    });
    const { pyodide, indexURL } = bootstrap;
    installMode = bootstrap.installMode;
    installDetail = bootstrap.installDetail;

    try {
      pyodide.runPython(`
from pyodide.ffi import run_sync
import asyncio
run_sync(asyncio.sleep(0))
`);
    } catch (error) {
      runSyncError = String(error);
    }

    const script = `
import math
import torch

a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.ones((2, 2))
c = a.add(b)
d = c.mul(torch.tensor([[2.0, 2.0], [2.0, 2.0]]))
e = d.sub(torch.ones((2,2)))
f = e.div(torch.tensor([[1.0, 5.0], [1.0, 3.0]]))
r = f.relu()
t = r.T
flat = t.reshape((4,))
m = a.matmul(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
s = d.sum()
mean = d.mean()

assert c.tolist() == [[2.0, 3.0], [4.0, 5.0]]
assert d.tolist() == [[4.0, 6.0], [8.0, 10.0]]
assert e.tolist() == [[3.0, 5.0], [7.0, 9.0]]
assert f.tolist() == [[3.0, 1.0], [7.0, 3.0]]
assert r.tolist() == [[3.0, 1.0], [7.0, 3.0]]
assert t.tolist() == [[3.0, 7.0], [1.0, 3.0]]
assert flat.tolist() == [3.0, 7.0, 1.0, 3.0]
assert m.tolist() == [[1.0, 2.0], [3.0, 4.0]]
assert abs(s.tolist() - 28.0) < 1e-6
assert abs(mean.tolist() - 7.0) < 1e-6
{
    "ok": True,
    "sum": s.tolist(),
    "mean": mean.tolist(),
    "shape": list(a.shape),
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
}
`;

    const result = (await pyodide.runPythonAsync(script)) as Record<string, unknown>;

    const statusPayload = {
      ok: true,
      result,
      indexURL,
      installMode,
      installDetail,
      runSyncError
    };
    (globalThis as typeof globalThis & { __torchMvpStatus?: unknown }).__torchMvpStatus = statusPayload;
    setStatus(JSON.stringify(statusPayload, null, 2));
  } catch (error) {
    const payload = { ok: false, error: String(error), installMode, installDetail, runSyncError };
    (globalThis as typeof globalThis & { __torchMvpStatus?: unknown }).__torchMvpStatus = payload;
    setStatus(JSON.stringify(payload, null, 2));
  }
}

main();
