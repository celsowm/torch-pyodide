from __future__ import annotations

from typing import Any


RUNTIME_GLOBAL_KEY = "__torch_pyodide_runtime__"


def _to_python(value: Any) -> Any:
    if hasattr(value, "to_py"):
        return value.to_py()
    return value


def _get_runtime() -> Any:
    try:
        from js import globalThis  # type: ignore
    except Exception as exc:  # pragma: no cover - browser specific
        raise RuntimeError("This package only runs inside Pyodide.") from exc

    runtime = getattr(globalThis, RUNTIME_GLOBAL_KEY, None)
    if runtime is None:
        raise RuntimeError(
            "Torch runtime bridge not found. Install runtime JS and set "
            "globalThis.__torch_pyodide_runtime__ before importing torch."
        )
    return runtime


def _run_js_awaitable(awaitable: Any) -> Any:
    from pyodide.ffi import can_run_sync, run_sync  # type: ignore

    if not can_run_sync():
        raise RuntimeError(
            "Cannot use synchronous torch API: run_sync is unavailable. "
            "Execute Python through pyodide.runPythonAsync(...) and call "
            "sync Python functions from JS via callPromising(...)."
        )
    return _to_python(run_sync(awaitable))

