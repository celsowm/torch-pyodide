from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from ._runtime import _get_runtime, _run_js_awaitable


@dataclass(slots=True)
class CudaLikeDeviceProperties:
    name: str
    total_memory: int
    major: int
    minor: int
    multi_processor_count: int
    vendor: str = ""
    architecture: str = ""
    description: str = ""
    device: str = ""
    is_fallback_adapter: bool = False
    subgroup_min_size: int = 0
    subgroup_max_size: int = 0
    limits: dict[str, int] | None = None


def _normalize_device(device: int | None) -> int:
    if device is None:
        return current_device()
    value = int(device)
    if value != 0:
        raise RuntimeError(f"Only device index 0 is supported in MVP, received: {value}.")
    return value


def is_available() -> bool:
    runtime = _get_runtime()
    return bool(runtime.isAvailable())


def is_initialized() -> bool:
    runtime = _get_runtime()
    return bool(runtime.isInitialized())


def init() -> None:
    runtime = _get_runtime()
    _run_js_awaitable(runtime.init())


def device_count() -> int:
    runtime = _get_runtime()
    return int(runtime.deviceCount())


def current_device() -> int:
    runtime = _get_runtime()
    return int(_run_js_awaitable(runtime.currentDevice()))


def get_device_name(device: int | None = None) -> str:
    runtime = _get_runtime()
    normalized = _normalize_device(device)
    return str(_run_js_awaitable(runtime.getDeviceName(normalized)))


def get_device_properties(device: int | None = None) -> CudaLikeDeviceProperties:
    runtime = _get_runtime()
    normalized = _normalize_device(device)
    raw = _run_js_awaitable(runtime.getDeviceProperties(normalized))
    if hasattr(raw, "to_py"):
        raw = raw.to_py()
    data = dict(raw)
    limits = data.get("limits")
    if hasattr(limits, "to_py"):
        limits = limits.to_py()
    limits_dict = dict(limits) if isinstance(limits, dict) else None
    return CudaLikeDeviceProperties(
        name=str(data.get("name", "WebGPU Adapter")),
        total_memory=int(data.get("total_memory", 0)),
        major=int(data.get("major", 0)),
        minor=int(data.get("minor", 0)),
        multi_processor_count=int(data.get("multi_processor_count", 0)),
        vendor=str(data.get("vendor", "")),
        architecture=str(data.get("architecture", "")),
        description=str(data.get("description", "")),
        device=str(data.get("device", "")),
        is_fallback_adapter=bool(data.get("is_fallback_adapter", False)),
        subgroup_min_size=int(data.get("subgroup_min_size", 0)),
        subgroup_max_size=int(data.get("subgroup_max_size", 0)),
        limits=limits_dict,
    )


def memory_allocated(device: int | None = None) -> int:
    runtime = _get_runtime()
    normalized = _normalize_device(device)
    return int(_run_js_awaitable(runtime.memoryAllocated(normalized)))


def memory_reserved(device: int | None = None) -> int:
    runtime = _get_runtime()
    normalized = _normalize_device(device)
    return int(_run_js_awaitable(runtime.memoryReserved(normalized)))


memory = SimpleNamespace(
    memory_allocated=memory_allocated,
    memory_reserved=memory_reserved,
)

