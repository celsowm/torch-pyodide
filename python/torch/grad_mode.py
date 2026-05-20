from __future__ import annotations

_grad_enabled = True


def set_grad_enabled(mode: bool) -> None:
    global _grad_enabled
    _grad_enabled = mode


def is_grad_enabled() -> bool:
    return _grad_enabled


class no_grad:
    """Context manager que desabilita o calculo de gradientes."""

    def __enter__(self) -> "no_grad":
        global _grad_enabled
        self._prev = _grad_enabled
        _grad_enabled = False
        return self

    def __exit__(self, *args: object) -> None:
        global _grad_enabled
        _grad_enabled = self._prev


class inference_mode:
    """Context manager para modo de inferencia (alias para no_grad)."""

    def __init__(self, mode: bool = True) -> None:
        self._mode = mode

    def __enter__(self) -> "inference_mode":
        global _grad_enabled
        self._prev = _grad_enabled
        if self._mode:
            _grad_enabled = False
        return self

    def __exit__(self, *args: object) -> None:
        global _grad_enabled
        _grad_enabled = self._prev
