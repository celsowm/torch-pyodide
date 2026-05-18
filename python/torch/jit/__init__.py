from __future__ import annotations

import functools


def script(obj: object) -> object:
    if isinstance(obj, type):
        return obj
    if callable(obj):
        @functools.wraps(obj)
        def wrapper(*args: object, **kwargs: object) -> object:
            return obj(*args, **kwargs)
        return wrapper
    return obj


def trace(func: object, *args: object, **kwargs: object) -> object:
    if callable(func):
        return func
    return func


def ignore(fn: object) -> object:
    return fn


def unused(fn: object) -> object:
    return fn


def load(f: str, map_location: str | None = None) -> object:
    raise NotImplementedError("torch.jit.load is not supported in torch-pyodide")


is_scripting = False
is_tracing = False
