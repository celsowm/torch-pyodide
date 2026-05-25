from __future__ import annotations

import warnings

from typing import Iterable, Callable, Optional
from torch import Tensor
from ..grad_mode import no_grad


_FRAME_FALLBACK_WARNED = False


def _warn_frame_fallback(where: str, exc: Exception) -> None:
    global _FRAME_FALLBACK_WARNED
    if _FRAME_FALLBACK_WARNED:
        return
    _FRAME_FALLBACK_WARNED = True
    warnings.warn(
        f"{where} unavailable in runtime; optimizer step running without frame batching ({type(exc).__name__}: {exc})",
        RuntimeWarning,
        stacklevel=3,
    )


def _begin_runtime_frame() -> tuple[object | None, bool, object]:
    from torch._runtime import _get_runtime, _run_js_awaitable

    runtime = None
    run_js = _run_js_awaitable
    try:
        runtime = _get_runtime()
        # beginFrame is synchronous in the JS runtime.
        runtime.beginFrame()
        return runtime, True, run_js
    except Exception as exc:
        _warn_frame_fallback("beginFrame", exc)
        return runtime, False, run_js


def _end_runtime_frame(runtime: object | None, frame_started: bool, run_js: object) -> None:
    if not frame_started or runtime is None:
        return
    try:
        run_js(runtime.endFrame())
    except Exception as exc:
        _warn_frame_fallback("endFrame", exc)


class Optimizer:
    def __init__(self, params: Iterable[Tensor], defaults: dict[str, object]) -> None:
        self.param_groups: list[dict[str, object]] = []
        self.state: dict[int, dict[str, object]] = {}

        # Inicializar grupos de parâmetros
        param_list = list(params)
        self.param_groups.append({
            "params": param_list,
            **defaults,
        })

        # Inicializar estado para cada parâmetro
        for p in param_list:
            if not p._requires_grad:
                raise ValueError(
                    "optimizer can only optimize parameters with `requires_grad=True`. "
                    "Use `param.requires_grad = True` to enable gradients."
                )

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zera os gradientes de todos os parâmetros."""
        for group in self.param_groups:
            for p in group["params"]:
                if set_to_none:
                    p.grad = None
                elif p.grad is not None:
                    import torch
                    p.grad = torch.zeros_like(p.grad)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        return loss

    def state_dict(self) -> dict[str, object]:
        """Retorna o estado do otimizador como dicionário."""
        return {
            "state": self.state,
            "param_groups": self.param_groups,
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Carrega o estado do otimizador."""
        self.state = state_dict["state"]
        self.param_groups = state_dict["param_groups"]


class SGD(Optimizer):
    """Otimizador SGD (Stochastic Gradient Descent) com momentum e weight decay."""
    
    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "dampening": dampening,
            "nesterov": nesterov,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = super().step(closure)
        runtime, frame_started, run_js = _begin_runtime_frame()
        with no_grad():
            for group in self.param_groups:
                lr = float(group["lr"])
                momentum = float(group["momentum"])
                weight_decay = float(group["weight_decay"])
                dampening = float(group["dampening"])
                nesterov = bool(group["nesterov"])

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    grad = p.grad
                    if id(p) not in self.state:
                        self.state[id(p)] = {}
                    state = self.state[id(p)]
                    if "momentum_buffer" not in state or state["momentum_buffer"] is None:
                        import torch
                        state["momentum_buffer"] = torch.zeros_like(p)
                    momentum_buffer = state["momentum_buffer"]

                    try:
                        run_js(runtime.sgdStep(
                            p._id,
                            grad._id,
                            momentum_buffer._id,
                            float(lr),
                            float(momentum),
                            float(weight_decay),
                            float(dampening),
                            bool(nesterov),
                        ))
                    except Exception:
                        if weight_decay != 0:
                            grad = grad.add(p.mul(weight_decay))
                        if momentum != 0:
                            buf = momentum_buffer.mul(momentum).add(grad.mul(1 - dampening))
                            state["momentum_buffer"] = buf
                            grad = grad.add(buf.mul(momentum)) if nesterov else buf
                        update = grad.mul(lr)
                        new_p = p.sub(update)
                        p._set(new_p)
        _end_runtime_frame(runtime, frame_started, run_js)
        return loss



class Adam(Optimizer):
    """Otimizador Adam com bias correction."""
    
    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= betas[0] < 1.0):
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not (0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid beta2 value: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
            
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = super().step(closure)
        runtime, frame_started, run_js = _begin_runtime_frame()
        with no_grad():
            for group in self.param_groups:
                lr = float(group["lr"])
                beta1, beta2 = group["betas"]
                eps = float(group["eps"])
                weight_decay = float(group["weight_decay"])
                amsgrad = bool(group["amsgrad"])
                
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    grad = p.grad

                    # Inicializar estado
                    if id(p) not in self.state:
                        self.state[id(p)] = {
                            "step": 0,
                            "exp_avg": None,
                            "exp_avg_sq": None,
                            "max_exp_avg_sq": None if amsgrad else None,
                        }
                    
                    state = self.state[id(p)]
                    state["step"] = int(state["step"]) + 1

                    if state["exp_avg"] is None:
                        import torch
                        state["exp_avg"] = torch.zeros_like(p)
                    if state["exp_avg_sq"] is None:
                        import torch
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    step = state["step"]
                    bias_correction1 = 1.0 - beta1 ** step
                    bias_correction2 = 1.0 - beta2 ** step

                    step_size = lr / bias_correction1
                    inv_sqrt_bc2 = 1.0 / (bias_correction2 ** 0.5)

                    try:
                        run_js(runtime.adamStep(
                            p._id,
                            grad._id,
                            exp_avg._id,
                            exp_avg_sq._id,
                            float(lr),
                            float(beta1),
                            float(beta2),
                            float(eps),
                            float(weight_decay),
                            float(step_size),
                            float(inv_sqrt_bc2),
                        ))
                    except Exception:
                        # Fallback path keeps previous behavior when fused runtime op is unavailable.
                        if weight_decay != 0:
                            grad = grad.add(p.mul(weight_decay))
                        exp_avg = exp_avg.mul(beta1).add(grad.mul(1 - beta1))
                        state["exp_avg"] = exp_avg
                        grad_sq = grad.mul(grad)
                        exp_avg_sq = exp_avg_sq.mul(beta2).add(grad_sq.mul(1 - beta2))
                        state["exp_avg_sq"] = exp_avg_sq
                        denom = exp_avg_sq.div(bias_correction2).sqrt().add(eps)
                        update = exp_avg.div(denom).mul(step_size)
                        new_p = p.sub(update)
                        p._set(new_p)
        _end_runtime_frame(runtime, frame_started, run_js)
        return loss


class AdamW(Optimizer):
    """Otimizador AdamW (Adam com weight decay desacoplado)."""
    
    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= betas[0] < 1.0):
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not (0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid beta2 value: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
            
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = super().step(closure)
        runtime, frame_started, run_js = _begin_runtime_frame()
        with no_grad():
            for group in self.param_groups:
                lr = float(group["lr"])
                beta1, beta2 = group["betas"]
                eps = float(group["eps"])
                weight_decay = float(group["weight_decay"])
                amsgrad = bool(group["amsgrad"])
                
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    grad = p.grad

                    if id(p) not in self.state:
                        self.state[id(p)] = {
                            "step": 0,
                            "exp_avg": None,
                            "exp_avg_sq": None,
                            "max_exp_avg_sq": None if amsgrad else None,
                        }
                    
                    state = self.state[id(p)]
                    state["step"] = int(state["step"]) + 1

                    if state["exp_avg"] is None:
                        import torch
                        state["exp_avg"] = torch.zeros_like(p)
                    if state["exp_avg_sq"] is None:
                        import torch
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    step = state["step"]
                    bias_correction1 = 1.0 - beta1 ** step
                    bias_correction2 = 1.0 - beta2 ** step

                    step_size = lr / bias_correction1
                    inv_sqrt_bc2 = 1.0 / (bias_correction2 ** 0.5)

                    try:
                        run_js(runtime.adamWStep(
                            p._id,
                            grad._id,
                            exp_avg._id,
                            exp_avg_sq._id,
                            float(lr),
                            float(beta1),
                            float(beta2),
                            float(eps),
                            float(weight_decay),
                            float(step_size),
                            float(inv_sqrt_bc2),
                        ))
                    except Exception:
                        if weight_decay != 0:
                            new_p = p.sub(p.mul(weight_decay * lr))
                            p._set(new_p)
                        exp_avg = exp_avg.mul(beta1).add(grad.mul(1 - beta1))
                        state["exp_avg"] = exp_avg
                        grad_sq = grad.mul(grad)
                        exp_avg_sq = exp_avg_sq.mul(beta2).add(grad_sq.mul(1 - beta2))
                        state["exp_avg_sq"] = exp_avg_sq
                        denom = exp_avg_sq.div(bias_correction2).sqrt().add(eps)
                        update = exp_avg.div(denom).mul(step_size)
                        new_p = p.sub(update)
                        p._set(new_p)
        _end_runtime_frame(runtime, frame_started, run_js)
        return loss


class RMSprop(Optimizer):
    """Otimizador RMSprop."""
    
    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= alpha < 1.0):
            raise ValueError(f"Invalid alpha value: {alpha}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
            
        defaults = {
            "lr": lr,
            "alpha": alpha,
            "eps": eps,
            "weight_decay": weight_decay,
            "momentum": momentum,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = super().step(closure)
        runtime, frame_started, run_js = _begin_runtime_frame()
        with no_grad():
            for group in self.param_groups:
                lr = float(group["lr"])
                alpha = float(group["alpha"])
                eps = float(group["eps"])
                weight_decay = float(group["weight_decay"])
                momentum = float(group["momentum"])
                
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    grad = p.grad

                    if id(p) not in self.state:
                        self.state[id(p)] = {
                            "square_avg": None,
                            "momentum_buffer": None,
                        }

                    state = self.state[id(p)]

                    if state["square_avg"] is None:
                        import torch
                        state["square_avg"] = torch.zeros_like(p)
                    if state["momentum_buffer"] is None:
                        import torch
                        state["momentum_buffer"] = torch.zeros_like(p)

                    square_avg = state["square_avg"]
                    momentum_buffer = state["momentum_buffer"]

                    try:
                        run_js(runtime.rmspropStep(
                            p._id,
                            grad._id,
                            square_avg._id,
                            momentum_buffer._id,
                            float(lr),
                            float(alpha),
                            float(eps),
                            float(weight_decay),
                            float(momentum),
                        ))
                    except Exception:
                        if weight_decay != 0:
                            grad = grad.add(p.mul(weight_decay))
                        grad_sq = grad.mul(grad)
                        square_avg = square_avg.mul(alpha).add(grad_sq.mul(1 - alpha))
                        state["square_avg"] = square_avg
                        denom = square_avg.sqrt().add(eps)
                        if momentum != 0:
                            buf = momentum_buffer.mul(momentum).add(grad.div(denom))
                            state["momentum_buffer"] = buf
                            update = buf
                        else:
                            update = grad.div(denom)
                        update = update.mul(lr)
                        new_p = p.sub(update)
                        p._set(new_p)
        _end_runtime_frame(runtime, frame_started, run_js)
        return loss
