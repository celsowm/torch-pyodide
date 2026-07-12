from __future__ import annotations

import math
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
        raw_state = state_dict["state"]
        # Pickle round-trip converts int keys to strings (Python's default
        # pickle protocol preserves dicts as a stream of (key, value)
        # pairs and Python pickles ints and strings as different opcodes
        # but the high-level dict reconstruction accepts both). The
        # optimizer's step() indexes self.state by id(p) (an int), so
        # convert any stringified keys back to ints.
        self.state = {int(k) if isinstance(k, str) and k.isdigit() else k: v for k, v in raw_state.items()}
        # Real PyTorch's state_dict stores `param_groups[*]['params']` as
        # a list of integer indices into the *original* param order; our
        # own state_dict stores the Tensors directly. In both cases the
        # saved list is positional — map it back to the *current* param
        # list so step() iterates over the right objects. (After a pickle
        # round-trip the loaded Tensors are fresh objects; without this
        # remap their `.grad` is None and step() silently skips them.)
        loaded_groups = state_dict["param_groups"]
        current_groups = self.param_groups
        for loaded_group, current_group in zip(loaded_groups, current_groups):
            loaded_params = loaded_group.get("params", [])
            current_params = list(current_group.get("params", []))
            rehydrated = []
            for idx, _ in enumerate(loaded_params):
                if idx < len(current_params):
                    rehydrated.append(current_params[idx])
                else:
                    rehydrated.append(loaded_params[idx])
            loaded_group["params"] = rehydrated
        self.param_groups = loaded_groups


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


class Adagrad(Optimizer):
    """Adagrad optimizer."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.01,
        lr_decay: float = 0.0,
        weight_decay: float = 0.0,
        initial_accumulator_value: float = 0.0,
        eps: float = 1e-10,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if lr_decay < 0.0:
            raise ValueError(f"Invalid lr_decay value: {lr_decay}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        defaults = {
            "lr": lr,
            "lr_decay": lr_decay,
            "weight_decay": weight_decay,
            "initial_accumulator_value": initial_accumulator_value,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = super().step(closure)
        runtime, frame_started, run_js = _begin_runtime_frame()
        with no_grad():
            for group in self.param_groups:
                lr = float(group["lr"])
                lr_decay = float(group["lr_decay"])
                weight_decay = float(group["weight_decay"])
                init_accum = float(group["initial_accumulator_value"])
                eps = float(group["eps"])
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if id(p) not in self.state:
                        import torch
                        self.state[id(p)] = {
                            "step": 0,
                            "sum": torch.full_like(p, init_accum),
                        }
                    state = self.state[id(p)]
                    state["step"] = int(state["step"]) + 1
                    sum_sq = state["sum"]
                    step = state["step"]
                    clr = lr / (1.0 + (step - 1) * lr_decay)
                    try:
                        run_js(runtime.adagradStep(
                            p._id,
                            grad._id,
                            sum_sq._id,
                            float(clr),
                            float(eps),
                            float(weight_decay),
                        ))
                    except Exception:
                        if weight_decay != 0:
                            grad = grad.add(p.mul(weight_decay))
                        sum_sq = sum_sq.add(grad.mul(grad))
                        state["sum"] = sum_sq
                        std_ = sum_sq.sqrt().add(eps)
                        new_p = p.sub(grad.div(std_).mul(clr))
                        p._set(new_p)
        _end_runtime_frame(runtime, frame_started, run_js)
        return loss


class Adamax(Optimizer):
    """Adamax optimizer (L-infinity variant of Adam)."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.002,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
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
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if id(p) not in self.state:
                        import torch
                        self.state[id(p)] = {
                            "step": 0,
                            "exp_avg": torch.zeros_like(p),
                            "exp_inf": torch.zeros_like(p),
                        }
                    state = self.state[id(p)]
                    state["step"] = int(state["step"]) + 1
                    exp_avg = state["exp_avg"]
                    exp_inf = state["exp_inf"]
                    step = state["step"]
                    bias_correction1 = 1.0 - beta1 ** step
                    step_size = lr / bias_correction1
                    try:
                        run_js(runtime.adamaxStep(
                            p._id,
                            grad._id,
                            exp_avg._id,
                            exp_inf._id,
                            float(lr),
                            float(beta1),
                            float(beta2),
                            float(eps),
                            float(weight_decay),
                            float(step_size),
                            float(1.0),  # bias_correction1 is folded into step_size on the shader side
                        ))
                    except Exception:
                        if weight_decay != 0:
                            grad = grad.add(p.mul(weight_decay))
                        exp_avg = exp_avg.mul(beta1).add(grad.mul(1.0 - beta1))
                        state["exp_avg"] = exp_avg
                        # exp_inf: max of (beta2 * exp_inf, |grad|)
                        import torch as _torch
                        new_exp_inf = _torch.maximum(exp_inf.mul(beta2), grad.abs())
                        state["exp_inf"] = new_exp_inf
                        denom = new_exp_inf.add(eps)
                        update = exp_avg.div(denom).mul(step_size)
                        new_p = p.sub(update)
                        p._set(new_p)
        _end_runtime_frame(runtime, frame_started, run_js)
        return loss


class NAdam(Optimizer):
    """NAdam optimizer (Nesterov-Adam)."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.002,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum_decay: float = 4e-3,
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
            "momentum_decay": momentum_decay,
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
                momentum_decay = float(group["momentum_decay"])
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if id(p) not in self.state:
                        import torch
                        self.state[id(p)] = {
                            "step": 0,
                            "exp_avg": torch.zeros_like(p),
                            "exp_avg_sq": torch.zeros_like(p),
                        }
                    state = self.state[id(p)]
                    state["step"] = int(state["step"]) + 1
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    step = state["step"]
                    # PyTorch NAdam: mu_t = beta1 * (1 - 0.5 * 0.96^(t * momentum_decay))
                    mu = beta1 * (1.0 - 0.5 * (0.96 ** (step * momentum_decay)))
                    mu_next = beta1 * (1.0 - 0.5 * (0.96 ** ((step + 1) * momentum_decay)))
                    bias_correction1 = 1.0 - beta1 ** step
                    bias_correction2 = 1.0 - beta2 ** step
                    step_size = lr / bias_correction1
                    try:
                        run_js(runtime.nadamStep(
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
                            float(mu_next),
                        ))
                    except Exception:
                        if weight_decay != 0:
                            grad = grad.add(p.mul(weight_decay))
                        exp_avg = exp_avg.mul(beta1).add(grad.mul(1.0 - beta1))
                        state["exp_avg"] = exp_avg
                        grad_sq = grad.mul(grad)
                        exp_avg_sq = exp_avg_sq.mul(beta2).add(grad_sq.mul(1.0 - beta2))
                        state["exp_avg_sq"] = exp_avg_sq
                        denom = exp_avg_sq.div(bias_correction2).sqrt().add(eps)
                        # Nesterov look-ahead
                        m_hat = exp_avg.mul(mu_next).add(grad.mul(1.0 - mu_next)).div(bias_correction1)
                        update = m_hat.div(denom).mul(step_size)
                        new_p = p.sub(update)
                        p._set(new_p)
        _end_runtime_frame(runtime, frame_started, run_js)
        return loss


class RAdam(Optimizer):
    """RAdam optimizer (Rectified Adam)."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
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
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if id(p) not in self.state:
                        import torch
                        self.state[id(p)] = {
                            "step": 0,
                            "exp_avg": torch.zeros_like(p),
                            "exp_avg_sq": torch.zeros_like(p),
                        }
                    state = self.state[id(p)]
                    state["step"] = int(state["step"]) + 1
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    step = state["step"]
                    bias_correction1 = 1.0 - beta1 ** step
                    bias_correction2 = 1.0 - beta2 ** step
                    step_size = lr / bias_correction1
                    try:
                        run_js(runtime.radamStep(
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
                            float(beta1 ** step),
                            float(beta2 ** step),
                            int(step),
                        ))
                    except Exception:
                        if weight_decay != 0:
                            grad = grad.add(p.mul(weight_decay))
                        exp_avg = exp_avg.mul(beta1).add(grad.mul(1.0 - beta1))
                        state["exp_avg"] = exp_avg
                        grad_sq = grad.mul(grad)
                        exp_avg_sq = exp_avg_sq.mul(beta2).add(grad_sq.mul(1.0 - beta2))
                        state["exp_avg_sq"] = exp_avg_sq
                        v_hat = exp_avg_sq.div(bias_correction2)
                        m_hat = exp_avg.div(bias_correction1)
                        # RAdam rectification
                        beta2_pow_t = beta2 ** step
                        rho_inf = 2.0 / (1.0 - beta2) - 1.0
                        import math
                        t_approx = max(step, 1)
                        rho_t = rho_inf - 2.0 * t_approx * beta2_pow_t / max(1.0 - beta2_pow_t, 1e-10)
                        if rho_t > 5.0:
                            rect = math.sqrt(
                                (rho_t - 4.0) * (rho_t - 2.0) * rho_inf
                                / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)
                            )
                            update = m_hat.mul(rect).mul(lr)
                        else:
                            update = m_hat.mul(lr)
                        new_p = p.sub(update)
                        p._set(new_p)
        _end_runtime_frame(runtime, frame_started, run_js)
        return loss


class LBFGS(Optimizer):
    """L-BFGS optimizer (Limited-memory BFGS quasi-Newton).

    Implements a simplified L-BFGS with a backtracking line search. The
    `closure` passed to `step(closure)` must zero gradients, evaluate the
    loss, call `loss.backward()`, and return the loss value (matching the
    contract of real PyTorch's `LBFGS.step(closure)`).

    The implementation:
      1. Stores the most recent `history_size` (s, y, rho) triples.
      2. Uses the two-loop recursion to compute the search direction.
      3. Performs a backtracking line search with `max_iter` evaluations
         of the closure, shrinking the step by `tolerance_change` factor.

    Args:
        lr: learning rate (initial step size for the line search).
        max_iter: maximum number of iterations per `step()` call.
        max_eval: maximum number of closure evaluations per `step()` call.
        tolerance_grad: termination tolerance on the gradient norm.
        tolerance_change: termination tolerance on the loss / step change.
        history_size: number of (s, y) pairs to keep.
        line_search_fn: None (uses our backtracking) or 'strong_wolfe'
            (a no-op stub that falls back to backtracking).
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1.0,
        max_iter: int = 20,
        max_eval: int | None = None,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        history_size: int = 100,
        line_search_fn: str | None = None,
    ) -> None:
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {max_iter}")
        if history_size <= 0:
            raise ValueError(f"history_size must be positive, got {history_size}")
        defaults = {
            "lr": lr,
            "max_iter": max_iter,
            "max_eval": int(max_eval),
            "tolerance_grad": tolerance_grad,
            "tolerance_change": tolerance_change,
            "history_size": history_size,
        }
        super().__init__(params, defaults)
        if line_search_fn not in (None, "strong_wolfe"):
            raise ValueError(
                f"line_search_fn must be None or 'strong_wolfe', got {line_search_fn}"
            )
        self._step_closure_eval_count = 0
        # Global state: per-param s, y, rho lists (limited-memory storage).
        for group in self.param_groups:
            for p in group["params"]:
                if id(p) not in self.state:
                    self.state[id(p)] = {"s_list": [], "y_list": [], "rho_list": []}

    def _gather_params(self):
        params: list[Tensor] = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    params.append(p)
        return params

    def _flatten(self, tensors: list[Tensor]) -> list[float]:
        out: list[float] = []
        for t in tensors:
            v = t.tolist()
            out.extend(self._flatten_list(v))
        return out

    def _flatten_list(self, v) -> list[float]:
        """Recursively flatten a (possibly nested) Python list of floats."""
        if isinstance(v, list):
            out: list[float] = []
            for item in v:
                out.extend(self._flatten_list(item))
            return out
        return [float(v)]

    def _flatten_nested(self, lists: list[list[float]]) -> list[float]:
        """Flatten a list of per-param float lists into a single flat list."""
        out: list[float] = []
        for lst in lists:
            for v in lst:
                out.append(float(v))
        return out

    def _unflatten_to_params(
        self, tensors: list[Tensor], flat: list[float]
    ) -> None:
        import torch as _torch
        idx = 0
        for t in tensors:
            size = 1
            for s in t._shape:
                size *= int(s)
            chunk = flat[idx:idx + size]
            t._set(_torch.tensor(chunk, dtype=t.dtype).reshape(list(t._shape)))
            idx += size

    def _norm(self, x: list[float]) -> float:
        s = 0.0
        for v in x:
            s += v * v
        return s ** 0.5

    def _add(self, a: list[float], b: list[float], sign: float = 1.0) -> list[float]:
        return [a[i] + sign * b[i] for i in range(len(a))]

    def _scale(self, a: list[float], s: float) -> list[float]:
        return [v * s for v in a]

    def _dot(self, a: list[float], b: list[float]) -> float:
        return sum(a[i] * b[i] for i in range(len(a)))

    def _two_loop(
        self,
        grad_flat: list[float],
        s_lists: list[list[list[float]]],
        y_lists: list[list[list[float]]],
        rho_lists: list[list[float]],
        history_size: int,
    ) -> list[float]:
        """L-BFGS two-loop recursion: returns H * g (the search direction)."""
        q = list(grad_flat)
        n = len(s_lists[0]) if s_lists[0] else 0
        k = min(n, history_size)
        alpha: list[float] = []
        for i in range(k - 1, -1, -1):
            s_i = [s_lists[p][n - k + i] for p in range(len(s_lists))]
            y_i = [y_lists[p][n - k + i] for p in range(len(y_lists))]
            rho_i = rho_lists[0][n - k + i]
            s_flat = self._flatten_nested(s_i)
            y_flat = self._flatten_nested(y_i)
            alpha_i = rho_i * self._dot(s_flat, q)
            alpha.append(alpha_i)
            q = self._add(q, y_flat, -alpha_i)
        # Initial Hessian approximation: H_0 = gamma * I, where
        # gamma = (s_{k-1} . y_{k-1}) / (y_{k-1} . y_{k-1}).
        gamma = 1.0
        if k > 0 and n > 0:
            s_last = [s_lists[p][n - 1] for p in range(len(s_lists))]
            y_last = [y_lists[p][n - 1] for p in range(len(y_lists))]
            sy = self._dot(self._flatten_nested(s_last), self._flatten_nested(y_last))
            yy = self._dot(self._flatten_nested(y_last), self._flatten_nested(y_last))
            if yy > 0:
                gamma = max(sy / yy, 1e-10)
        r = self._scale(q, gamma)
        for i in range(k):
            s_i = [s_lists[p][n - k + i] for p in range(len(s_lists))]
            y_i = [y_lists[p][n - k + i] for p in range(len(y_lists))]
            rho_i = rho_lists[0][n - k + i]
            s_flat = self._flatten_nested(s_i)
            y_flat = self._flatten_nested(y_i)
            beta = rho_i * self._dot(y_flat, r)
            r = self._add(r, s_flat, alpha[k - 1 - i] - beta)
        return r

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform a single L-BFGS optimization step.

        The `closure` must zero grads, compute the loss, call
        `loss.backward()`, and return the loss. This matches real PyTorch.
        """
        if closure is None:
            raise RuntimeError("LBFGS.step requires a closure.")
        loss = float(closure())
        self._step_closure_eval_count = 1

        # Converged already (loss is tiny).
        if abs(loss) < 1e-12:
            return loss

        group = self.param_groups[0]
        lr = float(group["lr"])
        max_iter = int(group["max_iter"])
        max_eval = int(group["max_eval"])
        tolerance_grad = float(group["tolerance_grad"])
        tolerance_change = float(group["tolerance_change"])
        history_size = int(group["history_size"])

        params = self._gather_params()
        if not params:
            return loss

        def flat_grad() -> list[float]:
            return self._flatten([p.grad for p in params])

        def set_params_from_flat(flat: list[float]) -> None:
            self._unflatten_to_params(params, flat)

        def current_flat_params() -> list[float]:
            return self._flatten(params)

        def eval_closure() -> float:
            self.zero_grad()
            v = float(closure())
            self._step_closure_eval_count += 1
            return v

        grad_old = flat_grad()
        # Convergence: gradient norm below tolerance.
        if self._norm(grad_old) < tolerance_grad:
            return loss

        # Gather per-param s/y/rho lists.
        s_lists = [self.state[id(p)]["s_list"] for p in params]
        y_lists = [self.state[id(p)]["y_list"] for p in params]
        rho_lists = [self.state[id(p)]["rho_list"] for p in params]

        for _ in range(max_iter):
            if self._step_closure_eval_count >= max_eval:
                break
            # Search direction: d = -H * g
            d = self._two_loop(grad_old, s_lists, y_lists, rho_lists, history_size)
            # Negate for descent direction.
            d = self._scale(d, -1.0)
            # Initial step: lr (real PyTorch scales by lr; H_0 already accounts for gamma).
            step_size = lr

            # Backtracking line search: shrink step_size until loss decreases
            # (with Armijo-style check). We require: f(x + t*d) <= f(x) + c1 * t * g . d
            # with c1 = 1e-4 (matching strong_wolfe's first Wolfe condition).
            c1 = 1e-4
            x = current_flat_params()
            f0 = loss
            g_dot_d = self._dot(grad_old, d)
            # If g_dot_d is positive, this is NOT a descent direction — fail
            # gracefully and break (we can't decrease loss in this direction).
            if g_dot_d >= 0:
                break
            t = step_size
            new_loss = f0
            for _ls in range(max(1, max_eval - self._step_closure_eval_count)):
                x_trial = self._add(x, d, t)
                set_params_from_flat(x_trial)
                new_loss = eval_closure()
                if new_loss <= f0 + c1 * t * g_dot_d:
                    break
                t *= 0.5
                if t < 1e-20:
                    break
            else:
                # No acceptable step found — restore and stop.
                set_params_from_flat(x)
                break

            # Update (s, y, rho) using the *new* gradient at x_trial.
            grad_new = flat_grad()
            s_step = self._scale(self._add(x_trial, x, -1.0), 1.0 / max(t, 1e-20))
            y_step = self._add(grad_new, grad_old, -1.0)
            sy = self._dot(s_step, y_step)
            yy = self._dot(y_step, y_step)
            if sy > 1e-10 and yy > 0:
                rho = 1.0 / sy
                for p, s_list, y_list, rho_list, s_i, y_i in zip(
                    params, s_lists, y_lists, rho_lists, s_step, y_step if False else [y_step]
                ):
                    pass
                # The above loop was a placeholder; do the real per-param append
                # by splitting the flat s_step / y_step by per-param sizes.
                idx = 0
                s_split: list[list[float]] = []
                y_split: list[list[float]] = []
                for p in params:
                    size = 1
                    for s in p._shape:
                        size *= int(s)
                    s_split.append(s_step[idx:idx + size])
                    y_split.append(y_step[idx:idx + size])
                    idx += size
                for p, s_list, y_list, rho_list, s_i, y_i in zip(
                    params, s_lists, y_lists, rho_lists, s_split, y_split
                ):
                    s_list.append(s_i)
                    y_list.append(y_i)
                    rho_list.append(rho)
                    # Cap history to `history_size`.
                    if len(s_list) > history_size:
                        s_list.pop(0)
                        y_list.pop(0)
                        rho_list.pop(0)

            loss = new_loss
            grad_old = grad_new

            # Convergence: loss change or gradient norm small.
            if self._norm(grad_old) < tolerance_grad:
                break
            if abs(f0 - new_loss) < tolerance_change:
                break

        return loss


class ASGD(Optimizer):
    """Averaged Stochastic Gradient Descent."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.01,
        lambd: float = 1e-4,
        alpha: float = 0.75,
        t0: float = 1e6,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = {
            "lr": lr,
            "lambd": lambd,
            "alpha": alpha,
            "t0": t0,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        import torch
        loss = super().step(closure)
        runtime, frame_started, run_js = _begin_runtime_frame()
        with no_grad():
            for group in self.param_groups:
                lr = float(group["lr"])
                lambd = float(group["lambd"])
                alpha = float(group["alpha"])
                t0 = float(group["t0"])
                weight_decay = float(group["weight_decay"])
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if id(p) not in self.state:
                        self.state[id(p)] = {
                            "step": 0,
                            "eta": lr,
                            "mu": 1.0,
                            "ax": torch.zeros_like(p),
                        }
                    state = self.state[id(p)]
                    state["step"] = int(state["step"]) + 1
                    step = state["step"]
                    eta = float(state["eta"])
                    mu = float(state["mu"])
                    if weight_decay != 0:
                        grad = grad.add(p.mul(weight_decay))
                    new_p = p.mul(1.0 - lambd * eta).sub(grad.mul(eta))
                    p._set(new_p)
                    ax = state["ax"]
                    if mu != 1.0:
                        state["ax"] = ax.add(new_p.sub(ax).mul(mu))
                    else:
                        state["ax"] = new_p
                    state["eta"] = lr / ((1.0 + lambd * lr * step) ** alpha)
                    state["mu"] = 1.0 / max(1.0, step - t0)
        _end_runtime_frame(runtime, frame_started, run_js)
        return loss


class Adadelta(Optimizer):
    """Adadelta optimizer."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= rho <= 1.0):
            raise ValueError(f"Invalid rho value: {rho}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = {"lr": lr, "rho": rho, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        import torch
        loss = super().step(closure)
        runtime, frame_started, run_js = _begin_runtime_frame()
        with no_grad():
            for group in self.param_groups:
                lr = float(group["lr"])
                rho = float(group["rho"])
                eps = float(group["eps"])
                weight_decay = float(group["weight_decay"])
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if id(p) not in self.state:
                        self.state[id(p)] = {
                            "step": 0,
                            "square_avg": torch.zeros_like(p),
                            "acc_delta": torch.zeros_like(p),
                        }
                    state = self.state[id(p)]
                    state["step"] = int(state["step"]) + 1
                    if weight_decay != 0:
                        grad = grad.add(p.mul(weight_decay))
                    square_avg = state["square_avg"].mul(rho).add(grad.square().mul(1.0 - rho))
                    state["square_avg"] = square_avg
                    std = square_avg.add(eps).sqrt()
                    acc_delta = state["acc_delta"]
                    delta = acc_delta.add(eps).sqrt().div(std).mul(grad)
                    p._set(p.sub(delta.mul(lr)))
                    state["acc_delta"] = acc_delta.mul(rho).add(delta.square().mul(1.0 - rho))
        _end_runtime_frame(runtime, frame_started, run_js)
        return loss


class Rprop(Optimizer):
    """Resilient backpropagation optimizer."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.01,
        etas: tuple[float, float] = (0.5, 1.2),
        step_sizes: tuple[float, float] = (1e-6, 50),
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 < etas[0] < 1.0 < etas[1]):
            raise ValueError(f"Invalid eta values: {etas[0]}, {etas[1]}")
        defaults = {"lr": lr, "etas": etas, "step_sizes": step_sizes}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        import torch
        loss = super().step(closure)
        runtime, frame_started, run_js = _begin_runtime_frame()
        with no_grad():
            for group in self.param_groups:
                lr = float(group["lr"])
                etaminus, etaplus = group["etas"]
                step_size_min, step_size_max = group["step_sizes"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if id(p) not in self.state:
                        self.state[id(p)] = {
                            "step": 0,
                            "prev": torch.zeros_like(p),
                            "step_size": torch.full_like(p, lr),
                        }
                    state = self.state[id(p)]
                    state["step"] = int(state["step"]) + 1
                    prev = state["prev"]
                    step_size = state["step_size"]
                    prod_sign = grad.mul(prev).sign()
                    ones = torch.full_like(p, 1.0)
                    eta_neg = torch.full_like(p, etaminus).where(prod_sign.lt(0), ones)
                    eta = torch.full_like(p, etaplus).where(prod_sign.gt(0), eta_neg)
                    step_size = step_size.mul(eta).clamp(min=step_size_min, max=step_size_max)
                    grad_used = grad.where(prod_sign.ge(0), torch.zeros_like(p))
                    p._set(p.sub(grad_used.sign().mul(step_size)))
                    state["prev"] = grad_used
                    state["step_size"] = step_size
        _end_runtime_frame(runtime, frame_started, run_js)
        return loss


class Adafactor(Optimizer):
    """Adafactor optimizer (memory-efficient, factored second moments)."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.01,
        beta2_decay: float = -0.8,
        eps: tuple[float | None, float] = (None, 1e-3),
        d: float = 1.0,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if beta2_decay > 0.0:
            raise ValueError(f"beta2_decay should be <= 0 but is: {beta2_decay}")
        if d < 1.0:
            raise ValueError(f"Clipping threshold d should be >= 1 but is: {d}")
        defaults = {
            "lr": lr,
            "beta2_decay": beta2_decay,
            "eps": eps,
            "d": d,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        import torch
        float32_eps = 1.1920928955078125e-07
        loss = super().step(closure)
        runtime, frame_started, run_js = _begin_runtime_frame()
        with no_grad():
            for group in self.param_groups:
                lr = float(group["lr"])
                beta2_decay = float(group["beta2_decay"])
                eps1, eps2 = group["eps"]
                d = float(group["d"])
                weight_decay = float(group["weight_decay"])
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if id(p) not in self.state:
                        self.state[id(p)] = {
                            "step": 0,
                            "row_var": None,
                            "col_var": None,
                            "variance": None,
                        }
                    state = self.state[id(p)]
                    state["step"] = int(state["step"]) + 1
                    step_float = float(state["step"])
                    e1 = float32_eps if eps1 is None else float(eps1)
                    one_minus_beta2_t = step_float ** beta2_decay
                    rho_t = min(lr, 1.0 / (step_float ** 0.5))
                    numel = p.numel
                    alpha = max(eps2, p.norm().item() / (numel ** 0.5)) * rho_t

                    if weight_decay != 0:
                        p._set(p.mul(1.0 - lr * weight_decay))

                    if grad.ndim > 1:
                        shape = list(grad.shape)
                        if state["row_var"] is None:
                            state["row_var"] = torch.zeros(shape[:-1] + [1], dtype=p.dtype)
                            state["col_var"] = torch.zeros(shape[:-2] + [1, shape[-1]], dtype=p.dtype)
                        row_mean = grad.square().mean(dim=-1, keepdim=True)
                        row_var = state["row_var"].lerp(row_mean, one_minus_beta2_t)
                        col_mean = grad.square().mean(dim=-2, keepdim=True)
                        col_var = state["col_var"].lerp(col_mean, one_minus_beta2_t)
                        state["row_var"] = row_var
                        state["col_var"] = col_var
                        var_estimate = row_var.matmul(col_var)
                        var_estimate = var_estimate.div(row_var.mean(dim=-2, keepdim=True).clamp(min=e1))
                    else:
                        if state["variance"] is None:
                            state["variance"] = torch.zeros_like(p)
                        variance = state["variance"].lerp(grad.square(), one_minus_beta2_t)
                        state["variance"] = variance
                        var_estimate = variance

                    update = var_estimate.clamp(min=e1 * e1).rsqrt().mul(grad)
                    denom = max(1.0, update.norm().item() / ((numel ** 0.5) * d))
                    p._set(p.sub(update.mul(alpha / denom)))
        _end_runtime_frame(runtime, frame_started, run_js)
        return loss


def _muon_zeropower_via_newtonschulz(
    grad: Tensor, ns_coefficients: tuple[float, float, float], ns_steps: int, eps: float
) -> Tensor:
    a, b, c = ns_coefficients
    x = grad
    transposed = False
    if x.shape[0] > x.shape[1]:
        x = x.transpose(0, 1)
        transposed = True
    x = x.div(x.norm().clamp(min=eps))
    for _ in range(ns_steps):
        gram = x.matmul(x.transpose(0, 1))
        gram_update = gram.addmm(gram, gram, beta=b, alpha=c)
        x = x.addmm(gram_update, x, beta=a, alpha=1.0)
    if transposed:
        x = x.transpose(0, 1)
    return x


def _muon_adjust_lr(lr: float, adjust_lr_fn: str | None, param_shape) -> float:
    A, B = param_shape[0], param_shape[1]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        adjusted_ratio = math.sqrt(max(1, A / B))
    elif adjust_lr_fn == "match_rms_adamw":
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    else:
        adjusted_ratio = 1.0
    return lr * adjusted_ratio


class Muon(Optimizer):
    """Muon optimizer (momentum orthogonalized via Newton-Schulz, 2D params)."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.001,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (3.4445, -4.775, 2.0315),
        eps: float = 1e-7,
        ns_steps: int = 5,
        adjust_lr_fn: str | None = None,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not (0.0 <= momentum <= 1.0):
            raise ValueError(f"Invalid momentum value: {momentum}")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        import torch
        loss = super().step(closure)
        runtime, frame_started, run_js = _begin_runtime_frame()
        with no_grad():
            for group in self.param_groups:
                lr = float(group["lr"])
                weight_decay = float(group["weight_decay"])
                momentum = float(group["momentum"])
                nesterov = bool(group["nesterov"])
                ns_coefficients = group["ns_coefficients"]
                eps = float(group["eps"])
                ns_steps = int(group["ns_steps"])
                adjust_lr_fn = group["adjust_lr_fn"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if grad.ndim != 2:
                        raise ValueError("Param gradient must be a 2D matrix")
                    if id(p) not in self.state:
                        self.state[id(p)] = {"momentum_buffer": torch.zeros_like(p)}
                    state = self.state[id(p)]
                    buf = state["momentum_buffer"].lerp(grad, 1.0 - momentum)
                    state["momentum_buffer"] = buf
                    update = grad.lerp(buf, momentum) if nesterov else buf
                    update = _muon_zeropower_via_newtonschulz(update, ns_coefficients, ns_steps, eps)
                    adjusted_lr = _muon_adjust_lr(lr, adjust_lr_fn, p.shape)
                    p._set(p.mul(1.0 - lr * weight_decay).sub(update.mul(adjusted_lr)))
        _end_runtime_frame(runtime, frame_started, run_js)
        return loss


class SparseAdam(Optimizer):
    """Adam variant that updates only entries present in the gradient.

    The runtime has no sparse-tensor type, so gradients are treated densely
    and Adam state is updated only where the gradient is non-zero, matching
    ``torch.optim.SparseAdam`` semantics on the equivalent dense tensor.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= betas[0] < 1.0):
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not (0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid beta2 value: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        defaults = {"lr": lr, "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        import torch
        loss = super().step(closure)
        runtime, frame_started, run_js = _begin_runtime_frame()
        with no_grad():
            for group in self.param_groups:
                lr = float(group["lr"])
                beta1, beta2 = group["betas"]
                eps = float(group["eps"])
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if id(p) not in self.state:
                        self.state[id(p)] = {
                            "step": 0,
                            "exp_avg": torch.zeros_like(p),
                            "exp_avg_sq": torch.zeros_like(p),
                        }
                    state = self.state[id(p)]
                    state["step"] = int(state["step"]) + 1
                    step = state["step"]
                    mask = grad.ne(torch.zeros_like(grad))
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    new_exp_avg = exp_avg.add(grad.sub(exp_avg).mul(1.0 - beta1))
                    exp_avg = new_exp_avg.where(mask, exp_avg)
                    new_exp_avg_sq = exp_avg_sq.add(grad.square().sub(exp_avg_sq).mul(1.0 - beta2))
                    exp_avg_sq = new_exp_avg_sq.where(mask, exp_avg_sq)
                    state["exp_avg"] = exp_avg
                    state["exp_avg_sq"] = exp_avg_sq
                    bias_correction1 = 1.0 - beta1 ** step
                    bias_correction2 = 1.0 - beta2 ** step
                    step_size = lr * (bias_correction2 ** 0.5) / bias_correction1
                    denom = exp_avg_sq.sqrt().add(eps)
                    update = exp_avg.div(denom).mul(step_size)
                    update = update.where(mask, torch.zeros_like(p))
                    p._set(p.sub(update))
        _end_runtime_frame(runtime, frame_started, run_js)
        return loss


# Re-export lr_scheduler so `from torch.optim import lr_scheduler` and
# `from torch.optim.lr_scheduler import StepLR` both work.
from . import lr_scheduler  # noqa: E402,F401
from .lr_scheduler import (
    StepLR,  # noqa: E402,F401
    MultiStepLR,  # noqa: E402,F401
    ExponentialLR,  # noqa: E402,F401
    CosineAnnealingLR,  # noqa: E402,F401
    ReduceLROnPlateau,  # noqa: E402,F401
)
