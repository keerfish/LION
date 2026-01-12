# Conditioning method defines how to enforce measurement consistency.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any     # Update the current sample (x_t corresponds to x_{t-1} in DDIM code style), Optional

import torch

__CONDITIONING_METHOD__ = {}

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Conditioning method '{name}' already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper


def get_conditioning_method(name: str, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Conditioning method '{name}' is not defined.")
    return __CONDITIONING_METHOD__[name](**kwargs)

class ConditioningMethod(ABC):
    def __init__(self, model, operator, noise_model="gaussian"):
        self.model = model
        self.operator = operator
        self.noise_model = noise_model
    """
    Expected objects:
       model: LDMWrapper
       operator_fn: Callable[[image_tensor], measurement_tensor]
       noise_model: string or callable describing likelihood
    """

    def grad_and_value(self, x_prev, measurement, operator_adjoint_fn=None):
      
        if not x_prev.requires_grad:
            x_prev = x_prev.detach().requires_grad_(True)

        x_img = self.model.differentiable_decode_first_stage(x_prev)  
        x_gray = x_img.mean(dim=1, keepdim=True)                      
        x_vol = x_gray[:, 0]                                          

        Ax = self.operator(x_vol)
        residual = Ax - measurement
        val = 0.5 * (residual**2).mean()

        grad_vol = operator_adjoint_fn(residual)  
        grad_gray = grad_vol.unsqueeze(1)         

        grad_latent = torch.autograd.grad(
            outputs=x_gray,        
            inputs=x_prev,
            grad_outputs=grad_gray,
            retain_graph=False,
        )[0]

        return grad_latent, val

    @abstractmethod
    def conditioning(
        self,
        x_prev: torch.Tensor,
        x_t: torch.Tensor,
        x_0_hat: torch.Tensor,
        measurement: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        **op_kwargs: Any,
    ):
        pass


@register_conditioning_method("ps")
class PosteriorSampling(ConditioningMethod): # Posterior Sampling step: x_t <- x_t - scale * ∇_x || y - A(D(x0_hat)) ||
    
    def conditioning(
        self,
        x_prev: torch.Tensor,
        x_t: torch.Tensor,
        measurement: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        operator_adjoint_fn=None,
        **op_kwargs: Any,
    ):
        if scale is None:
        
            scale = torch.tensor(0.3, device=x_t.device, dtype=x_t.dtype)

        grad, val = self.grad_and_value(
            x_prev=x_prev,
            measurement=measurement,
            operator_adjoint_fn=operator_adjoint_fn,
            **op_kwargs,
        )

        x_t = x_t - grad * scale
        return x_t, val

