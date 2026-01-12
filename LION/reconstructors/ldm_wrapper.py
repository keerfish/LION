# This small code wraps a LatentDiffusion model into a small, stable interface.

# Because DDIMSampler expects a model with:
   # num_timesteps
   # betas, alphas_cumprod, alphas_cumprod_prev
   # device
   # apply_model(x, t, cond)
# And ReSample additionally needs:
   # decode_first_stage / differentiable_decode_first_stage
   # encode_first_stage

# This wrapper simply forwards those calls to the underlying Latent Diffusion.
 
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn


@dataclass
class LDMWrapperConfig:
    name: str = "ldm_wrapper"


class LDMWrapper(nn.Module):
    
    def __init__(self, ldm_model: nn.Module, config: Optional[LDMWrapperConfig] = None):
        super().__init__()
        self.ldm = ldm_model
        self.config = config or LDMWrapperConfig()
        self.ldm.eval()

 
    @property
    def device(self) -> torch.device:
        return self.ldm.device

    @property
    def num_timesteps(self) -> int:
        return int(self.ldm.num_timesteps)

    @property
    def betas(self) -> torch.Tensor:
        return self.ldm.betas

    @property
    def alphas_cumprod(self) -> torch.Tensor:
        return self.ldm.alphas_cumprod

    @property
    def alphas_cumprod_prev(self) -> torch.Tensor:
        return self.ldm.alphas_cumprod_prev

    @property
    def sqrt_one_minus_alphas_cumprod(self) -> torch.Tensor:
        return self.ldm.sqrt_one_minus_alphas_cumprod


    @torch.no_grad()
    def apply_model(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[Union[torch.Tensor, Dict[str, Any], list]] = None,
        return_ids: bool = False,
    ) -> torch.Tensor:
        return self.ldm.apply_model(x_noisy, t, cond, return_ids=return_ids)


    @torch.no_grad()
    def decode_first_stage(self, z: torch.Tensor) -> torch.Tensor:
        return self.ldm.decode_first_stage(z)

    def differentiable_decode_first_stage(self, z: torch.Tensor) -> torch.Tensor:
        return self.ldm.differentiable_decode_first_stage(z)

    @torch.no_grad()
    def encode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        return self.ldm.encode_first_stage(x)

    @torch.no_grad()
    def get_first_stage_encoding(self, enc: torch.Tensor) -> torch.Tensor:
        return self.ldm.get_first_stage_encoding(enc)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.ldm.to(*args, **kwargs)
        return self
