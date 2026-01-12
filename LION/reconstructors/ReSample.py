from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from LION.reconstructors.ldm_wrapper import LDMWrapper
from LION.reconstructors.ReSampleDDIM import ReSampleDDIM
from LION.reconstructors.conditioning import get_conditioning_method


@dataclass
class ResampleConfig:
    steps: int = 200
    eta: float = 0.0
    use_original_steps: bool = True
    noise_model: str = "gaussian"
    cond_method: str = "ps"          # PosteriorSampling
    sampler_method: str = "resample" 
    log_every_t: int = 100


class ReSample:

    def __init__(
        self,
        model: LDMWrapper,
        operator_fn: Callable[..., torch.Tensor],
        config: Optional[ResampleConfig] = None,
    ):
        self.model = model
        self.operator_fn = operator_fn
        self.cfg = config or ResampleConfig()
        self.sampler = ReSampleDDIM(self.model)

        self.cond = get_conditioning_method(
            self.cfg.cond_method,
            model=self.model,
            operator=self.operator_fn,
            noise_model=self.cfg.noise_model,
        )

    def reconstruct(
        self,
        measurement: torch.Tensor,
        batch_size: int,
        latent_shape: Tuple[int, int, int],
        conditioning=None,
        unconditional_conditioning=None,
        unconditional_guidance_scale: float = 1.0,
        x_T: Optional[torch.Tensor] = None,
        **op_kwargs: Any,
    ):
        """
        Run ReSample posterior sampler in latent space.

        Args:
          measurement: y (measurement domain), torch.Tensor
          batch_size: number of samples
          latent_shape: (C,H,W) latent dims (e.g. (4,64,64) for SD-like VAE)
          conditioning: LDM conditioning (usually None for unconditional)
          op_kwargs: extra kwargs passed to operator_fn (e.g., mask=... for inpainting)

        Returns:
          recon_img: decoded image (B,C,H,W)
          z_final: final latent
          intermediates: sampler intermediates
        """

        measurement_cond_fn = self.cond.conditioning

        z_final, intermediates = self.sampler.posterior_sampler(
            measurement=measurement,
            measurement_cond_fn=measurement_cond_fn,
            operator_fn=self.operator_fn,
            S=self.cfg.steps,
            batch_size=batch_size,
            shape=latent_shape,
            eta=self.cfg.eta,
            ddim_use_original_steps=self.cfg.use_original_steps,
            x_T=x_T,
            log_every_t=self.cfg.log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            cond_method=self.cfg.sampler_method,
            conditioning=conditioning,
            **op_kwargs,
        )

        # Decode final latent to image space
        recon_img = self.model.decode_first_stage(z_final)
        return recon_img, z_final, intermediates
