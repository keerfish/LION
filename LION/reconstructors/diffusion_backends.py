
# Code tuning is in process ... 

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

import torch.nn.functional as F


@dataclass
class DummyBackendConfig:
    num_train_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    device: str = "cuda"
    dtype: torch.dtype = torch.float32
    mode: str = "identity"  # "identity" | "gaussian_denoise"

class DummyDDIMBackend:
    """
    A diffusion-backend stub with the same interface as DiffusersDDIMBackend.
    Useful to:
      - debug ReSample loop quickly
      - validate data-consistency behavior without heavy models
    """
    def __init__(self, cfg: DummyBackendConfig):
        self.cfg = cfg
        self._timesteps = None

        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.num_train_timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).to(cfg.device)

    def set_timesteps(self, num_steps: int) -> None:

        ts = torch.linspace(self.cfg.num_train_timesteps - 1, 0, num_steps).long()
        self._timesteps = ts.to(self.cfg.device)

    def timesteps(self) -> torch.Tensor:
        return self._timesteps

    @torch.no_grad()
    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
       
        if self.cfg.mode == "identity":
            return x_t
        elif self.cfg.mode == "gaussian_denoise":
            k = torch.ones((1, 1, 3, 3), device=x_t.device, dtype=x_t.dtype) / 9.0
            x = torch.nn.functional.pad(x_t, (1,1,1,1), mode="reflect")
            return torch.nn.functional.conv2d(x, k)
        else:
            raise ValueError(f"Unknown DummyBackend mode: {self.cfg.mode}")

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, t: torch.Tensor, x0_hat: torch.Tensor) -> torch.Tensor:

        return x0_hat

    @torch.no_grad()
    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
 
        t0 = int(t[0].item())
        alpha_bar = self.alphas_cumprod[t0].to(x0.device, x0.dtype)
        while alpha_bar.dim() < x0.dim():
            alpha_bar = alpha_bar.view(*alpha_bar.shape, 1)
        return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise


@dataclass
class DiffusersBackendConfig:
    model_id: Optional[str] = None
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    use_ddim: bool = True


class DiffusersDDIMBackend:
    """
    Minimal diffusion backend for ReSample using HuggingFace diffusers.

    It expects a diffusers pipeline-like object with:
      - .unet
      - .scheduler (DDIMScheduler recommended)
    or it can construct one from model_id.

    This backend exposes:
      - set_timesteps(num_steps)
      - timesteps()
      - predict_x0(x_t, t)
      - step(x_t, t, x0_hat) -> x_{t-1} proposal
      - add_noise(x0, t, noise) -> x_t
    """

    def __init__(self, cfg: DiffusersBackendConfig, pipeline=None):
        self.cfg = cfg
        self.pipeline = pipeline
        self._timesteps = None

        if self.pipeline is None:
            if cfg.model_id is None:
                raise ValueError("Provide either `pipeline` or `cfg.model_id`.")
            try:
                from diffusers import DDPMPipeline, DDIMScheduler  
            except Exception as e:
                raise ImportError(
                    "diffusers is required for DiffusersDDIMBackend. "
                    "Install with: pip install diffusers accelerate transformers"
                ) from e

            pipe = DDPMPipeline.from_pretrained(cfg.model_id)
            if cfg.use_ddim:
                pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

            self.pipeline = pipe

        self.pipeline = self.pipeline.to(cfg.device)
  
        try:
            self.pipeline = self.pipeline.to(dtype=cfg.dtype)
        except Exception:
            pass

        self.unet = self.pipeline.unet
        self.scheduler = self.pipeline.scheduler

    

    def _model_hw(self) -> tuple[int,int]:
      
        ss = getattr(self.unet.config, "sample_size", None)
        if isinstance(ss, int):
            return ss, ss
        if isinstance(ss, (tuple, list)) and len(ss) == 2:
            return int(ss[0]), int(ss[1])
        
        return -1, -1

    def _to_model_size(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int,int]]:
        H, W = x.shape[-2], x.shape[-1]
        mh, mw = self._model_hw()
        if mh == -1:
            return x, (H, W)
        if (H, W) == (mh, mw):
            return x, (H, W)
        x2 = F.interpolate(x, size=(mh, mw), mode="bilinear", align_corners=False)
        return x2, (H, W)

    def _from_model_size(self, x: torch.Tensor, orig_hw: tuple[int,int]) -> torch.Tensor:
        H, W = orig_hw
        mh, mw = x.shape[-2], x.shape[-1]
        if (mh, mw) == (H, W):
            return x
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)


    def _to_unet_channels(self, x: torch.Tensor) -> torch.Tensor:
       
        in_ch = x.shape[1]
        unet_ch = self.unet.config.in_channels

        if in_ch == unet_ch:
            return x

        if in_ch == 1 and unet_ch == 3:
            return x.repeat(1, 3, 1, 1)

        raise ValueError(
            f"Cannot adapt channels: input has {in_ch}, UNet expects {unet_ch}"
        )


    def _from_unet_channels(self, x: torch.Tensor, out_channels: int) -> torch.Tensor:
        """
        Convert UNet output back to desired channel count.
        For CT, we collapse RGB back to single channel.
        """
        if x.shape[1] == out_channels:
            return x

        if x.shape[1] == 3 and out_channels == 1:
            return x[:, :1]   # or x.mean(dim=1, keepdim=True)

        raise ValueError(
            f"Cannot adapt channels: UNet output has {x.shape[1]}, expected {out_channels}"
        )


    def set_timesteps(self, num_steps: int) -> None:
        self.scheduler.set_timesteps(num_steps)
        self._timesteps = self.scheduler.timesteps

    def timesteps(self) -> torch.Tensor:
        if self._timesteps is None:
            raise RuntimeError("Call set_timesteps(num_steps) first.")
        return self._timesteps

    @torch.no_grad()
    @torch.no_grad()
    def _predict_eps(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        
        p = next(self.unet.parameters())
        x_t = x_t.to(device=p.device, dtype=p.dtype)

        
        orig_channels = x_t.shape[1]
        x_t, orig_hw = self._to_model_size(x_t)

        x_t_unet = self._to_unet_channels(x_t)

        t = t.to(device=p.device, dtype=torch.long)

        out = self.unet(x_t_unet, t)
        


        eps = out.sample if hasattr(out, "sample") else out

        eps = self._from_model_size(eps, orig_hw)
        eps = self._from_unet_channels(eps, orig_channels)
        
        return eps



    @torch.no_grad()
    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Convert epsilon prediction to x0 prediction using scheduler alphas_cumprod.
        """
        eps = self._predict_eps(x_t, t)
        t0 = int(t[0].item())
        idx = (self.scheduler.timesteps == t0).nonzero(as_tuple=False).item()
        alpha_bar = self.scheduler.alphas_cumprod[self.scheduler.timesteps[idx]].to(x_t.device)

        while alpha_bar.dim() < x_t.dim():
            alpha_bar = alpha_bar.view(*alpha_bar.shape, 1)

        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus = torch.sqrt(1.0 - alpha_bar)

        x0 = (x_t - sqrt_one_minus * eps) / (sqrt_alpha_bar + 1e-12)
        return x0

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, t: torch.Tensor, x0_hat: torch.Tensor) -> torch.Tensor:
      
        t0 = int(t[0].item())
        idx = (self.scheduler.timesteps == t0).nonzero(as_tuple=False).item()
        alpha_bar = self.scheduler.alphas_cumprod[self.scheduler.timesteps[idx]].to(x_t.device)

        while alpha_bar.dim() < x_t.dim():
            alpha_bar = alpha_bar.view(*alpha_bar.shape, 1)

        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus = torch.sqrt(1.0 - alpha_bar)

        eps_hat = (x_t - sqrt_alpha_bar * x0_hat) / (sqrt_one_minus + 1e-12)

        out = self.scheduler.step(eps_hat, t0, x_t)
        # out.prev_sample is standard in diffusers
        return out.prev_sample if hasattr(out, "prev_sample") else out[0]

    @torch.no_grad()
    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        p = next(self.unet.parameters())
        orig_channels = x0.shape[1]

        x0 = x0.to(device=p.device, dtype=p.dtype)
        noise = noise.to(device=p.device, dtype=p.dtype)

        x0 = self._to_unet_channels(x0)
        noise = self._to_unet_channels(noise)
        noise, orig_hw = self._to_model_size(noise)

        t0 = int(t[0].item())
        xt = self.scheduler.add_noise(
            x0,
            noise,
            torch.tensor([t0], device=p.device, dtype=torch.long),
        )

        xt = self._from_model_size(xt, orig_hw)


        return self._from_unet_channels(xt, orig_channels)