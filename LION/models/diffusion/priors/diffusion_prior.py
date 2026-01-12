# Diffusion-based prior adapter for Plug-and-Play (PnP).

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional, Tuple, Any

import torch

from LION.models.diffusion.ldm.models.diffusion.ddim import DDIMSampler


class DiffusionPriorFn:
    def __init__(
        self,
        diffusion_model,
        device: torch.device | str,
        *,
        ddim_steps: int = 50,
        denoise_steps: int = 6,
        eta: float = 0.0,
        input_range: str = "01",
        micro_denoise: bool = True,
        use_fp16: bool = False,
        debug_checks: bool = False,
    ):
        self.model = diffusion_model.to(device).eval()
        self.device = torch.device(device)

        self.ddim_steps = int(ddim_steps)
        self.denoise_steps = int(denoise_steps)
        self.eta = float(eta)
        self.input_range = str(input_range)
        self.micro_denoise = bool(micro_denoise)
        self.use_fp16 = bool(use_fp16)
        self.debug_checks = bool(debug_checks)

        self.sampler = DDIMSampler(self.model)
        self.sampler.make_schedule(ddim_num_steps=self.ddim_steps, ddim_eta=self.eta, verbose=False)

        self.ddim_timesteps = self.sampler.ddim_timesteps

        self.model_parameters = SimpleNamespace(use_noise_level=True)

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def unnormalise(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _to_model_range(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_range == "01":
            return (2.0 * x - 1.0).clamp(-1.0, 1.0)
        if self.input_range == "-11":
            return x.clamp(-1.0, 1.0)
        raise ValueError(f"input_range must be '01' or '-11', got {self.input_range}")

    def _from_model_range(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_range == "01":
            return ((x + 1.0) / 2.0).clamp(0.0, 1.0)
        if self.input_range == "-11":
            return x
        raise ValueError(f"input_range must be '01' or '-11', got {self.input_range}")


    def _to_b1hw(self, x: torch.Tensor) -> Tuple[torch.Tensor, str]:
        if x.ndim == 2:
            return x.unsqueeze(0).unsqueeze(0), "HW"
        if x.ndim == 3:
            return x.unsqueeze(1), "BHW"
        if x.ndim == 4:
            return x, "B1HW"
        raise ValueError(f"Expected x with ndim 2/3/4, got {x.ndim} (shape={tuple(x.shape)})")

    def _restore_layout(self, x_b1hw: torch.Tensor, tag: str) -> torch.Tensor:
        if tag == "HW":
            return x_b1hw[0, 0]
        if tag == "BHW":
            return x_b1hw[:, 0]
        if tag == "B1HW":
            return x_b1hw
        raise ValueError(f"Unknown layout tag: {tag}")

    def _noise_level_to_tstart(self, noise_level: Optional[float]) -> int:
        nT = len(self.ddim_timesteps)
        frac = 0.6 if noise_level is None else float(noise_level)
        frac = max(0.0, min(1.0, frac))

        t_start = int(frac * (nT - 1))
        t_start = max(0, min(t_start, nT - 1))

        if self.micro_denoise:
            t_start = min(t_start, max(0, self.denoise_steps))

        return t_start

    # There is still no stochastic_encode

    def _noise_latent_ddpm(self, z0: torch.Tensor, t_ddpm: int, noise: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.model, "alphas_cumprod"):
            raise AttributeError("Model missing alphas_cumprod; cannot noise latent safely.")

        alpha_bar = self.model.alphas_cumprod
        if alpha_bar.device != z0.device:
            alpha_bar = alpha_bar.to(z0.device)

        T = int(alpha_bar.shape[0])
        if t_ddpm < 0 or t_ddpm >= T:
            raise ValueError(f"DDPM timestep out of range: t_ddpm={t_ddpm}, T={T}")

        a = alpha_bar[int(t_ddpm)]
        sqrt_a = torch.sqrt(a).view(1, 1, 1, 1)
        sqrt_one_minus = torch.sqrt(1.0 - a).view(1, 1, 1, 1)
        return sqrt_a * z0 + sqrt_one_minus * noise


    def _ddim_decode_robust(self, x_latent: torch.Tensor, t_start: int) -> torch.Tensor:

        x = x_latent

        for index in range(int(t_start), -1, -1):
            ts_val = int(self.ddim_timesteps[index])
            ts = torch.full((x.shape[0],), ts_val, device=x.device, dtype=torch.long)

            out: Any = self.sampler.p_sample_ddim(
                x,
                cond=None,
                t=ts,
                index=index,
                use_original_steps=False,
                unconditional_guidance_scale=1.0,
                unconditional_conditioning=None,
            )

            if isinstance(out, (tuple, list)):
                x = out[0]
            else:
                x = out

            if not torch.isfinite(x).all():
                raise FloatingPointError("Non-finite latent during DDIM decode loop")

        return x

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, noise_level: Optional[float] = None) -> torch.Tensor:
        # here is to test input dimensions
        x_b1hw, tag = self._to_b1hw(x)
        x_b1hw = x_b1hw.to(self.device)
        x_b1hw = torch.nan_to_num(x_b1hw, nan=0.0, posinf=1.0, neginf=0.0)

        if self.input_range == "01":
            x_b1hw = x_b1hw.clamp(0.0, 1.0)
        else:
            x_b1hw = x_b1hw.clamp(-1.0, 1.0)

        x_b1hw = x_b1hw.float()

        x_b3hw = x_b1hw.repeat(1, 3, 1, 1) # to make grayscale to RGB 
        x_b3hw = self._to_model_range(x_b3hw)

        enc = self.model.encode_first_stage(x_b3hw)
        if torch.is_tensor(enc):
            z0 = enc
        else:
            z0 = self.model.get_first_stage_encoding(enc)
        z0 = z0.float()

        t_start = self._noise_level_to_tstart(noise_level)
        t_ddpm = int(self.ddim_timesteps[t_start])

        noise = torch.randn_like(z0)
        try:
            zt = self._noise_latent_ddpm(z0, t_ddpm=t_ddpm, noise=noise)
        except Exception:
            return x.clone()

        try:
            if self.device.type == "cuda" and self.use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    z_denoised = self._ddim_decode_robust(zt, t_start=t_start)
            else:
                z_denoised = self._ddim_decode_robust(zt, t_start=t_start)
        except Exception:
            return x.clone()

        x_hat = self.model.decode_first_stage(z_denoised)  
        x_hat = self._from_model_range(x_hat)

        x_gray = x_hat.mean(dim=1, keepdim=True)
        x_gray = torch.nan_to_num(x_gray, nan=0.0, posinf=1.0, neginf=0.0)

        if self.input_range == "01":
            x_gray = x_gray.clamp(0.0, 1.0)
        else:
            x_gray = x_gray.clamp(-1.0, 1.0)

        return self._restore_layout(x_gray, tag)
