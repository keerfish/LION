
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from LION.models.diffusion.ldm.models.diffusion.ddim import DDIMSampler


class ReSampleDDIM(DDIMSampler):
    """
    Key external dependencies:
      - model must provide:
          apply_model, decode_first_stage, differentiable_decode_first_stage,
          encode_first_stage, get_first_stage_encoding,
          and diffusion buffers.
      - measurement_cond_fn: callable implementing conditioning (the posterior step)
      - operator_fn: callable A(x) used for data consistency and optimization
    """

    def posterior_sampler(
        self,
        measurement: torch.Tensor,
        measurement_cond_fn: Callable[..., Tuple[torch.Tensor, torch.Tensor]],
        operator_fn: Callable[..., torch.Tensor],
        S: int,
        batch_size: int,
        shape: Tuple[int, int, int],
        eta: float = 0.0,
        ddim_use_original_steps: bool = True,
        x_T: Optional[torch.Tensor] = None,
        log_every_t: int = 100,
        unconditional_guidance_scale: float = 1.0,
        unconditional_conditioning=None,
        cond_method: str = "resample",
        conditioning=None,
        **kwargs,
    ):
        
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=False)
        C, H, W = shape
        size = (batch_size, C, H, W)

        if cond_method is None or cond_method == "resample":
            return self.resample_sampling(
                measurement=measurement,
                measurement_cond_fn=measurement_cond_fn,
                operator_fn=operator_fn,
                cond=conditioning,
                shape=size,
                x_T=x_T,
                ddim_use_original_steps=ddim_use_original_steps,
                log_every_t=log_every_t,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                **kwargs,
            )

        raise ValueError(f"cond_method='{cond_method}' not supported")

    def resample_sampling(
        self,
        measurement: torch.Tensor,
        measurement_cond_fn: Callable[..., Tuple[torch.Tensor, torch.Tensor]],
        operator_fn: Callable[..., torch.Tensor],
        cond,
        shape: Tuple[int, int, int, int],
        inter_timesteps: int = 5,
        x_T: Optional[torch.Tensor] = None,
        ddim_use_original_steps: bool = True,
        log_every_t: int = 100,
        unconditional_guidance_scale: float = 1.0,
        unconditional_conditioning=None,
        enable_time_travel: bool = True,
        enable_pixel_opt: bool = False,
        enable_latent_opt: bool = True,
        **op_kwargs: Any,
    ):
     
        device = self.model.betas.device
        b = shape[0]

        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T
   
        img = img.requires_grad_(True) # We need gradients, Check Later for the robustness

        timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)

        alphas = self.model.alphas_cumprod if ddim_use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if ddim_use_original_steps else self.ddim_alphas_prev
        betas = self.model.betas

        intermediates = {"x_inter": [img.detach().clone()], "pred_x0": []}
        iterator = tqdm(time_range, desc="ReSample DDIM", total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            a_t = torch.full((b, 1, 1, 1), float(alphas[index]), device=device)
            a_prev = torch.full((b, 1, 1, 1), float(alphas_prev[index]), device=device)
            _b_t = torch.full((b, 1, 1, 1), float(betas[index]), device=device)

            x_prev, pred_x0, pseudo_x0 = self.p_sample_ddim(
                img,
                cond,
                ts,
                index=index,
                use_original_steps=ddim_use_original_steps,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
            )
    
            x_corr, _val = measurement_cond_fn(
                x_prev=img,            # x_t in diffusion model
                x_t=x_prev,            # x_{t-1} output in diffusion model
                measurement=measurement,
                operator_adjoint_fn=operator_fn.adjoint,
                scale=a_t * 0.5,        # typical scaling
                **op_kwargs,
            )
            img = x_corr

            if enable_time_travel and index > 0 and (index % 10 == 0):
                x_t_backup = img.detach().clone()
                
                for k in range(min(inter_timesteps, 3)): # move "forward" a few steps, can be others
                    
                    next_i = min(i + k + 1, total_steps - 1)
                    if ddim_use_original_steps:
                        step_ = (total_steps - next_i - 1)
                    else:
                        step_ = int(list(np.flip(self.ddim_timesteps))[next_i])
                    ts_ = torch.full((b,), step_, device=device, dtype=torch.long)
                    idx_ = total_steps - next_i - 1

                    img, _pred, pseudo_x0 = self.p_sample_ddim(
                        img,
                        cond,
                        ts_,
                        index=idx_,
                        use_original_steps=ddim_use_original_steps,
                        unconditional_guidance_scale=unconditional_guidance_scale,
                        unconditional_conditioning=unconditional_conditioning,
                    )

                # latent optimization, Not have to do it
                if enable_latent_opt:
                    pseudo_x0_opt, _ = self.latent_optimization(
                        measurement=measurement,
                        z_init=pseudo_x0.detach(),
                        operator_fn=operator_fn,
                        **op_kwargs,
                    )
                else:
                    pseudo_x0_opt = pseudo_x0.detach()

                # pixel optimization
                if enable_pixel_opt:
                    px = self.model.decode_first_stage(pseudo_x0_opt).detach()
                    px_opt = self.pixel_optimization(measurement, px, operator_fn, **op_kwargs)
                    pseudo_x0_opt = self.model.encode_first_stage(px_opt)
                    
                    if not torch.is_tensor(pseudo_x0_opt):
                        pseudo_x0_opt = self.model.get_first_stage_encoding(pseudo_x0_opt)

                # stochastic resample back toward x_t_backup
                sigma = 40.0 * (1 - a_prev) / (1 - a_t + 1e-8) * (1 - a_t / (a_prev + 1e-8))
                img = self.stochastic_resample(
                    pseudo_x0=pseudo_x0_opt,
                    x_t=x_t_backup,
                    a_t=a_prev,
                    sigma=sigma,
                )
                img = img.requires_grad_(True)

            if (index % log_every_t == 0) or (index == total_steps - 1):
                intermediates["x_inter"].append(img.detach().clone())
                intermediates["pred_x0"].append(pred_x0.detach().clone())

        return img.detach(), intermediates


    def pixel_optimization(
        self,
        measurement: torch.Tensor,
        x_prime: torch.Tensor,
        operator_fn: Callable[..., torch.Tensor],
        eps: float = 1e-3,
        max_iters: int = 500,
        lr: float = 1e-2,
        **op_kwargs: Any,
    ) -> torch.Tensor:
        loss_fn = torch.nn.MSELoss()
        opt_var = x_prime.detach().clone().requires_grad_(True)
        optimizer = torch.optim.AdamW([opt_var], lr=lr)

        meas = measurement.detach()
        for _ in range(max_iters):
            optimizer.zero_grad()
            # out = loss_fn(meas, operator_fn(opt_var, **op_kwargs))
            pred = opt_var.mean(dim=1) if opt_var.ndim == 4 else opt_var
            out = loss_fn(meas, operator_fn(pred, **op_kwargs))

            out.backward()
            optimizer.step()
            if out.detach() < eps**2:
                break
        return opt_var.detach()

    def latent_optimization(
        self,
        measurement: torch.Tensor,
        z_init: torch.Tensor,
        operator_fn: Callable[..., torch.Tensor],
        eps: float = 1e-3,
        max_iters: int = 200,
        lr: float = 5e-3,
        **op_kwargs: Any,
    ):
        if not z_init.requires_grad:
            z = z_init.detach().clone().requires_grad_(True)
        else:
            z = z_init

        # loss_fn = torch.nn.MSELoss()
        # optimizer = torch.optim.AdamW([z], lr=lr)

        meas = measurement.detach()
        init_loss = None

        for it in range(max_iters):
            # optimizer.zero_grad()

            x_img = self.model.differentiable_decode_first_stage(z)
            # out = loss_fn(meas, operator_fn(x_img, **op_kwargs))
            # CT uses grayscale, Check LATER
            x_gray = x_img.mean(dim=1, keepdim=True) # [B,1,H,W]
            x_vol  = x_gray[:, 0] # [B,H,W]

            Ax = operator_fn(x_vol, **op_kwargs) # [B,angles,det]
            res = Ax - meas

            out = 0.5 * (res**2).mean()

            if it == 0:
                init_loss = out.detach().clone()
            # out.backward()
            # optimizer.step()

            grad_vol = operator_fn.adjoint(res) # [B,H,W]
            grad_gray = grad_vol.unsqueeze(1) # [B,1,H,W]

            # Pull gradient back to latent
            grad_z = torch.autograd.grad(
                outputs=x_gray,
                inputs=z,
                grad_outputs=grad_gray,
                retain_graph=False,
            )[0]

            with torch.no_grad():
                z -= lr * grad_z

            if out.detach() < eps**2: # Optional stopping
                break

        return z.detach(), init_loss

    def stochastic_resample(self, pseudo_x0, x_t, a_t, sigma):
        device = pseudo_x0.device
        noise = torch.randn_like(pseudo_x0, device=device)
        denom = (sigma + 1 - a_t + 1e-8)
        term = (sigma * a_t.sqrt() * pseudo_x0 + (1 - a_t) * x_t) / denom
        var = torch.sqrt(1.0 / (1.0 / (sigma + 1e-8) + 1.0 / (1 - a_t + 1e-8)))
        return term + noise * var



