# The main purpose is to check the ReSample works with CT operators in LION
# This is pipeline check without accessing to any dataset
# It is a demo for debugging or itergration for ReSample

# It is NOT a real CT dataset
# Do NOT tune for benchmark results
# Ground truth is fake (generated)
# CT data is synthetic

import os
import inspect
import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from LION.models.diffusion.model_loader import load_model_from_config
from LION.models.diffusion.ldm.models.diffusion.plms import PLMSSampler  

from LION.CTtools.ct_geometry import Geometry
import LION.CTtools.ct_utils as ct_utils

from LION.reconstructors.ldm_wrapper import LDMWrapper
from LION.reconstructors.ReSampleDDIM import ReSampleDDIM
from LION.reconstructors.conditioning import PosteriorSampling  # or your ReSample conditioning class
from LION.reconstructors.ReSample import ReSample
from LION.reconstructors.ReSample import ResampleConfig


class _NoiserName: # for future use
    def __init__(self, name: str):
        self.__name__ = name


class CTForwardOperator: # there might be something existed in LION, CHECK LATER!
    def __init__(self, lion_ct_op):
        self.A = lion_ct_op 

    def forward(self, img: torch.Tensor, **kwargs) -> torch.Tensor:
        # img dimension is expected to be[B, 1, H, W] or [B, H, W] 
        if img.ndim == 4:
            img = img[:, 0]  
        return self.A(img)
    
    def transpose(self, sino: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.A.adjoint(sino)

    def project(self, data: torch.Tensor, measurement: torch.Tensor, **kwargs) -> torch.Tensor:
        
        return measurement - self.forward(data, **kwargs) # (I - A^T A)Y - A X 


def _to_01(x: torch.Tensor) -> torch.Tensor: # there might be something in utils, CHECK LATER!
    # x in [-1,1] -> [0,1]
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def save_image(path: str, x01: torch.Tensor):
    if x01.ndim == 4:
        x01 = x01[0]
    img = (x01.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
    plt.imsave(path, img)


def plot_sinogram(path: str, sino: torch.Tensor, title: str):
    if sino.ndim == 3:
        s = sino[0]
    else:
        s = sino
    s = s.detach().cpu().numpy()
    plt.figure(figsize=(8, 4))
    plt.imshow(s, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ldm_config_path = "LION/models/diffusion/configs/latent-diffusion/ffhq-ldm-vq-4.yaml"
    ckpt_path = "LION/models/diffusion/ldm/model.ckpt"
    out_dir = "results_resample_ct"
    os.makedirs(out_dir, exist_ok=True)

    # to create fake ground truth
    print("Loading Latent Diffusion model...")
    config = OmegaConf.load(ldm_config_path)
    base_model = load_model_from_config(config, ckpt_path).to(device).eval()

    model = LDMWrapper(base_model).to(device).eval()

    print("Sampling a ground-truth image (unconditional) for the smoke test...")
    plms = PLMSSampler(base_model)
    steps_gt = 50
    batch_size = 1
    latent_shape = (3, 64, 64)

    with torch.no_grad():
        z_gt, _ = plms.sample(
            S=steps_gt,
            batch_size=batch_size,
            shape=latent_shape,
            conditioning=None,
            verbose=True,
        )
        x_gt = base_model.decode_first_stage(z_gt)  # [1,3,256,256] typical for ffhq-ldm-vq-4
        x_gt_01 = _to_01(x_gt)

    # to test dimensions
    print("GT latent shape:", z_gt.shape)
    print("GT image shape :", x_gt.shape) 
    save_image(os.path.join(out_dir, "gt.png"), x_gt_01)

    geo = Geometry.parallel_default_parameters(image_shape=[1, 256, 256])

    A = ct_utils.make_operator(geo)  

    # CT operator in LION expects [B,1,H,W])
    # x_gt_gray = x_gt_01.mean(dim=1, keepdim=True).to(device)  # [1,1,256,256]
    x_gt_gray = x_gt_01.mean(dim=1).to(device)  # [1, 256, 256]

    with torch.no_grad():
        y_clean = A(x_gt_gray)  # sinogram shape shll be [1, n_angles, n_det]
    print("Clean sinogram shape:", y_clean.shape)

    plot_sinogram(os.path.join(out_dir, "sinogram_clean.png"), y_clean, "Clean sinogram")

    I0 = 2_000   # higher number correspondes to less Poisson noise
    sigma = 0.0  # set >0 is to add Gaussian noise 
    cross_talk = 0.0

    with torch.no_grad():
        y_noisy = ct_utils.sinogram_add_noise(
            y_clean,
            I0=I0,
            sigma=sigma,
            cross_talk=cross_talk,
            flat_field=None,
            dark_field=None,
            enable_gradients=False,
        )

    print("Noisy sinogram shape:", y_noisy.shape)
    plot_sinogram(os.path.join(out_dir, "sinogram_noisy.png"), y_noisy, f"Noisy sinogram (I0={I0})")

  
    ct_forward_op = CTForwardOperator(A)

    def operator_fn(img_01_gray: torch.Tensor) -> torch.Tensor:
        return ct_forward_op.forward(img_01_gray)

    cfg = ResampleConfig(
        steps=200,
        eta=0.0,
        noise_model="poisson",
        cond_method="ps",
        sampler_method="resample",
        log_every_t=50,
    )

    reconstructor = ReSample(
        model=model,
        operator_fn=A,
        config=cfg,
    )

    print("Running ReSample DDIM...")

    def operator_fn(img_01_gray: torch.Tensor) -> torch.Tensor:
        return ct_forward_op.forward(img_01_gray)

    reconstruct_kwargs = {
        "measurement": y_noisy,
        "batch_size": 1,
        "latent_shape": (3, 64, 64),
        "operator_fn": operator_fn,
        "steps": 1000,        # DDPM steps in latent diffusion
        "ddim_steps": 200,    # to use DDIM subset steps
        "ddim_eta": 0.0,
        "verbose": True,
        "log_every_t": 50,
    }

    sig = inspect.signature(reconstructor.reconstruct)
    filtered = {k: v for k, v in reconstruct_kwargs.items() if k in sig.parameters}

    out = reconstructor.reconstruct(**filtered)

    if isinstance(out, dict):
        recon_img = out.get("recon_img", None)
        z_final = out.get("z_final", None)
        # intermediates = out.get("intermediates", None)
    else:
      
        if len(out) == 3:
            recon_img, z_final
        elif len(out) == 2:
            recon_img, z_final = out
        else:
            recon_img = out
            z_final = None
        
    if recon_img is None and z_final is not None:
        with torch.no_grad():
            recon_img = model.decode_first_stage(z_final)

    if recon_img is None:
        raise RuntimeError("Reconstruction returned no image/latent. Check ReSample.reconstruct() return values.")

    if recon_img.min() < 0:
        recon_01 = _to_01(recon_img)
    else:
        recon_01 = recon_img.clamp(0, 1)

    save_image(os.path.join(out_dir, "recon.png"), recon_01)
    print("Saved outputs to:", out_dir)
    gt_gray = x_gt_01.mean(dim=1, keepdim=True)
    rec_gray = recon_01.mean(dim=1, keepdim=True)

    plt.figure(figsize=(9, 3))
    plt.subplot(1, 3, 1); plt.imshow(gt_gray[0,0].cpu(), cmap="gray"); plt.title("GT"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.imshow(rec_gray[0,0].cpu(), cmap="gray"); plt.title("Reconstruction"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.imshow((gt_gray-rec_gray).abs()[0,0].cpu(), cmap="magma"); plt.title("|GT-Rec|"); plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compare_gray.png"))
    plt.close()


if __name__ == "__main__":
    main()

