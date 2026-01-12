
import os
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from LION.models.diffusion.model_loader import load_model_from_config
from LION.models.diffusion.ldm.models.diffusion.plms import PLMSSampler

from LION.CTtools.ct_geometry import Geometry
import LION.CTtools.ct_utils as ct_utils

from LION.reconstructors.PnP import PnP
from LION.models.diffusion.priors.diffusion_prior import DiffusionPriorFn


def to_01(x):
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def save_gray(path, x, title=""):
    if x.ndim == 4:
        x = x[0, 0]
    elif x.ndim == 3:
        x = x[0]
    x = x.detach().cpu()
    plt.imshow(x, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.savefig(path)
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = "results_pnp_diffusion"
    os.makedirs(out_dir, exist_ok=True)

    # Load diffusion model
  
    config = OmegaConf.load(
        "LION/models/diffusion/configs/latent-diffusion/ffhq-ldm-vq-4.yaml"
    )
    diffusion_model = load_model_from_config(
        config, "LION/models/diffusion/ldm/model.ckpt"
    ).to(device)

    # Sample ground truth

    sampler = PLMSSampler(diffusion_model)
    with torch.no_grad():
        z, _ = sampler.sample(
            S=50, batch_size=1, shape=(3, 64, 64), conditioning=None
        )
        x_gt = diffusion_model.decode_first_stage(z)
        x_gt = to_01(x_gt).mean(dim=1)

    save_gray(os.path.join(out_dir, "gt.png"), x_gt, "Ground Truth")

    # CT forward model

    geo = Geometry.parallel_default_parameters(image_shape=[1, 256, 256])
    A = ct_utils.make_operator(geo)

    with torch.no_grad():
        y = A(x_gt.to(device))
        y = ct_utils.sinogram_add_noise(y, I0=2000)


    # Diffusion prior
  
    prior = DiffusionPriorFn(
        diffusion_model=diffusion_model,
        device=device,
        ddim_steps=50,
        denoise_steps=40,   # Impostant to set
        eta=0.0,
        input_range="01",
        use_fp16=False,
        micro_denoise=False,
    )

    # PnP-ADMM
    
    pnp = PnP(A, prior, algorithm="ADMM")

    x_rec = pnp.reconstruct_sample(
        y,
        prog_bar=True,
        eta=300,          # IMPORTANT
        max_iter=20,
        cg_max_iter=50,
        cg_tol=1e-6,
    )

    save_gray(os.path.join(out_dir, "recon.png"), x_rec, "PnP + Diffusion")


if __name__ == "__main__":
    main()
