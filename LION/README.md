# LION – Diffusion Extensions for CT Reconstruction

This repository extends the upstream **LION** framework with
**diffusion-based reconstruction and re-sampling methods** for CT inverse problems.

The code builds on latent diffusion models and integrates them into the LION
reconstruction pipeline for posterior sampling (ReSample) and plug-and-play (PnP) inference. 
The upstream repository is [LION](https://github.com/CambridgeCIA/LION)

---

## Overview

Compared to the original LION framework, this repository adds:

- **Diffusion-based reconstructors**
  - ReSample and ReSampleDDIM methods for posterior sampling
  - Explicit diffusion prior interface (`diffusion_prior.py`)
  - Diffusion denoising for the existed PnP for CT reconstruction

- **Latent diffusion integration**
  - Support for latent diffusion priors and first-stage autoencoders
  - Wrappers for diffusion models and sampling interfaces

- **Executable demo scripts**
  - Two examples demonstrating diffusion-based CT reconstruction

Pretrained diffusion and autoencoder models are **not included** in this repository
and must be downloaded separately.

---

## Repository Structure

```text
LION/
├── models/
│   ├── diffusion/
│   │   ├── priors/
│   │   │   └── diffusion_prior.py
│   │   └── ...
│   └── ...
├── reconstructors/
│   ├── ReSample.py
│   ├── ReSampleDDIM.py
│   ├── conditioning.py
│   ├── diffusion_backends.py
│   └── ldm_wrapper.py
├── demos/
│   ├── d05_resample_smoke.py
│   └── d06_pnp_diffusion_denoisor.py
└── ...
```

## Acknowledgments
The code and idea are heavily adopted from [Resample](https://github.com/soominkwon/resample). The code is inferred on FFHQ. The model can be downloaded from  [Latent Diffusion Model](https://github.com/CompVis/latent-diffusion/tree/main) and modified the checkpoint paths for other datasets and models
