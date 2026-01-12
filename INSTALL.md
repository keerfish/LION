# INSTALL (Diffusion Extensions)

This repository extends the upstream **LION** framework with diffusion-based reconstruction modules.

**Important** 
It is assumed that the **Python environment has already been created following the original LION installation**.  
This document describes the **additional installation steps (2–5)** required to run the diffusion extensions.

---

## Quick Install 
Run the following commands **inside the existing LION environment**.

1. Install PyTorch (example: CUDA 11.8)
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

2. Install diffusion stack dependencies (strict versions)
```
pip install pytorch-lightning==1.7.7 torchmetrics==0.9.3 omegaconf
```

3. Install taming-transformers (Python and pip might have to be downgraded)
```
git clone https://github.com/CompVis/taming-transformers.git
cd taming-transformers
pip install -e .
cd ..
```

4. Download pretrained checkpoints (this is not the optimal directon)
```
mkdir -p models/ldm
wget https://ommer-lab.com/files/latent-diffusion/ffhq.zip -P models/ldm
unzip models/ldm/ffhq.zip -d LION/models/diffusion/ldm

mkdir -p models/first_stage_models/vq-f4
wget https://ommer-lab.com/files/latent-diffusion/vq-f4.zip -P models/first_stage_models/vq-f4
unzip models/first_stage_models/vq-f4/vq-f4.zip -d models/first_stage_models/vq-f4
```

5. Verify checkpoints
```
ls LION/models/diffusion/ldm/model.ckpt
ls models/first_stage_models/vq-f4/model.ckpt
```

