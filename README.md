## diffusion-inpainting-pytorch

Implementation of an AI-based image inpainting system using Diffusion Models in PyTorch. Built for the AARUUSH '26 AI/ML task.

## Install

```bash
$ pip install -r requirements.txt
```

## Usage

```bash
$ python experiment_runner.py
```

## Features

- Generates rectangular, irregular, and multi-region masks automatically
- Uses Stable Diffusion for high-fidelity reconstruction
- Text-guided inpainting support
- Evaluates with PSNR, SSIM, and LPIPS metrics
