# Local image generation with Qwen Image 2.1

A small, local text-to-image CLI using [Qwen Image 2.1](https://github.com/QwenLM/Qwen-Image-2.1). Describe a subject and choose an ASCII-styled image, anime illustration, or photorealistic photograph. Images are generated on your hardware and saved as PNGs; there is no hosted inference API or per-image API fee.

This sample targets **Windows with an NVIDIA RTX 5070 Ti (16 GB VRAM) and 64 GB system RAM**. It uses BF16 weights and CPU group offloading so the full model does not have to fit in VRAM. The model download is approximately **33 GB**, and loading it needs additional disk and memory headroom. A 24 GB M3 MacBook Air is not a practical target for this unquantized sample. The desktop configuration is based on the published model sizes and Diffusers offloading guidance; it has not yet been run on that exact machine.

## Windows setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), Git, and a recent NVIDIA driver. uv will install Python 3.12 if needed. In PowerShell, from this repository:

```powershell
uv sync --locked
uv run --locked python -c "import torch; from diffusers import QwenImage21Pipeline; print('CUDA:', torch.cuda.is_available(), 'GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

The project selects PyTorch's CUDA 12.8 wheels on Windows through uv's package index configuration. Diffusers is locked to a Git commit because this new pipeline requires its source version. `uv.lock` records the resolved dependencies and should be committed with the project. The model weights download from Hugging Face on the first real generation and are cached outside this repository by default. If CUDA reports `False`, check the NVIDIA driver and the [PyTorch compatibility guidance](https://pytorch.org/get-started/locally/).

## Generate an image

Preview the effective prompt and settings without loading or downloading the model:

```powershell
uv run --locked local-image --preset anime --prompt "a fox reading under a cherry tree" --seed 42 --dry-run
```

Run the first desktop smoke test:

```powershell
uv run --locked local-image --preset anime --prompt "a fox reading under a cherry tree" --seed 42 --width 1024 --height 1024 --output outputs\fox-anime.png
```

Other presets:

```powershell
uv run --locked local-image --preset ascii --prompt "a mountain cabin beneath the moon"
uv run --locked local-image --preset photorealistic --prompt "a red bicycle outside a Zurich cafe"
```

`ascii` creates a **PNG that looks like ASCII art**, not a text file. Presets use 1024 × 1024 (ASCII), 896 × 1152 (anime), or 1152 × 896 (photorealistic), all at 40 denoising steps with the model's recommended no-guidance setting. `--width`, `--height`, `--steps`, and `--seed` override the preset defaults. Dimensions must be multiples of 16 between 256 and 2048. Higher resolutions need more memory and time. The CLI refuses an output path that already exists.

If group offloading fails on the desktop, try the lower-memory, slower path:

```powershell
uv run --locked local-image --preset anime --prompt "a fox reading under a cherry tree" --seed 42 --width 1024 --height 1024 --offload sequential --output outputs\fox-anime-sequential.png
```

For CUDA out-of-memory errors, close other GPU applications and try `--width 768 --height 768`. CPU offloading can be substantially slower than running a model entirely in VRAM. The sample generates one image per process and does not use an additional prompt-rewriting model, since that model would further increase resource use.

## License and sources

This repository's sample code is MIT licensed. **The Qwen Image 2.1 model is separate and uses the [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE), which permits noncommercial research and evaluation and requires a separate license for commercial use.** The model weights are not included here.

Implementation settings follow [Qwen's quick start](https://github.com/QwenLM/Qwen-Image-2.1), the [Diffusers pipeline reference](https://huggingface.co/docs/diffusers/main/api/pipelines/qwenimage21), and [Diffusers memory guidance](https://huggingface.co/docs/diffusers/optimization/memory).
