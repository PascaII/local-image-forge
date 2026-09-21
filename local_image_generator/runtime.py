"""Heavy imports live here so CLI help and dry runs need no model dependencies."""

from pathlib import Path


MODEL_ID = "Qwen/Qwen-Image-2.1"


def generate_image(*, prompt: str, width: int, height: int, steps: int,
                   seed: int, output: Path, offload: str) -> None:
    try:
        import torch
        from diffusers import QwenImage21Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Missing model dependencies. Run `uv sync --locked` from the project "
            "directory (see README.md)."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. This sample requires a CUDA-enabled PyTorch "
            "build and an NVIDIA GPU; see the Windows setup in README.md."
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not report BF16 support, which this sample requires.")

    try:
        # Do not move the complete 33 GB pipeline onto the 16 GB GPU.
        pipe = QwenImage21Pipeline.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
        pipe.vae.enable_tiling()
        if offload == "group":
            pipe.enable_group_offload(
                onload_device=torch.device("cuda"),
                offload_device=torch.device("cpu"),
                offload_type="block_level",
                num_blocks_per_group=1,
                use_stream=False,
            )
        else:
            pipe.enable_sequential_cpu_offload(device="cuda")
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "The installed Diffusers version could not configure Qwen Image 2.1 "
            "offloading. Reinstall the current Diffusers source version. "
            "If group offloading failed, retry with --offload sequential."
        ) from exc

    try:
        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                generator=torch.Generator(device="cuda").manual_seed(seed),
                true_cfg_scale=1.0,
            ).images[0]
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "CUDA ran out of memory. Close other GPU applications, try a smaller "
            "--width/--height, or retry with --offload sequential."
        ) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG")
