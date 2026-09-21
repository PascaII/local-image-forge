"""Heavy imports live here so CLI help and dry runs need no model dependencies."""

from pathlib import Path
import platform
import time


MODEL_ID = "Qwen/Qwen-Image-2.1"
GIB = 1024**3


def _peak_process_ram_gib() -> float | None:
    """Peak resident memory of this process (Windows working set or Unix max RSS)."""
    try:
        import psutil
    except ImportError:
        return None
    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)
    if peak is None:
        import resource

        # ru_maxrss is KiB on Linux and bytes on macOS.
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak *= 1 if platform.system() == "Darwin" else 1024
    return peak / GIB


def generate_image(*, prompt: str, width: int, height: int, steps: int,
                   seed: int, output: Path, offload: str,
                   postprocess: str | None = None) -> dict:
    """Generate and save one image; return timing and memory statistics."""
    started = time.perf_counter()
    try:
        import torch
        from diffusers import QwenImage21Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Missing model dependencies. Run `uv sync --locked` from the project "
            "directory (see README.md)."
        ) from exc
    imported = time.perf_counter()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. This sample requires a CUDA-enabled PyTorch "
            "build and an NVIDIA GPU; see the Windows setup in README.md."
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not report BF16 support, which this sample requires.")

    def now() -> float:
        # CUDA work is asynchronous; wait for it so phase boundaries are real.
        torch.cuda.synchronize()
        return time.perf_counter()

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
    loaded = now()

    # The pipeline calls self.encode_prompt, so an instance attribute times it.
    encode_spans = []
    encode_prompt = pipe.encode_prompt

    def timed_encode_prompt(*args, **kwargs):
        begin = now()
        result = encode_prompt(*args, **kwargs)
        encode_spans.append(now() - begin)
        return result

    pipe.encode_prompt = timed_encode_prompt
    step_ends = []

    def on_step_end(_pipe, _index, _timestep, callback_kwargs):
        step_ends.append(now())
        return callback_kwargs

    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.inference_mode():
            call_start = now()
            image = pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                generator=torch.Generator(device="cuda").manual_seed(seed),
                true_cfg_scale=1.0,
                callback_on_step_end=on_step_end,
            ).images[0]
            call_end = now()
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "CUDA ran out of memory. Close other GPU applications, try a smaller "
            "--width/--height, or retry with --offload sequential."
        ) from exc

    if postprocess is not None:
        from .postprocess import POSTPROCESSORS

        image = POSTPROCESSORS[postprocess](image)
    postprocessed = time.perf_counter()

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG")
    saved = time.perf_counter()

    text_encode = sum(encode_spans)
    step_times = [b - a for a, b in zip(step_ends, step_ends[1:])]
    first_step = step_ends[0] - call_start - text_encode
    megapixels = width * height / 1e6
    return {
        "import_s": imported - started,
        "model_load_s": loaded - imported,
        "text_encode_s": text_encode,
        # Closest image-model analogue of "time to first token".
        "time_to_first_step_s": step_ends[0] - call_start,
        "first_step_s": first_step,
        "step_mean_s": sum(step_times) / len(step_times) if step_times else first_step,
        "steps_per_s": (len(step_ends) - 1) / (step_ends[-1] - step_ends[0]) if step_times else 1 / first_step,
        "denoise_s": step_ends[-1] - call_start - text_encode,
        "vae_decode_s": call_end - step_ends[-1],
        "postprocess_s": postprocessed - call_end,
        "save_s": saved - postprocessed,
        "time_to_image_s": saved - call_start,
        "total_s": saved - started,
        "s_per_megapixel": (saved - call_start) / megapixels,
        "peak_vram_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
        "peak_vram_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
        "peak_ram_gib": _peak_process_ram_gib(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
