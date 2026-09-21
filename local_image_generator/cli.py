"""Command-line entry point for one local image at a time."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import secrets
import sys

from .presets import PRESETS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m local_image_generator",
        description="Generate one local image with Qwen Image 2.1 and a style preset.",
    )
    parser.add_argument("--prompt", required=True, help="Subject or scene to generate")
    parser.add_argument("--preset", choices=PRESETS, default="photorealistic")
    parser.add_argument("--output", type=Path, help="PNG path (default: outputs/<preset>-<time>-<seed>.png)")
    parser.add_argument("--seed", type=int, help="Nonnegative seed; random if omitted")
    parser.add_argument("--width", type=int, help="Image width; multiple of 16")
    parser.add_argument("--height", type=int, help="Image height; multiple of 16")
    parser.add_argument("--steps", type=int, help="Denoising steps (default: preset value)")
    parser.add_argument(
        "--offload", choices=("group", "sequential"), default="group",
        help="GPU memory strategy; sequential uses less VRAM but is slower",
    )
    parser.add_argument(
        "--no-postprocess", action="store_true",
        help="Save the raw model output, skipping the dither/pixel-art pass",
    )
    parser.add_argument("--stats-json", type=Path, help="Also write settings and timing/memory stats to this JSON file")
    parser.add_argument("--dry-run", action="store_true", help="Show settings without downloading weights")
    return parser


def format_stats(stats: dict) -> str:
    return "\n".join([
        f"Model load        {stats['model_load_s']:7.1f} s",
        f"Text encoding     {stats['text_encode_s']:7.1f} s",
        f"First step ready  {stats['time_to_first_step_s']:7.1f} s after start of generation",
        f"Denoising         {stats['denoise_s']:7.1f} s ({stats['step_mean_s']:.2f} s/step, {stats['steps_per_s']:.2f} steps/s)",
        f"VAE decode        {stats['vae_decode_s']:7.1f} s",
        f"Time to image     {stats['time_to_image_s']:7.1f} s ({stats['s_per_megapixel']:.1f} s/MP)",
        f"Total             {stats['total_s']:7.1f} s (includes imports and model load)",
        f"Peak VRAM         {stats['peak_vram_reserved_gib']:7.1f} GiB reserved, "
        f"{stats['peak_vram_allocated_gib']:.1f} GiB allocated",
        "Peak RAM          " + (
            f"{stats['peak_ram_gib']:7.1f} GiB" if stats["peak_ram_gib"] is not None else "    n/a"
        ),
    ])


def _dimension(parser: argparse.ArgumentParser, name: str, value: int) -> int:
    if value < 256 or value > 2048 or value % 16:
        parser.error(f"{name} must be a multiple of 16 between 256 and 2048")
    return value


def make_settings(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict:
    if not args.prompt.strip():
        parser.error("--prompt must contain non-whitespace text")
    preset = PRESETS[args.preset]
    width = _dimension(parser, "--width", args.width if args.width is not None else preset.width)
    height = _dimension(parser, "--height", args.height if args.height is not None else preset.height)
    steps = args.steps if args.steps is not None else preset.steps
    if not 1 <= steps <= 100:
        parser.error("--steps must be between 1 and 100")
    seed = args.seed if args.seed is not None else secrets.randbelow(2**32)
    if not 0 <= seed < 2**64:
        parser.error("--seed must be between 0 and 2^64 - 1")
    if args.output is not None:
        output = args.output
        if output.suffix.lower() != ".png":
            parser.error("--output must have a .png extension")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        output = Path("outputs") / f"{preset.name}-{stamp}-{seed}.png"
    return {
        "model": "Qwen/Qwen-Image-2.1",
        "preset": preset.name,
        "prompt": preset.prompt_for(args.prompt),
        "width": width,
        "height": height,
        "steps": steps,
        "seed": seed,
        "offload": args.offload,
        "postprocess": None if args.no_postprocess else preset.postprocess,
        "output": str(output),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = make_settings(args, parser)
    print(json.dumps(settings, indent=2))
    if args.dry_run:
        return 0

    output = Path(settings["output"])
    if output.exists():
        parser.error(f"output already exists: {output}")

    from .runtime import generate_image

    try:
        stats = generate_image(
            prompt=settings["prompt"],
            width=settings["width"],
            height=settings["height"],
            steps=settings["steps"],
            seed=settings["seed"],
            output=output,
            offload=settings["offload"],
            postprocess=settings["postprocess"],
        )
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {output}")
    print(format_stats(stats))
    if args.stats_json is not None:
        args.stats_json.parent.mkdir(parents=True, exist_ok=True)
        args.stats_json.write_text(json.dumps({**settings, "stats": stats}, indent=2) + "\n")
    return 0
