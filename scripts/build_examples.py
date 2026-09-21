"""Generate one benchmarked example per preset for the README.

Run from the repository root:

    uv run --locked python scripts/build_examples.py

Each preset runs in a fresh process, like a normal CLI call, so every run pays
the model load. Results go to docs/examples/: full-size PNGs, 256-px README
thumbnails, per-run JSON, and benchmarks.json with system info. Existing
images are skipped, so an interrupted run can be resumed.
"""

import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time

from PIL import Image

from local_image_generator.presets import PRESETS

OUT = Path("docs/examples")
THUMBS = OUT / "thumbs"
THUMB_WIDTH = 256
SEED = 7

EXAMPLES = {
    "photorealistic": "an elderly fisherman mending nets on a harbor pier at dawn",
    "anime": "a fox reading under a cherry tree",
    "cartoon": "a grumpy cat chef flipping pancakes in a tiny kitchen",
    "ascii": "a sailing ship on stormy waves",
    "dither": "an astronaut floating above the moon",
    "pixel-art": "a cozy village tavern on a snowy night",
    "watercolor": "a Venice canal with gondolas in the morning",
    "oil-painting": "a still life of pears, a copper jug, and a candle",
    "pencil-sketch": "an old steam locomotive waiting at a station",
    "ukiyo-e": "Mount Fuji behind a fishing village in autumn",
    "comic": "a superhero landing on a rooftop in the rain",
    "isometric": "a tiny ramen shop on a street corner",
    "papercraft": "a whale swimming above a coral reef",
    "low-poly": "a fox in an autumn forest",
}

# Examples picked from earlier runs keep the settings they were made with.
OVERRIDES = {
    "anime": {"seed": 42, "width": 1024, "height": 1024},
}


def nvidia_smi(query: str) -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return [part.strip() for part in result.stdout.strip().splitlines()[0].split(",")]


class GpuSampler:
    """Poll utilization, power, and temperature once per second."""

    def __init__(self):
        self.samples: list[tuple[float, float, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                util, power, temp = nvidia_smi("utilization.gpu,power.draw,temperature.gpu")
                self.samples.append((float(util), float(power), float(temp)))
            except (subprocess.CalledProcessError, ValueError, OSError):
                pass
            self._stop.wait(1.0)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def summary(self) -> dict:
        if not self.samples:
            return {}
        util, power, temp = zip(*self.samples)
        return {
            "gpu_util_mean_pct": statistics.fmean(util),
            "gpu_power_mean_w": statistics.fmean(power),
            "gpu_power_max_w": max(power),
            "gpu_temp_max_c": max(temp),
        }


def system_info() -> dict:
    import psutil
    import torch

    name, driver, vram = nvidia_smi("name,driver_version,memory.total")
    cpu = platform.processor()
    if platform.system() == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return {
        "os": platform.platform(),
        "cpu": cpu,
        "ram_gib": round(psutil.virtual_memory().total / 1024**3, 1),
        "gpu": name,
        "gpu_vram_mib": int(vram),
        "driver": driver,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def make_thumbnail(preset: str) -> Path:
    image = Image.open(OUT / f"{preset}.png").convert("RGB")
    size = (THUMB_WIDTH, round(image.height * THUMB_WIDTH / image.width))
    # Nearest keeps the pixel-art grid crisp; everything else downsamples smoothly.
    method = Image.Resampling.NEAREST if preset == "pixel-art" else Image.Resampling.LANCZOS
    THUMBS.mkdir(parents=True, exist_ok=True)
    path = THUMBS / f"{preset}.jpg"
    image.resize(size, method).save(path, quality=88, optimize=True)
    return path


def run_preset(preset: str) -> dict:
    image = OUT / f"{preset}.png"
    stats_path = OUT / f"{preset}.json"
    if image.exists() and stats_path.exists():
        print(f"skip {preset} (already generated)", flush=True)
        return json.loads(stats_path.read_text())
    image.unlink(missing_ok=True)
    options = {"seed": SEED, **OVERRIDES.get(preset, {})}
    command = [sys.executable, "-m", "local_image_generator", "--preset", preset,
               "--prompt", EXAMPLES[preset], "--output", str(image),
               "--stats-json", str(stats_path)]
    for name, value in options.items():
        command += [f"--{name}", str(value)]
    print(f"== {preset}: {EXAMPLES[preset]}", flush=True)
    with GpuSampler() as sampler:
        began = time.perf_counter()
        subprocess.run(command, check=True)
        wall = time.perf_counter() - began
    record = json.loads(stats_path.read_text())
    record["stats"]["process_wall_s"] = wall
    record["stats"].update(sampler.summary())
    stats_path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> int:
    assert set(EXAMPLES) == set(PRESETS), "every preset needs an example prompt"
    OUT.mkdir(parents=True, exist_ok=True)
    runs = {}
    for preset in PRESETS:
        runs[preset] = run_preset(preset)
        make_thumbnail(preset)
    (OUT / "benchmarks.json").write_text(json.dumps(
        {"system": system_info(), "runs": runs}, indent=2) + "\n")
    print(f"Wrote {len(runs)} examples to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
