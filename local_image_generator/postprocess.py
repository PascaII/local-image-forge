"""Pillow-only passes that make dither and pixel-art output authentic.

Diffusion models imitate these styles with blurry, inconsistent pixels, so
the generated image is re-rendered here onto a real grid or palette.
"""

from PIL import Image

DITHER_SCALE = 2
PIXEL_SCALE = 8
PIXEL_COLORS = 32


def _shrink(image: Image.Image, factor: int) -> Image.Image:
    return image.resize((image.width // factor, image.height // factor), Image.Resampling.BOX)


def _grow(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return image.resize(size, Image.Resampling.NEAREST)


def dither(image: Image.Image) -> Image.Image:
    """1-bit Floyd-Steinberg dither with dots large enough to see."""
    small = _shrink(image.convert("L"), DITHER_SCALE)
    bits = small.convert("1", dither=Image.Dither.FLOYDSTEINBERG)
    return _grow(bits, image.size).convert("RGB")


def pixelate(image: Image.Image) -> Image.Image:
    """Snap to a coarse pixel grid with a small palette and no anti-aliasing."""
    small = _shrink(image.convert("RGB"), PIXEL_SCALE)
    # k-means refinement keeps small but important hues (e.g. grass, white trim)
    # that plain median cut merges into the dominant colors.
    paletted = small.quantize(colors=PIXEL_COLORS, kmeans=4, dither=Image.Dither.NONE)
    return _grow(paletted.convert("RGB"), image.size)


POSTPROCESSORS = {"dither": dither, "pixelate": pixelate}
