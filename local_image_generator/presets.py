"""Prompt templates and modest defaults for a 16 GB CUDA GPU."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    template: str
    width: int = 1024
    height: int = 1024
    steps: int = 40

    def prompt_for(self, subject: str) -> str:
        return self.template.format(subject=subject.strip())


PRESETS = {
    "ascii": Preset(
        name="ascii",
        description="A raster image that resembles character-based ASCII art",
        template=(
            "Create an image of {subject} rendered entirely as meticulous monochrome "
            "ASCII art. Use only visible monospaced characters such as @, #, %, +, -, "
            ". and spaces to build the shapes, shading, and texture. Crisp aligned text "
            "grid, high contrast, dark characters on a plain light background, no "
            "photographic elements, no decorative lettering."
        ),
    ),
    "anime": Preset(
        name="anime",
        description="A polished anime illustration",
        width=896,
        height=1152,
        template=(
            "Anime illustration of {subject}. Expressive clean line art, deliberate "
            "cel shading, vivid but harmonious colors, detailed background, "
            "cinematic composition, consistent anatomy, polished 2D animation look."
        ),
    ),
    "photorealistic": Preset(
        name="photorealistic",
        description="A natural-looking photographic image",
        width=1152,
        height=896,
        template=(
            "A photorealistic photograph of {subject}. Natural materials and skin "
            "textures where applicable, believable lighting and shadows, realistic "
            "lens perspective, subtle depth of field, nuanced color, fine detail, "
            "authentic documentary photography."
        ),
    ),
}
