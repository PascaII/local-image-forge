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
    # Name of a Pillow pass in postprocess.POSTPROCESSORS, applied after generation.
    postprocess: str | None = None

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
    "cartoon": Preset(
        name="cartoon",
        description="A bright, playful cartoon",
        template=(
            "Cartoon illustration of {subject}. Bold rounded outlines, simplified "
            "exaggerated shapes, flat saturated colors with soft cel shadows, "
            "expressive faces, clean uncluttered background, cheerful Saturday-morning "
            "animation look."
        ),
    ),
    "dither": Preset(
        name="dither",
        description="1-bit Floyd-Steinberg dithered artwork",
        postprocess="dither",
        template=(
            "Striking high-contrast monochrome illustration of {subject}. Bold "
            "readable silhouettes, strong directional lighting, deep blacks and "
            "bright highlights, smooth tonal gradients, simple composition, "
            "black-and-white only, no color."
        ),
    ),
    "watercolor": Preset(
        name="watercolor",
        description="A loose watercolor painting",
        width=1152,
        height=896,
        template=(
            "Watercolor painting of {subject}. Soft translucent washes, pigments "
            "bleeding and blooming into wet paper, visible cold-press paper texture, "
            "loose confident brushwork, delicate pencil underdrawing, areas of "
            "untouched white paper, gentle harmonious palette."
        ),
    ),
    "oil-painting": Preset(
        name="oil-painting",
        description="A classical oil painting on canvas",
        width=896,
        height=1152,
        template=(
            "Oil painting of {subject}. Rich layered pigments, visible impasto "
            "brushstrokes, canvas weave texture, dramatic chiaroscuro lighting, "
            "warm glazes, painterly edges, museum-quality fine art composition."
        ),
    ),
    "pencil-sketch": Preset(
        name="pencil-sketch",
        description="A graphite pencil sketch",
        template=(
            "Graphite pencil sketch of {subject} on off-white sketchbook paper. "
            "Confident gestural lines, careful cross-hatching and tonal shading, "
            "smudged soft shadows, visible construction lines, monochrome, "
            "unfinished edges fading into the paper."
        ),
    ),
    "ukiyo-e": Preset(
        name="ukiyo-e",
        description="A Japanese Edo-period woodblock print",
        width=896,
        height=1152,
        template=(
            "Ukiyo-e Japanese woodblock print of {subject}. Bold flowing outlines, "
            "flat areas of traditional color such as indigo, vermilion, and ochre, "
            "subtle bokashi gradients, stylized waves and clouds, visible wood grain "
            "and washi paper texture, Edo-period composition."
        ),
    ),
    "pixel-art": Preset(
        name="pixel-art",
        description="A 16-bit pixel art scene snapped to a real pixel grid",
        postprocess="pixelate",
        template=(
            "16-bit pixel art of {subject}, like a classic console game scene. "
            "Large clearly defined pixels, limited color palette, crisp hard edges "
            "with no anti-aliasing, simple readable shapes, flat shading with a "
            "few highlight tones, retro video game aesthetic."
        ),
    ),
    "comic": Preset(
        name="comic",
        description="A pop-art comic book panel with halftone dots",
        width=896,
        height=1152,
        template=(
            "Comic book panel of {subject}. Heavy black ink outlines, dynamic "
            "composition, bold primary colors, Ben-Day halftone dot shading, "
            "dramatic action lines, vintage pop-art printing look, no speech "
            "bubbles, no text."
        ),
    ),
    "isometric": Preset(
        name="isometric",
        description="A tiny isometric 3D diorama",
        template=(
            "Isometric 3D diorama of {subject}, built as a tiny miniature scene on a "
            "floating square base. Precise isometric angle, cute detailed props, "
            "soft global illumination, tilt-shift miniature effect, clean pastel "
            "background, polished 3D render."
        ),
    ),
    "papercraft": Preset(
        name="papercraft",
        description="Layered cut-paper craft",
        width=1152,
        height=896,
        template=(
            "Layered papercraft artwork of {subject}. Shapes cut from colored card "
            "stock and stacked in depth, soft drop shadows between layers, visible "
            "paper fibers and clean cut edges, handmade shadow-box look, warm "
            "studio lighting."
        ),
    ),
    "low-poly": Preset(
        name="low-poly",
        description="A faceted low-poly 3D render",
        template=(
            "Low-poly 3D render of {subject}. Faceted geometric surfaces built from "
            "visible flat triangles, flat shading per facet, simple gradient sky, "
            "limited harmonious palette, clean minimalist composition, "
            "stylized game-art look."
        ),
    ),
}
