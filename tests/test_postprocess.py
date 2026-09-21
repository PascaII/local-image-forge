import unittest

from PIL import Image

from local_image_generator.postprocess import PIXEL_COLORS, PIXEL_SCALE, dither, pixelate


def gradient(width=256, height=192):
    image = Image.new("RGB", (width, height))
    image.putdata([(x % 256, y % 256, (x + y) % 256)
                   for y in range(height) for x in range(width)])
    return image


class PostprocessTests(unittest.TestCase):
    def test_dither_is_pure_black_and_white(self):
        result = dither(gradient())
        self.assertEqual(result.size, (256, 192))
        self.assertEqual(result.mode, "RGB")
        colors = {color for _, color in result.getcolors()}
        self.assertEqual(colors, {(0, 0, 0), (255, 255, 255)})

    def test_pixelate_snaps_to_grid_with_small_palette(self):
        result = pixelate(gradient())
        self.assertEqual(result.size, (256, 192))
        self.assertLessEqual(len(result.getcolors()), PIXEL_COLORS)
        for top in range(0, 192, PIXEL_SCALE):
            for left in range(0, 256, PIXEL_SCALE):
                block = result.crop((left, top, left + PIXEL_SCALE, top + PIXEL_SCALE))
                self.assertEqual(len(block.getcolors()), 1)


if __name__ == "__main__":
    unittest.main()
