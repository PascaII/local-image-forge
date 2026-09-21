import contextlib
from io import StringIO
import json
import sys
import unittest

from local_image_generator.cli import format_stats, main
from local_image_generator.presets import PRESETS


class CliTests(unittest.TestCase):
    def run_dry(self, *args):
        out = StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["--prompt", "a red kite", "--dry-run", *args])
        self.assertEqual(code, 0)
        self.assertNotIn("torch", sys.modules)
        return json.loads(out.getvalue())

    def test_all_presets_compose_subject_and_use_small_defaults(self):
        expected_sizes = {
            "ascii": (1024, 1024),
            "anime": (896, 1152),
            "photorealistic": (1152, 896),
            "cartoon": (1024, 1024),
            "dither": (1024, 1024),
            "watercolor": (1152, 896),
            "oil-painting": (896, 1152),
            "pencil-sketch": (1024, 1024),
            "ukiyo-e": (896, 1152),
            "pixel-art": (1024, 1024),
            "comic": (896, 1152),
            "isometric": (1024, 1024),
            "papercraft": (1152, 896),
            "low-poly": (1024, 1024),
        }
        self.assertEqual(set(expected_sizes), set(PRESETS))
        expected_postprocess = {"dither": "dither", "pixel-art": "pixelate"}
        for preset, size in expected_sizes.items():
            with self.subTest(preset=preset):
                settings = self.run_dry("--preset", preset, "--seed", "7")
                self.assertIn("a red kite", settings["prompt"])
                self.assertEqual((settings["width"], settings["height"]), size)
                self.assertEqual(settings["steps"], 40)
                self.assertEqual(settings["seed"], 7)
                self.assertEqual(settings["postprocess"], expected_postprocess.get(preset))

    def test_format_stats_summarizes_run(self):
        stats = dict.fromkeys([
            "model_load_s", "text_encode_s", "time_to_first_step_s", "denoise_s",
            "step_mean_s", "steps_per_s", "vae_decode_s", "time_to_image_s",
            "s_per_megapixel", "total_s", "peak_vram_reserved_gib",
            "peak_vram_allocated_gib",
        ], 1.5)
        stats["peak_ram_gib"] = None
        summary = format_stats(stats)
        self.assertIn("Time to image", summary)
        self.assertIn("n/a", summary)

    def test_no_postprocess_keeps_raw_output(self):
        settings = self.run_dry("--preset", "pixel-art", "--no-postprocess")
        self.assertIsNone(settings["postprocess"])

    def test_overrides_and_output(self):
        settings = self.run_dry("--width", "768", "--height", "512", "--steps", "20",
                                "--offload", "sequential", "--output", "custom.png")
        self.assertEqual((settings["width"], settings["height"], settings["steps"]),
                         (768, 512, 20))
        self.assertEqual(settings["offload"], "sequential")
        self.assertEqual(settings["output"], "custom.png")

    def test_invalid_dimension_exits_before_model_load(self):
        with contextlib.redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as caught:
                main(["--prompt", "kite", "--width", "777", "--dry-run"])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
