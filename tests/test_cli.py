import contextlib
from io import StringIO
import json
import sys
import unittest

from local_image_generator.cli import main


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
        }
        for preset, size in expected_sizes.items():
            with self.subTest(preset=preset):
                settings = self.run_dry("--preset", preset, "--seed", "7")
                self.assertIn("a red kite", settings["prompt"])
                self.assertEqual((settings["width"], settings["height"]), size)
                self.assertEqual(settings["steps"], 40)
                self.assertEqual(settings["seed"], 7)

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
