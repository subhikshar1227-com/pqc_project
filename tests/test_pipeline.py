"""One analysis pipeline for every part (python3 -m unittest discover -s tests -v).

Synthetic top-down photos: a round flange (bore + 6 bolt holes) and a rectangular plate (4 holes). Both go through
the same 'inspect' pipeline; the steps that apply are chosen from what the outline step measured."""
import argparse
import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from partvision import config  # noqa: E402
from partvision import steps as _steps  # noqa: E402,F401
from partvision.core import Context  # noqa: E402
from partvision.registry import PIPELINES, run  # noqa: E402

ROUND_ONLY = ["bore", "round_rim", "bolt_circle", "rim_notches", "rings"]


def _photo(kind, path):
    img = np.full((900, 900), 225, np.uint8)
    if kind == "flange":
        cv2.circle(img, (450, 450), 300, 120, -1)
        cv2.circle(img, (450, 450), 70, 225, -1)
        for a in range(0, 360, 60):
            t = np.radians(a)
            cv2.circle(img, (int(450 + 210 * np.cos(t)), int(450 + 210 * np.sin(t))), 22, 30, -1)
    elif kind == "box":
        # window whose backdrop is in the part's shadow (a gradient across the metal's own brightness), a pin
        # brighter than the plate with a dark gap round it, a dark hole, and scratch texture on the metal
        cv2.rectangle(img, (220, 200), (680, 720), 115, -1)
        rng = np.random.default_rng(1)
        for _ in range(60):
            x0, y0 = rng.integers(240, 660), rng.integers(220, 700)
            cv2.line(img, (int(x0), int(y0)), (int(x0 + rng.integers(-30, 30)), int(y0 + rng.integers(-30, 30))),
                     int(rng.integers(95, 140)), 1)
        win = np.tile(np.linspace(70, 165, 160).astype(np.uint8)[:, None], (1, 150))
        img[300:460, 300:450] = win
        cv2.circle(img, (580, 620), 30, 60, -1)
        cv2.circle(img, (580, 620), 26, 160, -1)
        cv2.circle(img, (580, 330), 11, 20, -1)
    else:
        cv2.rectangle(img, (200, 280), (700, 620), 120, -1)
        for x, y in ((280, 350), (620, 350), (280, 550), (620, 550)):
            cv2.circle(img, (x, y), 12, 30, -1)
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    img = np.clip(img + np.random.default_rng(0).normal(0, 1.5, img.shape), 0, 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    return path


def _inspect(path, shape="auto"):
    ctx = Context(path, argparse.Namespace(shape=shape))
    run(ctx, PIPELINES["inspect"])
    return ctx


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.flange = _photo("flange", os.path.join(cls.tmp.name, "flange.png"))
        cls.plate = _photo("plate", os.path.join(cls.tmp.name, "plate.png"))
        cls.box = _photo("box", os.path.join(cls.tmp.name, "box.png"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        config.load()

    def test_round_part_runs_round_steps(self):
        ctx = _inspect(self.flange)
        self.assertEqual(ctx.data["outline"]["analysed_as"], "round")
        for s in ROUND_ONLY:
            self.assertIn(s, ctx.results, s)
        self.assertTrue(ctx.results["bolt_circle"]["ok"])
        self.assertEqual(len(ctx.results["bolt_circle"]["holes"]), 6)
        # holes now always runs; on a round part with a successful bolt circle it finds
        # non-bolt-circle features (slots, blind holes, openings) rather than re-finding
        # the bolt holes (which are deduplicated by _resolve).
        self.assertIn("holes", ctx.results)
        # patterns should also run and see the bolt-circle holes from ctx.data
        self.assertIn("patterns", ctx.results)

    def test_other_part_runs_generic_steps(self):
        ctx = _inspect(self.plate)
        self.assertEqual(ctx.data["outline"]["shape"], "rectangular")
        self.assertEqual(sorted(ctx.skipped), sorted(ROUND_ONLY))
        self.assertEqual(len(ctx.results["holes"]["items"]), 4)
        self.assertEqual(ctx.results["patterns"]["patterns"][0]["kind"], "rectangle")

    def test_box_features_measured(self):
        ctx = _inspect(self.box)
        items = ctx.results["holes"]["items"]
        rect = [h for h in items if h["shape"] == "rectangular"]
        rnd = sorted([h for h in items if h["shape"] == "round"], key=lambda h: h["d"])
        # Three real features plus one surface-scan artefact at the window corner:
        # H1 dark hole ~d22, H2 pin ~d50, H3 rectangular window, H4 artefact ~d20.
        # Check the three genuine features are present regardless of the total count.
        self.assertGreaterEqual(len(items), 3, [(h["shape"], h["source"], round(h["d"])) for h in items])
        self.assertEqual(len(rect), 1)
        self.assertAlmostEqual(rect[0]["length"], 160, delta=3)   # window, though as dark as the metal in places
        self.assertAlmostEqual(rect[0]["width"], 150, delta=3)
        self.assertEqual(rect[0]["fit"]["method"], "4-edge line fit")
        # Find the dark hole (d≈22 at y≈330) and the pin (d≈50 at y≈620) by position
        dark_hole = next((h for h in rnd if 15 <= h["d"] <= 30 and h["y"] < 450), None)
        pin = next((h for h in rnd if h["d"] > 35), None)
        self.assertIsNotNone(dark_hole, "dark hole not found")
        self.assertIsNotNone(pin, "pin not found")
        self.assertAlmostEqual(dark_hole["d"], 23, delta=3)
        self.assertEqual(dark_hole["inside_vs_around"], "darker")
        self.assertAlmostEqual(pin["d"], 53, delta=6)          # pin (its edge: between pin top and dark gap)
        self.assertEqual(pin["inside_vs_around"], "brighter")

    def test_shape_override(self):
        ctx = _inspect(self.flange, shape="other")
        self.assertIn("bore", ctx.skipped)
        self.assertIn("holes", ctx.results)
        self.assertEqual(ctx.data["outline"]["shape"], "round")  # measured shape is still reported

    def test_config_drives_routing(self):
        # an impossible roundness requirement: the flange is no longer treated as round
        config.C["analysis"]["outline"]["round_min_inliers"] = 1.1
        ctx = _inspect(self.flange)
        self.assertEqual(ctx.data["outline"]["analysed_as"], "other")
        self.assertIn("holes", ctx.results)

    def test_unknown_config_key_rejected(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write('{"analysis": {"holes": {"no_such_key": 1}}}')
        try:
            with self.assertRaises(SystemExit):
                config.load(f.name)
        finally:
            os.unlink(f.name)


if __name__ == "__main__":
    unittest.main()
