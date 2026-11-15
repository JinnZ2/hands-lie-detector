"""Tests for hands_lie_detector.attention. Every image is constructed here; no
fixture is a photograph and nothing here is a reading of a hand."""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

from hands_lie_detector.attention import image as I
from hands_lie_detector.attention import log as L
from hands_lie_detector.attention import maps as M
from hands_lie_detector.attention import physics as P
from hands_lie_detector.attention.run import run, render

ROOT = Path(__file__).resolve().parents[1]


def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else (b if pb <= pc else c)


def png_bytes(w, h, pixels, ctype=2, palette=None, depth=8, cycle_filters=True):
    """Minimal PNG encoder for tests. pixels: rows of channel tuples."""
    ch = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ctype]
    raw, prev = bytearray(), bytearray(w * ch)
    for y in range(h):
        line = bytearray(v for px in pixels[y] for v in px)
        f = (y % 5) if cycle_filters else 0
        out = bytearray()
        for i, v in enumerate(line):
            a = line[i - ch] if i >= ch else 0
            b, c = prev[i], (prev[i - ch] if i >= ch else 0)
            pred = [0, a, b, (a + b) >> 1, _paeth(a, b, c)][f]
            out.append((v - pred) & 0xFF)
        raw += bytes([f]) + out
        prev = line

    def chunk(t, body):
        return struct.pack(">I", len(body)) + t + body + struct.pack(">I", zlib.crc32(t + body) & 0xFFFFFFFF)
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, ctype, 0, 0, 0))
    if palette:
        data += chunk(b"PLTE", bytes(v for c in palette for v in c))
    return data + chunk(b"IDAT", zlib.compress(bytes(raw))) + chunk(b"IEND", b"")


def write(tmp, name, data):
    p = os.path.join(tmp, name)
    with open(p, "wb") as fh:
        fh.write(data)
    return p


def step_rgb(w=64, h=64, edge=32):
    """Left half dark (40), right half bright (200): one vertical light-mass boundary."""
    return [[(40, 40, 40) if x < edge else (200, 200, 200) for x in range(w)] for _ in range(h)]


def a_log(kind, oid, regions, sha=None, before=True):
    return {"observer": {"kind": kind, "id": oid}, "image_sha256": sha,
            "before_task_prompt": before,
            "entries": [{"rank": i + 1, "noticed": f"thing {i + 1}", "region": r}
                        for i, r in enumerate(regions)]}


class Decode(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_png_rgb_every_filter(self):
        px = [[((x * 7 + y) % 256, (x * 3) % 256, (y * 11) % 256) for x in range(9)] for y in range(7)]
        f = I.load(write(self.tmp, "a.png", png_bytes(9, 7, px)))
        self.assertEqual((f.width, f.height, f.input_mode), (9, 7, "decoded:PNG"))
        self.assertEqual([f.pixel(x, y) for y in range(7) for x in range(9)],
                         [p for row in px for p in row])

    def test_png_gray_palette_rgba(self):
        g = I.load(write(self.tmp, "g.png", png_bytes(3, 2, [[(10,), (20,), (30,)]] * 2, ctype=0)))
        self.assertEqual(g.pixel(2, 1), (30, 30, 30))
        pal = [(1, 2, 3), (250, 251, 252)]
        p = I.load(write(self.tmp, "p.png", png_bytes(2, 1, [[(0,), (1,)]], ctype=3, palette=pal)))
        self.assertEqual(p.pixel(1, 0), (250, 251, 252))
        a = I.load(write(self.tmp, "r.png", png_bytes(1, 1, [[(5, 6, 7, 0)]], ctype=6)))
        self.assertEqual(a.pixel(0, 0), (5, 6, 7))
        self.assertTrue(a.bounds["alpha_dropped"])

    def test_ppm_binary_and_ascii(self):
        b = I.load(write(self.tmp, "b.ppm", b"P6\n# c\n2 1\n255\n" + bytes([1, 2, 3, 4, 5, 6])))
        self.assertEqual(b.pixel(1, 0), (4, 5, 6))
        a = I.load(write(self.tmp, "a.ppm", b"P3 1 1 15\n15 0 5\n"))
        self.assertEqual(a.pixel(0, 0), (255, 0, 85))

    def test_refusals_name_the_alternative(self):
        with self.assertRaises(I.UnsupportedImage) as e:
            I.load(write(self.tmp, "j.jpg", b"\xff\xd8\xff\xe0" + b"x" * 20))
        self.assertIn("coordinates-only", str(e.exception))
        with self.assertRaises(I.UnsupportedImage) as e:
            I.load(write(self.tmp, "h.heic", b"\x00\x00\x00\x18ftypheic" + b"x" * 20))
        self.assertIn("HEIC", str(e.exception))
        with self.assertRaises(I.UnsupportedImage):
            I.load(write(self.tmp, "s.png", png_bytes(1, 1, [[(0, 0, 0)]], depth=16)))
        bad = bytearray(png_bytes(2, 2, [[(0, 0, 0)] * 2] * 2))
        bad[-30] ^= 0xFF
        with self.assertRaises(I.UnsupportedImage) as e:
            I.load(write(self.tmp, "c.png", bytes(bad)))
        self.assertIn("CRC", str(e.exception))


class Physics(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.mkdtemp()
        self.frame = I.load(write(tmp, "s.png", png_bytes(64, 64, step_rgb())))
        self.field = P.Field(self.frame)

    def test_flat_region_reads_zero_and_edge_region_does_not(self):
        flat = P.region_reading(self.field, [2, 2, 20, 60])
        edge = P.region_reading(self.field, [28, 2, 36, 60])
        self.assertEqual((flat["luminance_gradient"], flat["edge_strength"]), (0.0, 0.0))
        self.assertGreater(edge["edge_strength"], 0)
        self.assertGreater(edge["luminance_gradient"], 0)

    def test_specular_diffuse_and_transmission(self):
        tmp = tempfile.mkdtemp()
        px = [[(250, 250, 250), (120, 60, 30)]]
        f = I.load(write(tmp, "t.png", png_bytes(2, 1, px)))
        fld = P.Field(f)
        r = P.region_reading(fld, [0, 0, 2, 1])
        self.assertEqual((r["specular"], r["diffuse"]), (0.5, 0.5))
        self.assertEqual(r["transmission_r_over_b"], P.NOT_APPLICABLE)
        rb = P.region_reading(fld, [0, 0, 2, 1], backlit=True)
        self.assertAlmostEqual(rb["transmission_r_over_b"], (250 + 120) / (250 + 30))

    def test_coordinates_only_is_unmeasured(self):
        f = I.coordinates_only(64, 64)
        r = P.region_reading(P.Field(f), [0, 0, 10, 10])
        self.assertTrue(all(v == P.UNMEASURED for v in r.values()))


class Logs(unittest.TestCase):
    def test_validation(self):
        good = a_log("claude", "c", [[0, 0, 4, 4], None], sha="abc")
        L.validate(good, 64, 64, "abc")
        for bad, msg in (
                (a_log("claude", "c", [[0, 0, 4, 4]], before=False), "before_task_prompt"),
                (a_log("claude", "c", [[0, 0, 99, 4]]), "outside"),
                (a_log("claude", "c", [[4, 0, 4, 4]]), "outside"),
                (a_log("robot", "c", []), "observer kind"),
                (a_log("claude", "c", [[0, 0, 4, 4]], sha="zzz"), "another frame")):
            with self.assertRaises(L.LogError) as e:
                L.validate(bad, 64, 64, "abc")
            self.assertIn(msg, str(e.exception))
        g = a_log("claude", "c", [[0, 0, 4, 4]])
        g["entries"][0]["rank"] = 2
        with self.assertRaises(L.LogError):
            L.validate(g, 64, 64)

    def test_logs_are_never_merged_and_slots_stay_null(self):
        held = L.ObserverLogs()
        self.assertEqual(held.slots(), {"claude": None, "other_models": None, "human": None})
        held.add(a_log("claude", "c1", []))
        with self.assertRaises(L.LogError):
            held.add(a_log("claude", "c1", []))
        held.add(a_log("model", "m1", []))
        self.assertEqual(held.slots(), {"claude": "c1", "other_models": ["m1"], "human": None})
        self.assertFalse([n for n in dir(L) if "merge" in n.lower() or "combine" in n.lower()])


class Maps(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.mkdtemp()
        self.frame = I.load(write(tmp, "s.png", png_bytes(64, 64, step_rgb())))

    def test_four_labels_and_controls(self):
        log = a_log("claude", "c", [[28, 0, 36, 64], [0, 0, 8, 8], [0, 40, 8, 48]], self.frame.sha256)
        rep = run(self.frame, [log], seed=1)
        r = rep["observers"][0]
        self.assertEqual([l["label"] for l in r["landings"]],
                         ["CALIBRATED_POINTER", "BIAS", "BIAS"])
        c = r["cell_counts"]
        self.assertGreater(c["CALIBRATED_POINTER"], 0)
        self.assertGreater(c["BIAS"], 0)
        self.assertEqual(c["BLIND_SPOT"], 0)   # every edge cell was attended
        rr = r["controls"]["random_regions"]
        self.assertGreater(rr["observed_mean_edge_strength"], 0)
        self.assertIn("share_of_draws_ge_observed", rr)
        self.assertIn("spearman_order_vs_edge_strength", r["controls"]["shuffled_order"])

    def test_blind_spot_when_the_edge_is_not_attended(self):
        log = a_log("human", "carrier", [[0, 0, 8, 8]], self.frame.sha256)
        r = run(self.frame, [log])["observers"][0]
        self.assertGreater(r["cell_counts"]["BLIND_SPOT"], 0)
        self.assertEqual(r["controls"]["shuffled_order"]["spearman_order_vs_edge_strength"], None)

    def test_divergence_lists_split_cells_from_separate_readings(self):
        a = a_log("claude", "c", [[28, 0, 36, 8]], self.frame.sha256)
        b = a_log("model", "m", [[0, 56, 8, 64]], self.frame.sha256)
        rep = run(self.frame, [a, b])
        self.assertEqual(len(rep["observers"]), 2)
        d = rep["divergence"]
        self.assertEqual(d["observers"], ["claude:c", "model:m"])
        self.assertTrue(d["split_cells"])
        self.assertTrue(all(len(s["attended_by"]) == 1 for s in d["split_cells"]))

    def test_coordinates_only_cells_not_evaluable(self):
        f = I.coordinates_only(64, 64)
        r = run(f, [a_log("claude", "c", [[0, 0, 8, 8]])])
        self.assertEqual(r["input_mode"], "coordinates_only")
        o = r["observers"][0]
        self.assertEqual(o["cell_counts"]["NOT_EVALUABLE"], 64)
        self.assertEqual(o["controls"]["state"], "NOT_EVALUABLE")
        self.assertEqual(r["bounds"]["frame_edge_fraction"], P.UNMEASURED)

    def test_small_scales_frame_and_regions(self):
        tmp = tempfile.mkdtemp()
        f = I.load(write(tmp, "b.png", png_bytes(300, 300, step_rgb(300, 300, 150), cycle_filters=False)))
        rep = run(f, [a_log("claude", "c", [[140, 0, 160, 300]], f.sha256)], small=True)
        self.assertEqual(rep["bounds"]["small_factor"], 3)
        self.assertEqual((rep["frame"]["width"], rep["frame"]["height"]), (100, 100))
        self.assertEqual(rep["observers"][0]["landings"][0]["region"], [46, 0, 54, 100])

    def test_spearman_known_answers(self):
        self.assertAlmostEqual(M.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(M.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertIsNone(M.spearman([1, 2], [1, 2]))
        self.assertIsNone(M.spearman([1, 2, 3], [5, 5, 5]))


class PrivacyAndCli(unittest.TestCase):
    def test_no_pixel_level_data_in_the_output(self):
        tmp = tempfile.mkdtemp()
        f = I.load(write(tmp, "s.png", png_bytes(64, 64, step_rgb())))
        rep = run(f, [a_log("claude", "c", [[28, 0, 36, 64]], f.sha256)], backlit=True)
        def lists(x):
            if isinstance(x, list):
                yield x
                for v in x: yield from lists(v)
            elif isinstance(x, dict):
                for v in x.values(): yield from lists(v)
        self.assertFalse([l for l in lists(rep) if len(l) >= 64 * 64 // 4])
        self.assertIn("stay on device", rep["privacy"])
        self.assertNotIn("interpret", render(rep).lower().replace("no interpretation", ""))

    def test_cli(self):
        tmp = tempfile.mkdtemp()
        img = write(tmp, "s.png", png_bytes(32, 32, step_rgb(32, 32, 16)))
        lg = os.path.join(tmp, "c.json")
        with open(lg, "w") as fh:
            json.dump(a_log("claude", "c", [[12, 0, 20, 32]]), fh)
        out = os.path.join(tmp, "r.json")
        cmd = [sys.executable, "-m", "hands_lie_detector.attention", "--image", img, "--log", lg,
               "--json", out]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("CALIBRATED_POINTER", r.stdout)
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("not overwritten", r.stderr)
        jpg = write(tmp, "x.jpg", b"\xff\xd8\xff\xe0" + b"0" * 10)
        r = subprocess.run([sys.executable, "-m", "hands_lie_detector.attention", "--image", jpg],
                           cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("REFUSED", r.stderr)


class Doc(unittest.TestCase):
    def test_doc_status_and_rules(self):
        text = (ROOT / "attention-as-sensor.md").read_text(encoding="utf-8")
        for s in ("Status: **PROPOSED**", "never merged", "before any task prompt",
                  "coordinates-only", "path opacity", "stay on device"):
            self.assertIn(s, text)


if __name__ == "__main__":
    unittest.main()


class Sweep(unittest.TestCase):
    def test_labels_shift_with_the_edge_threshold_and_hold_elsewhere(self):
        from hands_lie_detector.attention.sweep import sweep, render
        tmp = tempfile.mkdtemp()
        # columns 24..40: a ramp of 24 per pixel (between EDGE_T 16 and 32); flat elsewhere
        def lum(x):
            return 20 if x < 24 else (20 + 24 * (x - 24) if x <= 33 else 236)
        px = [[(min(255, lum(x)),) * 3 for x in range(64)] for _ in range(64)]
        f = I.load(write(tmp, "r.png", png_bytes(64, 64, px)))
        rep = sweep(f, [a_log("claude", "c", [[25, 0, 33, 64], [0, 0, 8, 8]], f.sha256)])
        o = rep["observers"][0]
        ramp, flat = o["landings"]
        self.assertEqual(ramp["state"], "SHIFTS")
        self.assertIn("CALIBRATED_POINTER", ramp["labels"])
        self.assertIn("BIAS", ramp["labels"])
        self.assertTrue(any(d["EDGE_T"] == 16.0 for d in ramp["differs_from_default_at"]))
        self.assertEqual((flat["state"], flat["labels"]), ("HOLDS", ["BIAS"]))
        self.assertEqual(rep["bounds"]["settings"], 27)
        self.assertEqual(set(o["cells_by_grid"]), {"6", "8", "12"})
        lat = o["lattice"]
        self.assertEqual(sum(lat["held_across_all_settings"].values()) + lat["shifting"],
                         lat["points"])
        self.assertGreater(lat["shifting"], 0)
        self.assertIn("SHIFTS", render(rep))

    def test_coordinates_only_sweep_not_evaluable(self):
        from hands_lie_detector.attention.sweep import sweep
        rep = sweep(I.coordinates_only(32, 32), [a_log("claude", "c", [[0, 0, 8, 8]])])
        self.assertEqual(rep["state"], "NOT_EVALUABLE")


class Template(unittest.TestCase):
    def test_carried_log_has_void_regions_and_is_refused_until_set(self):
        from hands_lie_detector.attention.template import build
        lg = build("claude", "m", ["a", "b"], "chat transcript, 2026-11-15 morning")
        self.assertEqual([e["region"] for e in lg["entries"]], [None, None])
        self.assertEqual(lg["provenance"]["grade"], "CARRIED")
        with self.assertRaises(L.LogError):
            L.validate(lg, 64, 64)            # before_task_prompt unset
        lg["before_task_prompt"] = True
        L.validate(lg, 64, 64)

    def test_morning_claude_log_as_filled(self):
        t = json.loads((ROOT / "attention_runs" / "claude_2026-11-15_morning.json").read_text())
        self.assertEqual(t["provenance"]["grade"], "CARRIED")
        self.assertIs(t["before_task_prompt"], True)
        self.assertEqual([e["region"] for e in t["entries"]], [None] * 4)
        self.assertEqual(t["entries"][0]["noticed"], "sun / low sun glare through windshield")
        self.assertEqual(t["provenance"]["excluded"]["items"], ["palm creases", "weathered texture"])
        noticed = " ".join(e["noticed"] for e in t["entries"])
        self.assertNotIn("crease", noticed)
        L.validate(t, 64, 64)
        self.assertFalse((ROOT / "attention_runs" / "claude_2026-11-15_morning.template.json").exists())
