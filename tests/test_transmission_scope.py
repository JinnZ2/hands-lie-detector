"""Claim tests for backlit-transmission.md: transmission is a separate
quantity and does not make the callus thickness map measurable."""
import unittest
from pathlib import Path

from hands_lie_detector.band.contrast import LightCondition

ROOT = Path(__file__).resolve().parents[1]


class TransmissionScope(unittest.TestCase):
    def test_backlit_still_does_not_resolve_plate_thickness(self):
        self.assertFalse(LightCondition.BACKLIT.resolves_thickness)
        self.assertTrue(LightCondition.RAKING.resolves_thickness)

    def test_doc_keeps_scope_lines(self):
        text = (ROOT / "backlit-transmission.md").read_text(encoding="utf-8")
        self.assertIn("opacity = thickness, NOT composition", text)
        self.assertIn("**path opacity**", text)
        self.assertIn("Status: **PROPOSED**", text)

    def test_references_carry_verification_state(self):
        text = (ROOT / "REFERENCES" / "related_fields.md").read_text(encoding="utf-8")
        for n in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5."):
            section = text.split(n, 1)[1].split("\n## ", 1)[0]
            self.assertTrue("VERIFIED (abstract)" in section or "UNVERIFIED" in section, n)


if __name__ == "__main__":
    unittest.main()
