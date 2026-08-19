"""
Tests for hands_lie_detector.band.

Claim tests, same as tests/test_integration.py: each asserts something
`band-not-scale.md` states in prose. The sign-error demonstration in particular
must not quietly stop working.

Run: python -m unittest discover tests
"""

import unittest

from hands_lie_detector.band import (
    HandState,
    interpret_acute_damage,
    monotone_score,
    read_band,
)

# Stipulated maps, one per state. Not measurements.
SOFT = {"thumb_crotch": 0.10, "palm_below_index": 0.12, "base_of_fingers": 0.09,
        "fingertip_pads": 0.11, "heel_of_palm": 0.10, "across_palm_crease": 0.08,
        "thumb_pad": 0.12}
BANDED = {"thumb_crotch": 0.78, "palm_below_index": 0.72, "base_of_fingers": 0.80,
          "fingertip_pads": 0.18, "heel_of_palm": 0.55, "across_palm_crease": 0.70,
          "thumb_pad": 0.22}
GLASSY = {"thumb_crotch": 0.82, "palm_below_index": 0.80, "base_of_fingers": 0.84,
          "fingertip_pads": 0.79, "heel_of_palm": 0.81, "across_palm_crease": 0.83,
          "thumb_pad": 0.80}


class TestThreeStates(unittest.TestCase):
    def test_states_separate(self):
        self.assertIs(read_band(SOFT).state, HandState.SOFT)
        self.assertIs(read_band(BANDED).state, HandState.BANDED)
        self.assertIs(read_band(GLASSY).state, HandState.GLASSY)

    def test_mean_cannot_separate_banded_from_glassy(self):
        """Both are thick. This is why mean is not the primary feature."""
        banded, glassy = read_band(BANDED).reading, read_band(GLASSY).reading
        self.assertGreater(banded.mean, 0.4)
        self.assertGreater(glassy.mean, 0.4)
        self.assertLess(abs(banded.mean - glassy.mean), 0.3)

    def test_contrast_separates_all_three(self):
        d = [read_band(m).reading.dispersion for m in (SOFT, BANDED, GLASSY)]
        soft, banded, glassy = d
        self.assertGreater(banded, soft * 5)
        self.assertGreater(banded, glassy * 5)

    def test_only_the_banded_hand_records_a_decision(self):
        self.assertTrue(read_band(BANDED).reading.records_a_decision)
        self.assertFalse(read_band(SOFT).reading.records_a_decision)
        self.assertFalse(read_band(GLASSY).reading.records_a_decision)


class TestSignError(unittest.TestCase):
    def test_monotone_scorer_ranks_the_out_of_band_hand_highest(self):
        """The sign error, asserted so the demonstration cannot rot."""
        scores = {name: monotone_score(read_band(m).reading)
                  for name, m in (("soft", SOFT), ("banded", BANDED), ("glassy", GLASSY))}
        self.assertEqual(max(scores, key=scores.get), "glassy")
        self.assertGreater(scores["glassy"], scores["banded"])

    def test_band_readout_flags_the_disagreement(self):
        self.assertTrue(read_band(GLASSY).monotone_disagrees)
        self.assertFalse(read_band(BANDED).monotone_disagrees)
        self.assertIn("SIGN ERROR", read_band(GLASSY).report())

    def test_grime_is_uniform_and_filtered_by_contrast(self):
        """A uniform layer added to any map moves mean, not state."""
        grimed = {z: min(1.0, t + 0.25) for z, t in SOFT.items()}
        self.assertGreater(read_band(grimed).reading.mean,
                           read_band(SOFT).reading.mean)
        self.assertIsNot(read_band(grimed).state, HandState.BANDED)

    def test_washing_does_not_change_the_banded_state(self):
        """The boundary map is high-frequency; a removed uniform layer spares it."""
        washed = {z: max(0.0, t - 0.15) for z, t in BANDED.items()}
        self.assertIs(read_band(washed).state, HandState.BANDED)


class TestAcuteDamage(unittest.TestCase):
    def test_blister_on_a_banded_hand_is_the_price_not_a_demerit(self):
        reading = interpret_acute_damage(HandState.BANDED, has_acute_lesion=True)
        self.assertIn("not a demerit", reading)

    def test_absent_lesion_on_a_glassy_hand_is_not_evidence_of_skill(self):
        reading = interpret_acute_damage(HandState.GLASSY, has_acute_lesion=False)
        self.assertIn("not evidence of skill", reading)


class TestProvenance(unittest.TestCase):
    def test_thresholds_are_flagged_as_stipulated(self):
        from hands_lie_detector.band import DEFAULT_THRESHOLDS

        self.assertFalse(DEFAULT_THRESHOLDS.is_evidence_based)
        self.assertIn("stipulated", read_band(SOFT).report())

    def test_management_acts_are_enumerated_and_uncertified(self):
        from hands_lie_detector.band import MANAGEMENT_ACTS

        self.assertGreaterEqual(len(MANAGEMENT_ACTS), 5)

    def test_thickness_outside_unit_range_is_rejected(self):
        with self.assertRaises(ValueError):
            read_band({"thumb_crotch": 1.4})


if __name__ == "__main__":
    unittest.main()
