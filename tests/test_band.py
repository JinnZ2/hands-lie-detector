"""
Tests for hands_lie_detector.band.

Claim tests per `band-not-scale.md`, including its RETRACTION: dispersion reads
the geometric concentration of the load history, not competence. The test that
matters most is the one asserting a variable-geometry generalist and a
fixed-geometry specialist are NOT ranked against each other.

Run: python -m unittest discover tests
"""

import unittest

from hands_lie_detector.band import (
    BandPosition,
    LightCondition,
    MapState,
    Sensing,
    interpret_acute_damage,
    monotone_score,
    read_band,
)

# Stipulated maps. Not measurements.
SOFT = {"thumb_crotch": 0.10, "palm_below_index": 0.12, "base_of_fingers": 0.09,
        "fingertip_pads": 0.11, "heel_of_palm": 0.10, "across_palm_crease": 0.08,
        "thumb_pad": 0.12}
SPECIALIST = {"thumb_crotch": 0.78, "palm_below_index": 0.72, "base_of_fingers": 0.80,
              "fingertip_pads": 0.18, "heel_of_palm": 0.55,
              "across_palm_crease": 0.70, "thumb_pad": 0.22}
GENERALIST = {"thumb_crotch": 0.52, "palm_below_index": 0.48, "base_of_fingers": 0.55,
              "fingertip_pads": 0.44, "heel_of_palm": 0.50,
              "across_palm_crease": 0.53, "thumb_pad": 0.46}
SATURATED = {"thumb_crotch": 0.82, "palm_below_index": 0.80, "base_of_fingers": 0.84,
             "fingertip_pads": 0.79, "heel_of_palm": 0.81,
             "across_palm_crease": 0.83, "thumb_pad": 0.80}


class TestConcentrationNotSkill(unittest.TestCase):
    """The retraction, asserted."""

    def test_generalist_and_saturated_are_not_separable_by_thickness(self):
        """The core of the retraction. Both land in the ambiguous class."""
        self.assertIs(read_band(GENERALIST).state, MapState.UNIFORM_THICK)
        self.assertIs(read_band(SATURATED).state, MapState.UNIFORM_THICK)

    def test_the_ambiguous_state_says_it_is_ambiguous(self):
        readout = read_band(GENERALIST)
        self.assertTrue(readout.state.ambiguous)
        self.assertTrue(readout.requires_sensing_test)
        self.assertIn("AMBIGUOUS", readout.report())

    def test_concentration_is_not_reported_as_quality(self):
        """A variable-geometry history scores low. That is not a demerit."""
        report = read_band(GENERALIST, light=LightCondition.RAKING).report()
        self.assertIn("NOT skill", report)
        self.assertNotIn("the working hand", report)

    def test_specialist_reads_as_concentrated_not_as_better(self):
        self.assertIs(read_band(SPECIALIST).state, MapState.CONCENTRATED)
        self.assertIn("concentration, not skill",
                      read_band(SPECIALIST).interpretation())


class TestBandPositionNeedsSensing(unittest.TestCase):
    def test_position_is_unresolved_without_a_sensing_test(self):
        self.assertIs(read_band(GENERALIST).position, BandPosition.UNRESOLVED)
        self.assertIs(read_band(SPECIALIST).position, BandPosition.UNRESOLVED)

    def test_intact_sensing_puts_a_thick_hand_in_band(self):
        readout = read_band(GENERALIST, sensing=Sensing.INTACT)
        self.assertIs(readout.position, BandPosition.IN_BAND)

    def test_degraded_sensing_reads_as_saturated(self):
        readout = read_band(SATURATED, sensing=Sensing.DEGRADED)
        self.assertIs(readout.position, BandPosition.OUT_SATURATED)

    def test_a_soft_map_needs_no_sensing_test(self):
        self.assertIs(read_band(SOFT).position, BandPosition.OUT_SOFT)


class TestInstrumentLimits(unittest.TestCase):
    def test_thickness_is_not_measurable_without_raking_light(self):
        for light in (LightCondition.FLAT_OVERHEAD, LightCondition.BACKLIT,
                      LightCondition.UNKNOWN):
            self.assertFalse(read_band(SOFT, light=light).reading.measurable)
        self.assertTrue(read_band(SOFT, light=LightCondition.RAKING).reading.measurable)

    def test_unmeasurable_frames_say_so_before_the_state(self):
        report = read_band(SPECIALIST, light=LightCondition.BACKLIT).report()
        self.assertIn("NOT MEASURABLE", report)

    def test_dorsal_zones_are_refused_by_the_thickness_map(self):
        """Dorsal has no adaptation route. Different instrument, different clock."""
        with self.assertRaises(ValueError):
            read_band({"dorsal_metacarpal": 0.4})

    def test_thickness_outside_unit_range_is_rejected(self):
        with self.assertRaises(ValueError):
            read_band({"thumb_crotch": 1.4})


class TestMonotoneSignError(unittest.TestCase):
    def test_monotone_scorer_still_ranks_the_saturated_hand_highest(self):
        scores = {
            name: monotone_score(read_band(m).reading)
            for name, m in (("soft", SOFT), ("specialist", SPECIALIST),
                            ("generalist", GENERALIST), ("saturated", SATURATED))
        }
        self.assertEqual(max(scores, key=scores.get), "saturated")

    def test_grime_is_uniform_and_does_not_create_concentration(self):
        grimed = {z: min(1.0, t + 0.25) for z, t in SOFT.items()}
        self.assertIsNot(read_band(grimed).state, MapState.CONCENTRATED)

    def test_washing_does_not_change_a_concentrated_map(self):
        washed = {z: max(0.0, t - 0.15) for z, t in SPECIALIST.items()}
        self.assertIs(read_band(washed).state, MapState.CONCENTRATED)


class TestAcuteDamage(unittest.TestCase):
    def test_lesion_in_band_is_the_price_not_a_demerit(self):
        self.assertIn("not a demerit",
                      interpret_acute_damage(BandPosition.IN_BAND, True))

    def test_no_lesion_when_saturated_is_not_evidence_of_skill(self):
        self.assertIn("not evidence of skill",
                      interpret_acute_damage(BandPosition.OUT_SATURATED, False))

    def test_an_unresolved_position_cannot_interpret_a_lesion(self):
        self.assertIn("unresolved",
                      interpret_acute_damage(BandPosition.UNRESOLVED, True))


class TestProvenance(unittest.TestCase):
    def test_thresholds_are_flagged_as_stipulated(self):
        from hands_lie_detector.band import DEFAULT_THRESHOLDS

        self.assertFalse(DEFAULT_THRESHOLDS.is_evidence_based)
        self.assertIn("stipulated", read_band(SOFT).report())

    def test_management_acts_are_enumerated_and_uncertified(self):
        from hands_lie_detector.band import MANAGEMENT_ACTS

        self.assertGreaterEqual(len(MANAGEMENT_ACTS), 5)


if __name__ == "__main__":
    unittest.main()
