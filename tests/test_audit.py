"""
Tests for hands_lie_detector.audit.

Claim tests, per `calibration-standard.md`. The refusals matter most here: a
slope that quietly became available, or a summary field that got added back,
would undo the design rather than extend it.

Run: python -m unittest discover tests
"""

import unittest

from hands_lie_detector.audit import (
    CrossSection,
    LeakageVerdict,
    ModelResponse,
    Provenance,
    Specimen,
    SpecimenLine,
    commit_stimulus,
    compare_across,
    vocabulary_signature,
)


def _response(model: str, date: str = "2026-08-19", stim: str = "no_props_001"):
    return ModelResponse(model_string=model, date=date, stimulus_id=stim,
                         verbatim="...", routing_note="")


class TestCrossSection(unittest.TestCase):
    def test_model_string_is_never_a_stable_identifier(self):
        self.assertFalse(_response("some-model-v3").is_stable_identifier)

    def test_no_summary_field(self):
        from dataclasses import fields

        names = {f.name for f in fields(ModelResponse)}
        self.assertNotIn("summary", names)
        self.assertIn("verbatim", names)

    def test_mixed_dates_are_rejected(self):
        with self.assertRaises(ValueError):
            CrossSection("2026-08-19", "no_props_001",
                         [_response("a"), _response("b", date="2025-01-01")])

    def test_mismatched_stimulus_is_rejected(self):
        with self.assertRaises(ValueError):
            CrossSection("2026-08-19", "no_props_001",
                         [_response("a"), _response("b", stim="other")])

    def test_compare_across_refuses_a_slope(self):
        early = CrossSection("2025-03-01", "s1",
                             [_response("a", "2025-03-01", "s1")])
        late = CrossSection("2026-08-01", "s1",
                            [_response("a", "2026-08-01", "s1")])
        result = compare_across(early, late, {"a": 0.2}, {"a": 0.9})
        self.assertFalse(result.slope_available)
        self.assertIn("NOT AVAILABLE", result.report())

    def test_comparison_requires_a_shared_stimulus(self):
        early = CrossSection("2025-03-01", "s1", [_response("a", "2025-03-01", "s1")])
        late = CrossSection("2026-08-01", "s2", [_response("a", "2026-08-01", "s2")])
        with self.assertRaises(ValueError):
            compare_across(early, late, {"a": 0.2}, {"a": 0.9})

    def test_envelope_is_the_reported_quantity(self):
        cs = CrossSection("2026-08-19", "s1",
                          [_response(m, stim="s1") for m in ("a", "b", "c")])
        self.assertEqual(cs.spread({"a": 0.1, "b": 0.5, "c": 0.3}), (0.1, 0.5))


class TestSpecimenProvenance(unittest.TestCase):
    def test_only_measured_is_stable_across_the_interval(self):
        stable = [m for m in Provenance if m.stable_across_interval]
        self.assertEqual(stable, [Provenance.MEASURED])

    def test_model_rationale_is_reconstructed_not_observed(self):
        """A model has no readout of its own vectors."""
        line = SpecimenLine("pass 1 attributed to the wet task because...",
                            Provenance.RECONSTRUCTED)
        self.assertFalse(line.provenance.stable_across_interval)
        self.assertIn("reconstructed", str(line))

    def test_specimen_without_measured_lines_says_so(self):
        spec = Specimen("003", "2026-08-19",
                        model_read=[SpecimenLine("out", Provenance.OBSERVED)])
        self.assertEqual(spec.stable_lines, [])
        self.assertIn("no MEASURED line", spec.report())

    def test_census_counts_every_mark(self):
        spec = Specimen(
            "004", "2026-08-19",
            observation=[SpecimenLine("markers", Provenance.MEASURED),
                         SpecimenLine("two domains", Provenance.TESTIMONY)],
            model_read=[SpecimenLine("wet task", Provenance.OBSERVED)],
        )
        census = spec.provenance_census()
        self.assertEqual(census["measured"], 1)
        self.assertEqual(census["testimony"], 1)
        self.assertEqual(census["observed"], 1)
        self.assertEqual(len(spec.stable_lines), 1)


class TestLeakage(unittest.TestCase):
    def test_repo_vocabulary_marks_contamination(self):
        sig = vocabulary_signature("The band state is maintained; skin memory persists.")
        self.assertIs(sig.verdict, LeakageVerdict.CONTAMINATED)
        self.assertGreater(sig.penetration, 0.0)

    def test_mechanism_without_repo_terms_reads_as_derived(self):
        sig = vocabulary_signature(
            "Shear at the interface with a stiffness mismatch drives delamination."
        )
        self.assertIs(sig.verdict, LeakageVerdict.DERIVED)
        self.assertEqual(sig.penetration, 0.0)

    def test_neither_vocabulary_is_inconclusive(self):
        self.assertIs(
            vocabulary_signature("These look like office worker hands.").verdict,
            LeakageVerdict.INCONCLUSIVE,
        )

    def test_contamination_is_itself_a_measurement(self):
        sig = vocabulary_signature("no_context_no_props, clean_but_used, banded")
        self.assertIn("penetration measurement", sig.report())


class TestStimulusCommitment(unittest.TestCase):
    def test_commitment_verifies_the_original_bytes(self):
        data = b"held-out stimulus bytes"
        c = commit_stimulus("held_007", data, "2026-08-19")
        self.assertTrue(c.verify(data))
        self.assertFalse(c.verify(data + b"!"))

    def test_commitment_does_not_store_the_item(self):
        from dataclasses import fields

        names = {f.name for f in fields(type(commit_stimulus("x", b"y", "2026-01-01")))}
        for forbidden in ("data", "bytes", "content", "payload"):
            self.assertNotIn(forbidden, names)

    def test_commitment_is_deterministic(self):
        a = commit_stimulus("x", b"same", "2026-01-01")
        b = commit_stimulus("x", b"same", "2026-01-02")
        self.assertEqual(a.digest, b.digest)


if __name__ == "__main__":
    unittest.main()


class TestConditionCoordinates(unittest.TestCase):
    """A contrast point needs its condition specified, or it plots nowhere."""

    def test_the_dense_arm_ships_unstated_and_fails_its_own_check(self):
        from hands_lie_detector.audit.condition import ARM_A_UNSTATED

        self.assertFalse(ARM_A_UNSTATED.is_plottable)
        self.assertEqual(len(ARM_A_UNSTATED.unstated), len(ARM_A_UNSTATED.coordinates))

    def test_a_coordinate_needs_all_four_fields(self):
        from hands_lie_detector.audit.condition import (
            Cadence, NeedCoordinate, Provision,
        )

        partial = NeedCoordinate("heat", Provision.SELF_MET)
        full = NeedCoordinate("heat", Provision.SELF_MET, "8 cord", 20, Cadence.SEASONAL)
        self.assertFalse(partial.specified)
        self.assertTrue(full.specified)

    def test_n1_supports_direction_but_not_variance(self):
        from hands_lie_detector.audit.condition import ARM_A_UNSTATED, ContrastPoint

        point = ContrastPoint(ARM_A_UNSTATED, ARM_A_UNSTATED)
        supported = " ".join(point.SUPPORTED)
        not_supported = " ".join(point.NOT_SUPPORTED)
        self.assertIn("direction", supported)
        self.assertIn("magnitude", supported)
        self.assertIn("variance", not_supported)
        self.assertIn("prevalence", not_supported)

    def test_an_unspecified_sparse_arm_is_not_usable(self):
        from hands_lie_detector.audit.condition import ARM_A_UNSTATED, ContrastPoint

        self.assertFalse(ContrastPoint(ARM_A_UNSTATED, ARM_A_UNSTATED).usable)
