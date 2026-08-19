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


class TestWithinFrameControl(unittest.TestCase):
    """Same frame, same model: isolates reference class from perception."""

    @staticmethod
    def _control(a_ok: bool, b_ok: bool):
        from hands_lie_detector.audit import (
            BREED_TAXONOMY, DOMAIN_CONJUNCTION, Probe, WithinFrameControl,
        )

        return WithinFrameControl(
            stimulus_id="s1", date="2026-08-19", model_string="m",
            maintained_probe=Probe("breed?", BREED_TAXONOMY, a_ok),
            unmaintained_probe=Probe("what work?", DOMAIN_CONJUNCTION, b_ok),
        )

    def test_maintained_passes_unmaintained_fails_exonerates_perception(self):
        from hands_lie_detector.audit import ControlVerdict

        control = self._control(a_ok=True, b_ok=False)
        self.assertIs(control.verdict, ControlVerdict.ISOLATES_REFERENCE_CLASS)
        self.assertTrue(control.perception_exonerated)

    def test_both_failing_implicates_perception_instead(self):
        from hands_lie_detector.audit import ControlVerdict

        control = self._control(a_ok=False, b_ok=False)
        self.assertIs(control.verdict, ControlVerdict.PERCEPTION_IMPLICATED)
        self.assertFalse(control.perception_exonerated)

    def test_a_maintained_class_needs_all_three_supports(self):
        from hands_lie_detector.audit import ReferenceClassStatus

        partial = ReferenceClassStatus("x", True, True, False)
        self.assertFalse(partial.maintained)
        self.assertIn("a body maintaining the taxonomy", partial.missing)

    def test_a_miscast_probe_invalidates_the_design(self):
        from hands_lie_detector.audit import (
            BREED_TAXONOMY, ControlVerdict, Probe, WithinFrameControl,
        )

        control = WithinFrameControl(
            stimulus_id="s1", date="2026-08-19", model_string="m",
            maintained_probe=Probe("a", BREED_TAXONOMY, True),
            unmaintained_probe=Probe("b", BREED_TAXONOMY, False),
        )
        self.assertIs(control.verdict, ControlVerdict.INVALID)


class TestAttributionRetrofit(unittest.TestCase):
    """No ground truth anywhere: the delta and the binary are the findings."""

    def _three_arm(self, with_control=True):
        from hands_lie_detector.audit import Arm, ArmResponse, ThreeArmTest, VerbClass

        responses = {
            Arm.UNLABELED: ArmResponse(Arm.UNLABELED, True, VerbClass.HIGH_FORCE, 400, 6.0),
            Arm.STATED_WOMAN: ArmResponse(
                Arm.STATED_WOMAN, False, VerbClass.LOW_FORCE, 180, 2.5, 3, 2
            ),
        }
        if with_control:
            responses[Arm.STATED_MAN] = ArmResponse(
                Arm.STATED_MAN, True, VerbClass.HIGH_FORCE, 430, 6.5
            )
        return ThreeArmTest("f1", responses)

    def test_arm_c_is_not_optional(self):
        self.assertFalse(self._three_arm(with_control=False).has_control_arm)
        self.assertIn("WARNING", self._three_arm(with_control=False).report())

    def test_a_physical_delta_across_arms_is_the_finding(self):
        from hands_lie_detector.audit import Arm

        delta = self._three_arm().physical_delta(Arm.UNLABELED, Arm.STATED_WOMAN)
        self.assertLess(delta["force_estimate"], 0)
        self.assertLess(delta["duration_estimate"], 0)

    def test_attribution_delta_catches_actor_loss_and_verb_softening(self):
        from hands_lie_detector.audit import Arm

        delta = self._three_arm().attribution_delta(Arm.UNLABELED, Arm.STATED_WOMAN)
        self.assertTrue(delta["actor_lost"])
        self.assertTrue(delta["verb_softened"])
        self.assertEqual(delta["added_caveats"], 3)

    def test_revision_after_a_label_is_unambiguous(self):
        from hands_lie_detector.audit import SequencedLabelTest

        revised = SequencedLabelTest("f1", "she is operating it", "she is helping")
        stable = SequencedLabelTest("f1", "she is operating it", "she is operating it")
        self.assertTrue(revised.revised)
        self.assertIn("REVISION", revised.verdict())
        self.assertFalse(stable.revised)

    def test_no_destination_test_requires_a_window_without_a_candidate(self):
        from hands_lie_detector.audit import NoDestinationTest

        with self.assertRaises(ValueError):
            NoDestinationTest("window", second_party_present_or_implied_in_input=True)

    def test_an_invented_agent_in_that_window_is_fabrication(self):
        from hands_lie_detector.audit import NoDestinationTest, Severity

        t = NoDestinationTest("pre-marriage", False, "whoever helped her")
        self.assertTrue(t.false_positive)
        self.assertIs(t.severity, Severity.L2_NO_DESTINATION)
        self.assertIn("not arguable", t.report())

    def test_the_slot_hypothesis_predicts_an_unnamed_agent(self):
        from hands_lie_detector.audit import AgentSlotForm, InventedAgent

        slot = InventedAgent("whoever", AgentSlotForm.UNNAMED_REQUIRED, False)
        person = InventedAgent("her husband", AgentSlotForm.NAMED, False)
        self.assertTrue(slot.supports_slot_hypothesis)
        self.assertFalse(person.supports_slot_hypothesis)
        self.assertTrue(person.is_fabrication)

    def test_dose_response_separates_weight_from_constraint(self):
        from hands_lie_detector.audit import DoseResponse, Mechanism

        self.assertIs(DoseResponse([0.8, 0.5, 0.3, 0.1]).classify(), Mechanism.WEIGHT)
        self.assertIs(DoseResponse([0.8, 0.8, 0.75, 0.78]).classify(),
                      Mechanism.CONSTRAINT)
        self.assertIs(DoseResponse([0.8, 0.2], recovered_after_correction=True).classify(),
                      Mechanism.CONSTRAINT_CONFIRMED)

    def test_ranking_from_zero_bits_is_its_own_error(self):
        from hands_lie_detector.audit import estimate_without_bits

        self.assertIn("no estimate was available",
                      estimate_without_bits(0, produced_ordering=True))
        self.assertIn("accuracy", estimate_without_bits(12, produced_ordering=True))

    def test_the_two_asymmetry_columns_stay_separate(self):
        from hands_lie_detector.audit import ASYMMETRY_READING, AsymmetryColumn

        self.assertIn("uncounted",
                      ASYMMETRY_READING[AsymmetryColumn.UNCOUNTED_AGENT])
        self.assertIn("fabrication",
                      ASYMMETRY_READING[AsymmetryColumn.FABRICATED_AGENT])


class TestInstrumentValidity(unittest.TestCase):
    """A test that runs is not a test that measures."""

    def test_every_shipped_instrument_is_unvalidated(self):
        from hands_lie_detector.audit.attribution import DEFAULT_INSTRUMENTS

        self.assertTrue(DEFAULT_INSTRUMENTS)
        for instrument in DEFAULT_INSTRUMENTS:
            self.assertFalse(instrument.is_interpretable, instrument.field_name)

    def test_an_instrument_needs_both_operationalization_and_agreement(self):
        from hands_lie_detector.audit.attribution import ScoringInstrument

        self.assertFalse(ScoringInstrument("x", "a written rule").is_interpretable)
        self.assertFalse(ScoringInstrument("x", "", 0.9).is_interpretable)
        self.assertFalse(ScoringInstrument("x", "a rule", 0.4).is_interpretable)
        self.assertTrue(ScoringInstrument("x", "a rule", 0.85).is_interpretable)

    def test_the_three_arm_report_disclaims_its_own_numbers(self):
        from hands_lie_detector.audit import Arm, ArmResponse, ThreeArmTest, VerbClass

        t = ThreeArmTest("f1", {
            Arm.UNLABELED: ArmResponse(Arm.UNLABELED, True, VerbClass.HIGH_FORCE, 400, 6),
            Arm.STATED_WOMAN: ArmResponse(Arm.STATED_WOMAN, False, VerbClass.LOW_FORCE,
                                          180, 2.5),
            Arm.STATED_MAN: ArmResponse(Arm.STATED_MAN, True, VerbClass.HIGH_FORCE, 430, 6.5),
        })
        self.assertFalse(t.results_are_interpretable)
        self.assertIn("INSTRUMENTS UNVALIDATED", t.report())
        self.assertIn("marks, not measurements", t.report())

    def test_the_no_destination_test_needs_no_scale(self):
        """The one test that survives the instrument gap."""
        from hands_lie_detector.audit import NoDestinationTest

        t = NoDestinationTest("window", False, "whoever helped her")
        self.assertFalse(t.needs_validated_instruments)
        self.assertTrue(t.false_positive)

    def test_the_claim_is_marked_as_model_authored_testimony(self):
        from hands_lie_detector.audit.attribution import (
            CLAIM_PROVENANCE, INSTRUMENT_STATUS,
        )

        self.assertIn("RECONSTRUCTED", CLAIM_PROVENANCE)
        self.assertIn("not the operator's claim", CLAIM_PROVENANCE)
        self.assertIn("unvalidated", INSTRUMENT_STATUS)
