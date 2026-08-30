"""
Tests for the channel split and the precedence probe spec.

Claim tests. The load-bearing ones here are the non-overlap of the two failure
lists, the fragility of a fused claim, and the probe refusing to report a rate
it does not have.

Run: python -m unittest discover tests
"""

import unittest

from hands_lie_detector.channel import (
    FAILURE_MODES,
    Annotation,
    Channel,
    ChannelSet,
    Claim,
    PhysicalRecord,
    Support,
)


class TestChannelSplit(unittest.TestCase):
    def test_the_two_failure_lists_do_not_overlap(self):
        """The reason for the split: discounting one does not touch the other."""
        ch1 = {m.split(" — ")[0] for m in FAILURE_MODES[Channel.PHYSICAL_RECORD]}
        ch2 = {m.split(" — ")[0] for m in FAILURE_MODES[Channel.ANNOTATION]}
        self.assertFalse(ch1 & ch2)

    def test_ch1_has_no_field_for_a_caption(self):
        """Mute by construction, not by convention."""
        from dataclasses import fields

        names = {f.name for f in fields(PhysicalRecord)}
        for forbidden in ("caption", "annotation", "text", "note", "interpretation"):
            self.assertNotIn(forbidden, names)

    def test_ch1_cannot_be_dishonest(self):
        record = PhysicalRecord("r1", "2026-08-26", "image")
        self.assertFalse(record.can_be_dishonest)

    def test_citation_runs_one_direction_and_must_land(self):
        cs = ChannelSet()
        cs.add_record(PhysicalRecord("r1", "d", "image"))
        cs.add_annotation(Annotation("a1", "r1", "text"))
        with self.assertRaises(KeyError):
            cs.add_annotation(Annotation("a2", "nonexistent", "text"))

    def test_an_annotation_written_later_exposes_memory(self):
        later = Annotation("a1", "r1", "t")
        contemporaneous = Annotation("a2", "r1", "t",
                                     recorded_at_time_of_capture=True)
        self.assertTrue(later.memory_exposed)
        self.assertFalse(contemporaneous.memory_exposed)


class TestClaimSupport(unittest.TestCase):
    def test_a_ch1_claim_survives_the_annotation_being_wrong(self):
        claim = Claim("the patch sits in the skin plane", rests_on_records=("r1",))
        self.assertIs(claim.support, Support.CH1_ONLY)
        self.assertTrue(claim.survives_ch2_failure)

    def test_a_ch2_claim_survives_the_record_being_unreadable(self):
        claim = Claim("it happened during board work",
                      rests_on_annotations=("a1",))
        self.assertTrue(claim.survives_ch1_failure)

    def test_a_fused_claim_is_more_fragile_not_less(self):
        """Resting on both channels means falling if either falls."""
        claim = Claim("the patch is carbon from board work",
                      rests_on_records=("r1",), rests_on_annotations=("a1",))
        self.assertTrue(claim.falls_with_either)
        self.assertFalse(claim.survives_ch1_failure)
        self.assertFalse(claim.survives_ch2_failure)
        self.assertIn("FUSED", claim.report())

    def test_an_unsupported_claim_is_named_as_such(self):
        self.assertIs(Claim("assertion").support, Support.NEITHER)


class TestPrecedenceProbeSpec(unittest.TestCase):
    """SPEC ONLY. These assert the spec's shape, not any result."""

    def test_the_probe_ships_unrun_and_reports_no_rate(self):
        from hands_lie_detector.audit import Localization, PrecedenceProbe

        probe = PrecedenceProbe()
        self.assertFalse(probe.is_run)
        self.assertIs(probe.localize(), Localization.INCONCLUSIVE)
        self.assertIn("SPEC ONLY", probe.report())
        self.assertNotIn("insertion=", probe.report())

    def test_stock_input_is_refused_by_the_spec(self):
        from hands_lie_detector.audit import ProbeInput, SubjectArm

        stock = ProbeInput("i1", "r1", SubjectArm.HUMAN, "female",
                           second_agent_in_frame=False, template_absent=False)
        self.assertTrue(any("recognition" in p for p in stock.problems))

    def test_a_caption_travelling_with_ch1_is_refused(self):
        from hands_lie_detector.audit import ProbeInput, SubjectArm

        captioned = ProbeInput("i1", "r1", SubjectArm.HUMAN, "female",
                               second_agent_in_frame=False, template_absent=True,
                               mute=False)
        self.assertTrue(any("fuses the channels" in p for p in captioned.problems))

    def test_the_metric_needs_no_second_agent_in_frame(self):
        from hands_lie_detector.audit import ProbeInput, SubjectArm

        present = ProbeInput("i1", "r1", SubjectArm.HUMAN, "female",
                             second_agent_in_frame=True, template_absent=True)
        self.assertFalse(present.valid_for_insertion_metric)

    def test_precedence_is_position_not_content(self):
        """Same insertion, different position, different verdict."""
        from hands_lie_detector.audit import TrialScore

        before = TrialScore("i", "m", "d", first_physical_feature_index=4,
                            first_role_referent_index=1,
                            role_referent_text="his hands")
        after = TrialScore("i", "m", "d", first_physical_feature_index=1,
                           first_role_referent_index=4,
                           role_referent_text="his hands")
        self.assertTrue(before.inserted_an_agent)
        self.assertTrue(after.inserted_an_agent)
        self.assertTrue(before.precedence_violation)
        self.assertFalse(after.precedence_violation)
        self.assertTrue(after.ordering_held)

    def test_never_observing_at_all_is_a_violation_not_a_null(self):
        from hands_lie_detector.audit import TrialScore

        t = TrialScore("i", "m", "d", first_physical_feature_index=None,
                       first_role_referent_index=0, role_referent_text="he")
        self.assertTrue(t.precedence_violation)
        self.assertFalse(t.ordering_held)

    def test_a_referent_in_frame_is_description_not_insertion(self):
        from hands_lie_detector.audit import TrialScore

        t = TrialScore("i", "m", "d", first_physical_feature_index=4,
                       first_role_referent_index=1, referent_in_frame=True)
        self.assertFalse(t.inserted_an_agent)
        self.assertFalse(t.precedence_violation)

    def test_the_animal_arm_localizes_the_violation(self):
        from hands_lie_detector.audit import (
            ArmResult, Localization, PrecedenceProbe, SubjectArm, TrialScore,
        )

        def violating():
            return TrialScore("i", "m", "d", first_physical_feature_index=4,
                              first_role_referent_index=1, role_referent_text="he")

        def clean():
            return TrialScore("i", "m", "d", first_physical_feature_index=1)

        probe = PrecedenceProbe(
            human=ArmResult(SubjectArm.HUMAN, [violating(), violating()]),
            animal=ArmResult(SubjectArm.ANIMAL, [clean(), clean()]),
        )
        self.assertIs(probe.localize(), Localization.HUMAN_ROLE_PRIOR)

    def test_both_arms_violating_reads_as_weak_observation(self):
        from hands_lie_detector.audit import (
            ArmResult, Localization, PrecedenceProbe, SubjectArm, TrialScore,
        )

        def violating():
            return TrialScore("i", "m", "d", first_physical_feature_index=4,
                              first_role_referent_index=1, role_referent_text="he")

        probe = PrecedenceProbe(
            human=ArmResult(SubjectArm.HUMAN, [violating()]),
            animal=ArmResult(SubjectArm.ANIMAL, [violating()]),
        )
        self.assertIs(probe.localize(), Localization.GENERAL_WEAK_VISION)

    def test_both_scope_flags_are_raised_in_code(self):
        from hands_lie_detector.audit import NODE_INDEPENDENCE_FLAG, SAME_NODE_FLAG

        self.assertIn("DISSIMILAR models", SAME_NODE_FLAG)
        self.assertIn("RECONSTRUCTED", SAME_NODE_FLAG)
        self.assertIn("must not share a node", NODE_INDEPENDENCE_FLAG)


if __name__ == "__main__":
    unittest.main()
