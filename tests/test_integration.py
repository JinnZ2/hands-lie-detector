"""
Tests for hands_lie_detector.integration.

These are mostly claim tests: each one asserts something the accompanying
documents state in prose, so that the prose cannot drift away from the code.

Run: python -m unittest discover tests
"""

import unittest
from dataclasses import fields

from hands_lie_detector.integration import (
    ClassificationSystem,
    LoadBlock,
    LoadHistory,
    PartitionVerdict,
    Seam,
    SeamKind,
    Verdict,
    Zone,
    boundary_audit,
    classify_relation,
    discontinuity,
    double_dissociation,
    load_share,
    propose_partition,
    read_hand,
    relabel,
    transferable_across_domains,
)
from hands_lie_detector.integration.carve_audit import CarveVerdict


# Illustrative parameters throughout. Not measurements.
PAID = LoadBlock("driving_rotary_clamp", hours=70, force=1.0,
                 shear_cycles_per_hour=40, geometry_mismatch=0.25)
UNPAID = LoadBlock("cob_wet_abrasive", hours=12, force=1.4,
                   shear_cycles_per_hour=120, geometry_mismatch=0.85)


class TestLoadWeight(unittest.TestCase):
    def test_load_block_has_no_payment_field(self):
        """The absence is the claim, so it gets asserted rather than assumed."""
        names = {f.name for f in fields(LoadBlock)}
        for forbidden in ("paid", "payment", "wage", "employed", "occupational"):
            self.assertNotIn(forbidden, names)

    def test_weight_invariant_under_relabeling(self):
        self.assertEqual(UNPAID.weight, relabel(UNPAID, "hobby").weight)
        self.assertEqual(UNPAID.weight, relabel(UNPAID, "trade").weight)

    def test_discontinuity_is_a_definitional_artifact(self):
        """Mechanics held fixed, payment status varied: readout jumps, body doesn't."""
        result = discontinuity([PAID, UNPAID], paid={PAID.name}, move=UNPAID.name)
        self.assertEqual(result.delta_physical, 0.0)
        self.assertNotEqual(result.delta_ledger, 0.0)
        self.assertTrue(result.is_artifact)

    def test_unpaid_block_can_dominate_the_physical_share(self):
        """The 1:0 weighting is not merely wrong, it can be inverted."""
        share = load_share([PAID, UNPAID])
        self.assertGreater(share[UNPAID.name], share[PAID.name])


class TestPartition(unittest.TestCase):
    HISTORY = LoadHistory(
        frozenset({Zone.THUMB_CROTCH, Zone.PALM_BELOW_INDEX, Zone.BASE_OF_FINGERS,
                   Zone.FINGERTIP_PADS, Zone.HEEL_OF_PALM})
    )

    def test_unearned_partition_returns_unpartitioned_history(self):
        claim = propose_partition(self.HISTORY, ["keyboard", "fine_manipulation"])
        self.assertIs(claim.verdict, PartitionVerdict.NOT_EARNED)
        self.assertIsInstance(claim.result(), LoadHistory)

    def test_coarse_registry_reports_degenerate_not_a_verdict(self):
        """With 8 domains, no single-domain split can reach alpha=0.05."""
        claim = propose_partition(self.HISTORY, ["keyboard"])
        self.assertIs(claim.verdict, PartitionVerdict.DEGENERATE)
        self.assertIn("too coarse", claim.notes)
        self.assertIsInstance(claim.result(), LoadHistory)

    def test_full_registry_partition_is_degenerate(self):
        from hands_lie_detector.integration import DEFAULT_DOMAINS

        claim = propose_partition(self.HISTORY, sorted(DEFAULT_DOMAINS))
        self.assertIs(claim.verdict, PartitionVerdict.DEGENERATE)

    def test_verdict_is_reproducible(self):
        a = propose_partition(self.HISTORY, ["rotary_hand_tool", "wet_task"])
        b = propose_partition(self.HISTORY, ["rotary_hand_tool", "wet_task"])
        self.assertEqual(a.p_value, b.p_value)
        self.assertIs(a.verdict, b.verdict)

    def test_unknown_domain_raises(self):
        with self.assertRaises(KeyError):
            propose_partition(self.HISTORY, ["blacksmithing_on_tuesdays"])


class TestResidual(unittest.TestCase):
    def test_residual_is_never_attributed_to_an_enrolled_domain(self):
        readout = read_hand(
            ["thumb_crotch", "palm_below_index", "fingertip_pads", "outer_palm_edge"],
            ["rotary_hand_tool", "wet_task"],
        )
        self.assertIn(Zone.OUTER_PALM_EDGE, readout.residual_zones)
        self.assertFalse(readout.residual_zones & readout.predicted)

    def test_defaults_are_flagged_as_stipulated(self):
        from hands_lie_detector.integration import DEFAULT_DOMAINS

        self.assertTrue(DEFAULT_DOMAINS)
        for name, sig in DEFAULT_DOMAINS.items():
            self.assertFalse(sig.is_evidence_based, f"{name} claims evidence it lacks")

    def test_systemic_conflicts_are_flagged_post_hoc(self):
        readout = read_hand(["thumb_crotch"], ["rotary_hand_tool", "wet_task"])
        systemic = [c for c in readout.conflicts if c.systemic]
        self.assertTrue(systemic)
        self.assertFalse(any(c.is_evidence_based for c in systemic))


class TestDissociation(unittest.TestCase):
    NOISE = {"thumb_crotch_depth": 0.4, "recovery_lag_days": 0.5}

    def test_crossing_rejects_h1(self):
        result = double_dissociation(
            {"thumb_crotch_depth": 1.6, "recovery_lag_days": 0.1},
            {"thumb_crotch_depth": 0.2, "recovery_lag_days": 2.4},
            self.NOISE,
        )
        self.assertIs(result.verdict, Verdict.DOUBLE_DISSOCIATION)
        self.assertTrue(result.rejects_h1)

    def test_single_dissociation_is_not_support_for_h2(self):
        result = double_dissociation(
            {"thumb_crotch_depth": 1.6, "recovery_lag_days": 0.1},
            {"thumb_crotch_depth": 0.1, "recovery_lag_days": 0.1},
            self.NOISE,
        )
        self.assertIs(result.verdict, Verdict.SINGLE_DISSOCIATION)
        self.assertFalse(result.rejects_h1)

    def test_movement_is_judged_against_the_carriers_own_noise(self):
        deltas = {"thumb_crotch_depth": 1.0, "recovery_lag_days": 1.0}
        quiet = double_dissociation(deltas, deltas, {"thumb_crotch_depth": 0.1,
                                                     "recovery_lag_days": 0.1})
        noisy = double_dissociation(deltas, deltas, {"thumb_crotch_depth": 5.0,
                                                    "recovery_lag_days": 5.0})
        self.assertIs(quiet.verdict, Verdict.NO_DISSOCIATION)
        self.assertIs(noisy.verdict, Verdict.INSUFFICIENT_MOVEMENT)


class TestCarveAudit(unittest.TestCase):
    def test_incidence_relations_do_not_transfer(self):
        for term in ("incidence", "lifetime prevalence in welders", "exposure limit"):
            self.assertFalse(transferable_across_domains(term), term)

    def test_mechanism_relations_transfer(self):
        for term in ("shear delamination", "stiffness mismatch", "creep"):
            self.assertTrue(transferable_across_domains(term), term)

    def test_unknown_relations_fail_closed(self):
        self.assertFalse(transferable_across_domains("grip endurance"))
        self.assertEqual(classify_relation("grip endurance").kind.value, "unknown")

    def test_audit_ships_unrun(self):
        from hands_lie_detector.integration import SYSTEM_REGISTRY

        self.assertEqual(SYSTEM_REGISTRY, {})
        result = boundary_audit(ClassificationSystem("SOC"), ClassificationSystem("ICD"))
        self.assertIs(result.verdict, CarveVerdict.INSUFFICIENT_DATA)

    def test_pay_seam_alignment_reads_as_co_authored(self):
        seam = Seam("employment status", SeamKind.PAY_CODE, ("employed", "self"))
        result = boundary_audit(
            ClassificationSystem("SOC", seams=(seam,), roster=frozenset({"x"})),
            ClassificationSystem("ICD", seams=(seam,), roster=frozenset({"x"})),
        )
        self.assertIs(result.verdict, CarveVerdict.CO_AUTHORED)

    def test_physical_seam_alignment_reads_as_convergent_not_authorship(self):
        """The correction to the discriminator: alignment alone is not authorship."""
        seam = Seam("grip mode", SeamKind.FORCE_GEOMETRY, ("power", "precision"))
        result = boundary_audit(
            ClassificationSystem("A", seams=(seam,)),
            ClassificationSystem("B", seams=(seam,)),
        )
        self.assertIs(result.verdict, CarveVerdict.CONVERGENT)


if __name__ == "__main__":
    unittest.main()


class TestStrip(unittest.TestCase):
    """The strip: render a category noun into the units of the governing equation."""

    def test_mechanic_and_hobby_render_identically_so_the_category_drops_out(self):
        from hands_lie_detector.integration import StripVerdict, strip

        self.assertIs(strip("mechanic", "hobby").verdict, StripVerdict.DROPS_OUT)

    def test_a_noun_with_no_load_referent_is_a_ledger_class(self):
        from hands_lie_detector.integration import StripVerdict, strip

        for noun in ("occupation", "employment status", "risk class", "SOC code"):
            self.assertIs(strip(noun).verdict, StripVerdict.LEDGER_CLASS, noun)

    def test_every_default_band_label_strips_to_a_ledger_class(self):
        """This repo's own scale, run through this repo's own diagnostic."""
        from hands_lie_detector.integration import StripVerdict, strip_all

        labels = ["podcast hands", "casual hobbyist", "working hands",
                  "experienced trade", "field work"]
        verdicts = strip_all(labels)
        self.assertTrue(all(v is StripVerdict.LEDGER_CLASS for v in verdicts.values()),
                        verdicts)

    def test_unregistered_nouns_fail_closed(self):
        from hands_lie_detector.integration import StripVerdict, strip

        self.assertIs(strip("cordwainer").verdict, StripVerdict.UNREGISTERED)

    def test_operator_can_register_a_rendering(self):
        from hands_lie_detector.integration import StripVerdict, strip
        from hands_lie_detector.integration.strip import (
            DEFAULT_RENDERINGS, MECHANICAL_UNITS, Rendering, register,
        )

        registry = dict(DEFAULT_RENDERINGS)
        register(Rendering("cordwainer", MECHANICAL_UNITS), registry)
        self.assertIs(
            strip("cordwainer", "hobby", registry=registry).verdict,
            StripVerdict.DROPS_OUT,
        )


class TestSeam(unittest.TestCase):
    """Relations transfer, coefficients don't — the living-tissue seam."""

    def test_governing_relations_transfer_in_full(self):
        from hands_lie_detector.integration import TransferScope, classify_relation

        for term in ("shear delamination", "creep", "stress concentration"):
            v = classify_relation(term)
            self.assertIs(v.transfer, TransferScope.FULL, term)
            self.assertTrue(v.magnitude_transfers, term)

    def test_living_tissue_parameters_transfer_only_relationally(self):
        from hands_lie_detector.integration import TransferScope, classify_relation

        for term in ("stiffness", "fatigue limit", "adaptation rate",
                     "hydration response"):
            v = classify_relation(term)
            self.assertIs(v.transfer, TransferScope.RELATIONAL_ONLY, term)
            self.assertTrue(v.transferable, term)
            self.assertFalse(v.magnitude_transfers, term)

    def test_engineered_material_parameters_are_not_at_the_seam(self):
        from hands_lie_detector.integration import TransferScope, classify_relation

        self.assertIs(classify_relation("steel modulus").transfer, TransferScope.FULL)

    def test_specific_terms_beat_substrings(self):
        """'fatigue limit' is at the seam; 'fatigue' is a clean relation."""
        from hands_lie_detector.integration import RelationKind, classify_relation

        self.assertIs(classify_relation("fatigue").kind, RelationKind.GOVERNING)
        self.assertIs(classify_relation("fatigue limit").kind,
                      RelationKind.MATERIAL_LIVING)

    def test_retrieval_splits_three_ways(self):
        from hands_lie_detector.integration import retrieve_mechanism_first

        full, relational, refused = retrieve_mechanism_first(
            ["creep", "elastic modulus", "prevalence", "contact geometry"]
        )
        self.assertEqual(sorted(full), ["contact geometry", "creep"])
        self.assertEqual(relational, ["elastic modulus"])
        self.assertEqual(refused, ["prevalence"])


class TestGatedForm(unittest.TestCase):
    """Layers gate; they do not add. Wrong form, not wrong number."""

    @staticmethod
    def _stack(env=0.9, cap=1.0, draw=70.0):
        from hands_lie_detector.integration import GatedStack, Stratum, StratumState

        return GatedStack({
            Stratum.ENVIRONMENT: StratumState(Stratum.ENVIRONMENT, env),
            Stratum.CAPACITY: StratumState(Stratum.CAPACITY, cap),
            Stratum.JOB: StratumState(Stratum.JOB, 1.0, draw=draw),
        })

    def test_a_failed_lower_layer_zeroes_the_whole_product(self):
        self.assertEqual(self._stack(cap=0.0).output(), 0.0)
        self.assertTrue(self._stack(cap=0.0).collapsed)

    def test_the_additive_form_cannot_reach_zero(self):
        """No coefficient assignment repairs this. That is the claim."""
        stack = self._stack(cap=0.0)
        self.assertEqual(stack.output(), 0.0)
        self.assertGreater(stack.additive_output(), 0.0)

    def test_sensitivity_is_multiplicative_not_constant(self):
        from hands_lie_detector.integration import Stratum

        low = self._stack(env=0.5).sensitivity(Stratum.CAPACITY)
        high = self._stack(env=1.0).sensitivity(Stratum.CAPACITY)
        self.assertNotAlmostEqual(low, high)

    def test_the_ledger_reverses_the_arrow_on_the_base_layers(self):
        from hands_lie_detector.integration import LEDGER_SIGN, PHYSICS_SIGN, Stratum

        self.assertEqual(LEDGER_SIGN[Stratum.CAPACITY], "consumption")
        self.assertIn("production", PHYSICS_SIGN[Stratum.CAPACITY])
        self.assertEqual(LEDGER_SIGN[Stratum.JOB], "production")
        self.assertIn("draw", PHYSICS_SIGN[Stratum.JOB])

    def test_band_maintenance_is_capacity_solvency(self):
        """Both exits from the band are insolvency, for different reasons."""
        from hands_lie_detector.band import BandPosition
        from hands_lie_detector.integration import solvency_from_band

        in_band = solvency_from_band(BandPosition.IN_BAND).solvency
        saturated = solvency_from_band(BandPosition.OUT_SATURATED).solvency
        soft = solvency_from_band(BandPosition.OUT_SOFT).solvency
        self.assertGreater(in_band, saturated)
        self.assertGreater(in_band, soft)

    def test_an_unresolved_band_position_is_not_solvent(self):
        """The map alone cannot establish solvency. Unresolved is not solvent."""
        from hands_lie_detector.band import BandPosition
        from hands_lie_detector.integration import solvency_from_band

        self.assertEqual(solvency_from_band(BandPosition.UNRESOLVED).solvency, 0.0)

    def test_saturation_spends_capacity_to_fund_near_term_output(self):
        from hands_lie_detector.band import BandPosition
        from hands_lie_detector.integration import (
            GatedStack, Stratum, StratumState, solvency_from_band,
        )

        def stack_for(position):
            return GatedStack({
                Stratum.ENVIRONMENT: StratumState(Stratum.ENVIRONMENT, 1.0),
                Stratum.CAPACITY: solvency_from_band(position),
                Stratum.JOB: StratumState(Stratum.JOB, 1.0, draw=70.0),
            })

        self.assertGreater(stack_for(BandPosition.IN_BAND).output(),
                           stack_for(BandPosition.OUT_SATURATED).output())

    def test_missing_strata_are_rejected(self):
        from hands_lie_detector.integration import GatedStack, Stratum, StratumState

        with self.assertRaises(KeyError):
            GatedStack({Stratum.JOB: StratumState(Stratum.JOB, 1.0, draw=1.0)})

    def test_solvency_outside_unit_range_is_rejected(self):
        from hands_lie_detector.integration import Stratum, StratumState

        with self.assertRaises(ValueError):
            StratumState(Stratum.CAPACITY, 1.5)


class TestDorsalSurface(unittest.TestCase):
    """The zone vocabulary was palmar-only. Specimen 003 found the gap."""

    def test_dorsal_zones_exist_and_are_typed(self):
        from hands_lie_detector.integration import DORSAL_ZONES, PALMAR_ZONES, Surface

        self.assertTrue(DORSAL_ZONES)
        self.assertFalse(DORSAL_ZONES & PALMAR_ZONES)
        for z in DORSAL_ZONES:
            self.assertIs(z.surface, Surface.DORSAL)

    def test_no_shipped_domain_predicts_a_dorsal_zone(self):
        """An instrument built for grip, handed a strike."""
        from hands_lie_detector.integration import DEFAULT_DOMAINS, DORSAL_ZONES

        predicted = set().union(*(s.zones for s in DEFAULT_DOMAINS.values()))
        self.assertFalse(predicted & DORSAL_ZONES)

    def test_dorsal_markers_come_back_unexplained_not_generic(self):
        from hands_lie_detector.integration import Zone, read_hand

        readout = read_hand(
            ["base_of_fingers", "dorsal_metacarpal", "dorsal_mcp_knuckles"],
            ["rotary_hand_tool"],
        )
        self.assertIn(Zone.DORSAL_METACARPAL, readout.unexplained_residual)
        self.assertNotIn(Zone.DORSAL_METACARPAL, readout.generic_residual)

    def test_dorsal_adjacency_does_not_leak_into_the_grip_graph(self):
        from hands_lie_detector.integration import Zone
        from hands_lie_detector.integration.domains import ADJACENCY, Surface

        for zone in (Zone.DORSAL_MCP_KNUCKLES, Zone.DORSAL_WEB_SPACE,
                     Zone.DORSAL_PHALANX):
            for neighbour in ADJACENCY[zone]:
                self.assertIs(neighbour.surface, Surface.DORSAL, f"{zone}->{neighbour}")


class TestEventLog(unittest.TestCase):
    """Dorsal marks are events, not load history. Different instrument, clock."""

    def test_the_log_never_carries_load_history(self):
        from hands_lie_detector.integration import EventLog

        self.assertFalse(EventLog().carries_load_history)

    def test_palmar_zones_are_refused(self):
        from hands_lie_detector.integration import DorsalMark, Zone

        with self.assertRaises(ValueError):
            DorsalMark(Zone.THUMB_CROTCH, "2026-01-05")

    def test_external_request_gating_blocks_rate_claims(self):
        from hands_lie_detector.integration import DorsalMark, EventLog, Zone

        log = EventLog(
            marks=[DorsalMark(Zone.DORSAL_METACARPAL, "2026-01-05")],
            sampling_gate="external request during a phone call",
        )
        self.assertFalse(log.supports_rate_claims)
        self.assertIn("NOT LICENSED", log.report())
        self.assertIn("few marks does not mean few events", log.report())

    def test_an_unstated_gate_also_blocks_rate_claims(self):
        from hands_lie_detector.integration import EventLog

        self.assertFalse(EventLog().supports_rate_claims)

    def test_conditions_not_load_are_what_the_log_reports(self):
        from hands_lie_detector.integration import DorsalMark, EventLog, MarkKind, Zone

        log = EventLog(marks=[
            DorsalMark(Zone.DORSAL_METACARPAL, "2026-01-05", MarkKind.LACERATION),
            DorsalMark(Zone.DORSAL_PHALANX, "2026-01-19", MarkKind.SPLIT),
        ])
        signature = " ".join(log.condition_signature())
        self.assertIn("edge density", signature)
        self.assertIn("elasticity", signature)


class TestWearTaxonomy(unittest.TestCase):
    def test_every_load_mode_maps_to_a_standard_wear_mode(self):
        from hands_lie_detector.integration import LOAD_MODE_TO_WEAR
        from hands_lie_detector.integration.domains import LoadMode

        for mode in LoadMode:
            self.assertIn(mode, LOAD_MODE_TO_WEAR)

    def test_a_hand_without_a_counterface_is_half_a_specimen(self):
        from hands_lie_detector.integration import WearSystem

        self.assertFalse(WearSystem("across_palm_crease").is_complete_specimen)
        self.assertTrue(
            WearSystem("across_palm_crease", counterface="rope").is_complete_specimen
        )

    def test_the_residual_without_analogue_is_stated_in_code(self):
        from hands_lie_detector.integration import RESIDUAL_WITHOUT_ANALOGUE

        self.assertIn("remodels", RESIDUAL_WITHOUT_ANALOGUE.lower())


class TestDepositDrawSign(unittest.TestCase):
    """Third form error in the stack: weights, gates, and now sign."""

    def test_draw_domains_deposit_nothing(self):
        from hands_lie_detector.integration import DEFAULT_DOMAINS, LoadClass

        probe = DEFAULT_DOMAINS["probe_work"]
        self.assertFalse(probe.deposits)
        self.assertTrue(probe.draws)
        self.assertIn(LoadClass.DRAW_SPEND, probe.load_classes)

    def test_some_domains_deposit_and_suppress_at_once(self):
        """The open-station tractor is the first. Chainsaw suppresses only."""
        from hands_lie_detector.integration import DEFAULT_DOMAINS

        self.assertTrue(DEFAULT_DOMAINS["tractor_open_station"].decouples_map_from_band)
        self.assertFalse(DEFAULT_DOMAINS["chainsaw"].decouples_map_from_band)
        self.assertFalse(DEFAULT_DOMAINS["shovel_haul"].decouples_map_from_band)

    def test_the_balance_names_the_sign_problem(self):
        from hands_lie_detector.integration import deposit_draw_balance

        report = deposit_draw_balance(
            ["rotary_hand_tool", "probe_work", "tractor_open_station"]
        )
        self.assertIn("opposite signs", report)
        self.assertIn("DECOUPLING", report)

    def test_variable_geometry_domain_spreads_across_many_zones(self):
        """firewood_handling is the case that broke the contrast metric."""
        from hands_lie_detector.integration import DEFAULT_DOMAINS

        self.assertGreaterEqual(len(DEFAULT_DOMAINS["firewood_handling"].zones), 5)


class TestBundlesAndNarrativeGate(unittest.TestCase):
    def test_firewood_is_a_bundle_of_four_sub_domains(self):
        from hands_lie_detector.integration import DEFAULT_DOMAINS

        bundle = DEFAULT_DOMAINS["firewood_handling"]
        self.assertTrue(bundle.is_bundle)
        self.assertEqual(len(bundle.components), 4)
        for name in bundle.components:
            self.assertIn(name, DEFAULT_DOMAINS)

    def test_sub_domains_carry_different_load_classes(self):
        """One word, four contact distributions, not all on the same ledger side."""
        from hands_lie_detector.integration import DEFAULT_DOMAINS

        classes = {
            frozenset(DEFAULT_DOMAINS[n].load_classes)
            for n in DEFAULT_DOMAINS["firewood_handling"].components
        }
        self.assertGreater(len(classes), 1)

    def test_the_tractor_deposits_and_suppresses(self):
        from hands_lie_detector.integration import DEFAULT_DOMAINS

        self.assertTrue(DEFAULT_DOMAINS["tractor_open_station"].decouples_map_from_band)

    def test_rotary_deposits_only_and_chainsaw_suppresses_only(self):
        from hands_lie_detector.integration import DEFAULT_DOMAINS, LoadClass

        self.assertEqual(DEFAULT_DOMAINS["rotary_hand_tool"].load_classes,
                         frozenset({LoadClass.DEPOSIT}))
        self.assertEqual(DEFAULT_DOMAINS["chainsaw"].load_classes,
                         frozenset({LoadClass.DRAW_SUPPRESS}))

    def test_an_event_gated_log_has_no_baseline_coverage(self):
        """Gate 4: steady state is missing from every arm, not just this one."""
        from hands_lie_detector.integration import DorsalMark, EventLog, Zone

        log = EventLog(marks=[DorsalMark(Zone.DORSAL_PHALANX, "2026-01-05")])
        self.assertFalse(log.has_baseline_coverage)
        self.assertIn("boring frames", log.report())
        self.assertTrue(EventLog(baseline_frames=3).has_baseline_coverage)


class TestCardinalityReduction(unittest.TestCase):
    """Three moves, one operation: a reduction with the step deleted."""

    def test_waste_heat_unwelds_into_two_variables(self):
        from hands_lie_detector.integration.strip import ReductionKind, unweld

        r = unweld("waste heat")
        self.assertIs(r.kind, ReductionKind.VARIABLE_WELD)
        self.assertTrue(r.undeclared)
        self.assertIn("required load", r.actual)

    def test_strange_is_a_frequency_claim_with_a_built_denominator(self):
        from hands_lie_detector.integration.strip import ReductionKind, unweld

        r = unweld("strange")
        self.assertIs(r.kind, ReductionKind.POPULATION)
        self.assertIn("MODAL", r.actual)

    def test_unknown_terms_return_none_rather_than_a_guess(self):
        from hands_lie_detector.integration.strip import unweld

        self.assertIsNone(unweld("thermal mass"))

    def test_confusion_is_recorded_as_a_detection_event(self):
        from hands_lie_detector.integration.strip import CONFUSION_IS_A_DETECTION_EVENT

        self.assertIn("detection event", CONFUSION_IS_A_DETECTION_EVENT)


class TestCounterfaces(unittest.TestCase):
    def test_feet_have_a_counterface_like_hands_do(self):
        from hands_lie_detector.integration.wear import BOOT_WEAR_ITEMS, CONJUGATE_PAIRS

        self.assertEqual(CONJUGATE_PAIRS["foot"], "boot")
        self.assertEqual(CONJUGATE_PAIRS["hand"], "tool handle")
        self.assertGreaterEqual(len(BOOT_WEAR_ITEMS), 5)

    def test_caretaking_deposits_and_is_readable_by_the_same_taxonomy(self):
        from hands_lie_detector.integration import DEFAULT_DOMAINS
        from hands_lie_detector.integration.wear import wear_mode

        caretaking = DEFAULT_DOMAINS["caretaking"]
        self.assertTrue(caretaking.deposits)
        for mode in caretaking.zone_modes.values():
            self.assertIsNotNone(wear_mode(mode))
