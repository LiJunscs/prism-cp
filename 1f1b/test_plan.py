"""Pure tests for TMF cohort planning and initial PP boundary constraints."""

import unittest

from plan import build_execution_plan, parse_stage_layer_patterns


class TMFExecutionPlanTest(unittest.TestCase):
    def build(self, pattern, *, microbatches=8, microbatch_size=1):
        stage_plans = parse_stage_layer_patterns(pattern, 2, 2, 4)
        return build_execution_plan(
            stage_plans,
            pp_size=2,
            stage_mesh_size=4,
            logical_microbatches=microbatches,
            logical_microbatch_size=microbatch_size,
            seq_length=4096,
            hidden_size=128,
        )

    def test_fixed_cp_degenerates_to_standard_1f1b(self):
        plan = self.build(
            "attention:4,attention:4;attention:4,attention:4"
        )
        self.assertEqual(plan.cohort_width, 1)
        self.assertEqual(plan.pipeline_cohorts, 8)
        self.assertEqual(plan.boundaries[0].wire_item_shape, (1024, 1, 128))

    def test_cp4_cp1_cp4_requires_four_microbatches(self):
        plan = self.build(
            "attention:4,gdn:1,attention:4;"
            "attention:4,gdn:1,attention:4"
        )
        self.assertEqual(plan.stage_cohort_widths, (4, 4))
        self.assertEqual(plan.cohort_width, 4)
        self.assertEqual(plan.pipeline_cohorts, 2)

    def test_global_cohort_is_lcm_of_stage_requirements(self):
        plan = self.build(
            "attention:4,gdn:1,attention:4;"
            "attention:4,gdn:2,attention:4"
        )
        self.assertEqual(plan.stage_cohort_widths, (4, 2))
        self.assertEqual(plan.cohort_width, 4)

    def test_rejects_incomplete_optimizer_step_cohort(self):
        with self.assertRaisesRegex(ValueError, "M=6.*K=4"):
            self.build(
                "attention:4,gdn:1,attention:4;"
                "attention:4,gdn:1,attention:4",
                microbatches=6,
            )

    def test_rejects_cross_stage_cp_transition(self):
        with self.assertRaisesRegex(ValueError, "forbids CP transitions"):
            self.build(
                "attention:4,gdn:1;attention:4,gdn:1",
                microbatch_size=4,
            )

    def test_rejects_non_unit_stage_boundary_width(self):
        with self.assertRaisesRegex(ValueError, "entry/exit temporal width 1"):
            self.build(
                "gdn:1,attention:4;attention:4,gdn:1",
                microbatch_size=1,
            )

    def test_pp1_mixed_cp_degenerates_without_pipeline_boundaries(self):
        stage_plans = parse_stage_layer_patterns(
            "attention:4,gdn:1,attention:4", 1, 3, 4
        )
        plan = build_execution_plan(
            stage_plans,
            pp_size=1,
            stage_mesh_size=4,
            logical_microbatches=8,
            logical_microbatch_size=1,
            seq_length=4096,
            hidden_size=128,
        )
        self.assertEqual(plan.stage_cohort_widths, (4,))
        self.assertEqual(plan.cohort_width, 4)
        self.assertEqual(plan.pipeline_cohorts, 2)
        self.assertEqual(plan.boundaries, ())

    def test_fixed_low_cp_boundary_requires_batch_divisible_by_dp(self):
        stage_plans = parse_stage_layer_patterns(
            "attention:1,attention:1", 1, 2, 8
        )
        with self.assertRaisesRegex(ValueError, "entry/exit temporal width 1"):
            build_execution_plan(
                stage_plans,
                pp_size=1,
                stage_mesh_size=8,
                logical_microbatches=8,
                logical_microbatch_size=1,
                seq_length=4096,
                hidden_size=128,
            )

        plan = build_execution_plan(
            stage_plans,
            pp_size=1,
            stage_mesh_size=8,
            logical_microbatches=8,
            logical_microbatch_size=8,
            seq_length=4096,
            hidden_size=128,
        )
        self.assertEqual(plan.cohort_width, 1)


if __name__ == "__main__":
    unittest.main()
