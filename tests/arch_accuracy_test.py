# SPDX-License-Identifier: Apache-2.0
"""Check the logit measures and calibration controls with independent small rows.

Inputs: no arguments or model files.
Outputs: unittest case counts and failures.
Exit codes: zero on success, nonzero on failure.
"""
import math
import unittest

import numpy as np

from arch_accuracy_metrics import assess, calibrate, measure, validate_bounds


class AccuracyTests(unittest.TestCase):
    def test_independent_two_token_distribution(self):
        # exp(log(3)) gives odds of three to one, and the other row reverses them.
        value = math.log(3)
        row = measure([value, 0], [0, value])
        self.assertAlmostEqual(row["linf"], value)
        self.assertAlmostEqual(row["relative_l2"], math.sqrt(2))
        self.assertAlmostEqual(row["total_variation"], 0.5)
        self.assertAlmostEqual(row["reference_gap"], value)
        self.assertEqual((row["reference_top"], row["device_top"]), (0, 1))

    def test_constant_shift_preserves_probabilities(self):
        row = measure([1, 2, 4], [6, 7, 9])
        self.assertEqual(row["linf"], 5)
        self.assertEqual(row["total_variation"], 0)

    def test_low_rank_tail_is_measured(self):
        reference = np.ones(128)
        device = reference.copy()
        device[-1] = -99
        self.assertEqual(measure(reference, device)["linf"], 100)

    def test_large_logits_keep_finite_probabilities(self):
        self.assertEqual(measure([10000, 9999], [10000, 9999])["total_variation"], 0)

    def test_nonfinite_values_fail_on_both_sides(self):
        for value in (math.nan, math.inf, -math.inf):
            for reference, device in (([1, value], [1, 2]), ([1, 2], [value, 1])):
                with self.subTest(value=value, reference=reference), self.assertRaises(ValueError):
                    measure(reference, device)

    def test_absent_short_wrong_shape_and_zero_norm_rows_fail(self):
        for reference, device in (([], []), ([1], [1]), ([1, 2], [1]),
                                  ([[1, 2]], [[1, 2]]), ([0, 0], [1, 2])):
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                measure(reference, device)

    def test_calibration_uses_only_its_split(self):
        row = dict(split="validation", linf=1, relative_l2=0.1, total_variation=0.1)
        with self.assertRaises(ValueError):
            calibrate([row])
        with self.assertRaises(ValueError):
            calibrate([])

    def test_fixed_calibration_and_validation_failure(self):
        row = dict(split="calibration", linf=2, relative_l2=0.25, total_variation=0.125)
        result = calibrate([row])
        self.assertEqual(result["bounds"], dict(linf=3, relative_l2=0.375, total_variation=0.1875))
        validation = dict(row, linf=3.01, reference_gap=0, reference_top=0, device_top=0)
        self.assertEqual(assess(validation, result["bounds"])["failed"], ["linf"])
        self.assertEqual(result["bounds"]["linf"], 3)

    def test_bounds_fail_when_absent_negative_nonfinite_or_trivial(self):
        for bounds in ({}, dict(linf=-1, relative_l2=0, total_variation=0),
                       dict(linf=math.inf, relative_l2=0, total_variation=0),
                       dict(linf=0, relative_l2=0, total_variation=1)):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                validate_bounds(bounds)

    def test_clear_margin_uses_twice_the_uniform_bound(self):
        bounds = dict(linf=0.125, relative_l2=0.5, total_variation=0.5)
        row = dict(linf=0, relative_l2=0, total_variation=0,
                   reference_gap=0.25, reference_top=0, device_top=1)
        self.assertFalse(assess(row, bounds)["clear"])
        row["reference_gap"] = 0.25001
        self.assertEqual(assess(row, bounds)["failed"], ["clear_argmax"])

    def test_each_bound_has_a_failing_case(self):
        bounds = dict(linf=0.1, relative_l2=0.1, total_variation=0.1)
        row = dict(bounds, reference_gap=0, reference_top=0, device_top=0)
        for key in bounds:
            with self.subTest(key=key):
                self.assertEqual(assess(dict(row, **{key: 0.101}), bounds)["failed"], [key])


if __name__ == "__main__":
    unittest.main()
