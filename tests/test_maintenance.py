"""Bakım modülü testleri.

Bu testler sen src/rul/maintenance.py'yi yazana kadar ATLANIR (skipped).
Fonksiyonları yazdıkça geçmeye başlarlar. Hiçbir testi değiştirmeden geçirmeye çalış.
"""

import unittest

import pandas as pd

from rul import maintenance
from rul.maintenance import CostModel

SKIP_REASON = "maintenance.py henüz yazılmadı (TODO)"


def _implemented(func, *args):
    try:
        func(*args)
        return True
    except NotImplementedError:
        return False


@unittest.skipUnless(_implemented(maintenance.should_maintain, 0.0, 1.0), SKIP_REASON)
class ShouldMaintainTest(unittest.TestCase):
    def test_below_equal_above(self):
        self.assertTrue(maintenance.should_maintain(8, 10))
        self.assertTrue(maintenance.should_maintain(10, 10))
        self.assertFalse(maintenance.should_maintain(15, 10))


@unittest.skipUnless(_implemented(maintenance.unit_outcome, 5, True, CostModel()), SKIP_REASON)
class UnitOutcomeTest(unittest.TestCase):
    def setUp(self):
        self.costs = CostModel(cost_per_wasted_cycle=2.0, cost_of_failure=100.0, inspection_interval=10)

    def test_maintained_engine_wastes_remaining_life(self):
        out = maintenance.unit_outcome(30, True, self.costs)
        self.assertEqual(out, {"cost": 60.0, "failed": False, "wasted_cycles": 30})

    def test_waiting_engine_fails_before_inspection(self):
        out = maintenance.unit_outcome(10, False, self.costs)
        self.assertEqual(out, {"cost": 100.0, "failed": True, "wasted_cycles": 0})

    def test_waiting_engine_is_safe(self):
        out = maintenance.unit_outcome(11, False, self.costs)
        self.assertEqual(out, {"cost": 0.0, "failed": False, "wasted_cycles": 0})


def _toy_predictions():
    return pd.DataFrame({
        "true_rul": [5, 40, 8, 90],
        "pred":     [12, 35, 20, 80],
        "lower":    [4, 25, 9, 60],
    })


@unittest.skipUnless(
    _implemented(maintenance.simulate_policy, _toy_predictions(), "pred", 10.0), SKIP_REASON
)
class SimulatePolicyTest(unittest.TestCase):
    def test_point_estimate_policy(self):
        # eşik 10, nokta tahmini: hiçbir motor bakıma alınmaz (12, 35, 20, 80 hepsi > 10)
        # true_rul 5 ve 8 olan motorlar kontrolden önce arızalanır -> 2 * 100
        result = maintenance.simulate_policy(_toy_predictions(), "pred", 10)
        self.assertEqual(result["n_maintained"], 0)
        self.assertEqual(result["n_failures"], 2)
        self.assertAlmostEqual(result["total_cost"], 200.0)
        self.assertAlmostEqual(result["mean_wasted_cycles"], 0.0)

    def test_lower_bound_policy(self):
        # eşik 10, alt sınır: 4 ve 9 -> bakım (boşa: 5 + 8 = 13 döngü), arıza yok
        result = maintenance.simulate_policy(_toy_predictions(), "lower", 10)
        self.assertEqual(result["n_maintained"], 2)
        self.assertEqual(result["n_failures"], 0)
        self.assertAlmostEqual(result["total_cost"], 13.0)
        self.assertAlmostEqual(result["mean_wasted_cycles"], 6.5)


@unittest.skipUnless(
    _implemented(maintenance.best_threshold, _toy_predictions(), "pred", [10.0]), SKIP_REASON
)
class BestThresholdTest(unittest.TestCase):
    def test_finds_cheapest_threshold(self):
        # pred ile: eşik 10 -> 200, eşik 20 -> bakım 12 ve 20 (5+8=13), maliyet 13
        #           eşik 40 -> bakım 12, 35, 20 (5+40+8=53)
        thr, result = maintenance.best_threshold(_toy_predictions(), "pred", [10, 20, 40])
        self.assertEqual(thr, 20)
        self.assertAlmostEqual(result["total_cost"], 13.0)

    def test_tie_keeps_first_threshold(self):
        thr, _ = maintenance.best_threshold(_toy_predictions(), "pred", [20, 25])
        self.assertEqual(thr, 20)


if __name__ == "__main__":
    unittest.main()
