import math
import unittest

from rul.metrics import interval_coverage, mae, mean_interval_width, nasa_score, rmse


class MetricsTest(unittest.TestCase):
    def test_rmse_and_mae(self):
        self.assertAlmostEqual(rmse([0, 0], [3, 4]), math.sqrt(12.5))
        self.assertAlmostEqual(mae([0, 0], [3, -4]), 3.5)

    def test_nasa_score_is_zero_for_perfect_predictions(self):
        self.assertEqual(nasa_score([10, 20], [10, 20]), 0.0)

    def test_nasa_score_penalizes_late_more_than_early(self):
        late = nasa_score([50], [60])   # 10 döngü geç
        early = nasa_score([50], [40])  # 10 döngü erken
        self.assertAlmostEqual(late, math.e - 1)
        self.assertAlmostEqual(early, math.exp(10 / 13) - 1)
        self.assertGreater(late, early)

    def test_interval_metrics(self):
        self.assertEqual(interval_coverage([5, 15, 25], [0, 0, 0], [10, 20, 20]), 2 / 3)
        self.assertEqual(mean_interval_width([0, 10], [10, 30]), 15.0)


if __name__ == "__main__":
    unittest.main()
