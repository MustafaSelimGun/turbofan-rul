import unittest

import numpy as np

from rul.features import FeatureBuilder, _rolling_slope
from _helpers import make_fleet

import pandas as pd


class FeatureBuilderTest(unittest.TestCase):
    def test_constant_sensors_are_dropped(self):
        builder = FeatureBuilder().fit(make_fleet())
        for constant in ("s1", "s5", "s10", "s16", "s18", "s19"):
            self.assertNotIn(constant, builder.sensors_)

    def test_features_use_only_past(self):
        """Gelecekteki döngüleri silmek, geçmiş satırların özelliklerini DEĞİŞTİRMEMELİ."""
        fleet = make_fleet()
        builder = FeatureBuilder().fit(fleet)
        full = builder.transform(fleet)
        truncated_input = fleet[fleet["cycle"] <= 20]
        truncated = builder.transform(truncated_input)
        pd.testing.assert_frame_equal(full.loc[truncated.index], truncated)

    def test_transform_does_not_refit(self):
        """Test verisinin istatistikleri normalizasyonu etkilememeli."""
        train, other = make_fleet(seed=1), make_fleet(seed=2) * 1.0
        other[[c for c in other.columns if c.startswith("s")]] += 50.0
        builder = FeatureBuilder().fit(train)
        means_before = builder._means.copy()
        builder.transform(other)
        pd.testing.assert_frame_equal(builder._means, means_before)

    def test_condition_normalization_removes_condition_offset(self):
        fleet = make_fleet(n_units=8, n_conditions=6, seed=3)
        builder = FeatureBuilder(n_conditions=6).fit(fleet)
        normalized = builder._normalize(fleet)
        conditions = builder._assign_conditions(fleet)
        per_condition_mean = normalized.groupby(conditions).mean().abs().max().max()
        self.assertLess(per_condition_mean, 1e-6)

    def test_unfitted_builder_raises(self):
        with self.assertRaises(RuntimeError):
            FeatureBuilder().transform(make_fleet())


class RollingSlopeTest(unittest.TestCase):
    def test_slope_of_line(self):
        series = pd.Series(3.0 * np.arange(30) + 5.0)
        slope = _rolling_slope(series, window=10)
        self.assertAlmostEqual(slope.iloc[-1], 3.0, places=6)
        self.assertEqual(slope.iloc[0], 0.0)


if __name__ == "__main__":
    unittest.main()
