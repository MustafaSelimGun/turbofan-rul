import unittest

from sklearn.model_selection import GroupKFold

from rul.data import add_rul_labels
from rul.evaluation import unit_cross_validation
from _helpers import make_fleet


class UnitSplitTest(unittest.TestCase):
    def test_group_kfold_never_shares_units(self):
        labelled = add_rul_labels(make_fleet(n_units=10))
        for tr, va in GroupKFold(n_splits=5).split(labelled, groups=labelled["unit"]):
            self.assertTrue(set(labelled.iloc[tr]["unit"]).isdisjoint(labelled.iloc[va]["unit"]))

    def test_cross_validation_runs_and_returns_scores(self):
        result = unit_cross_validation(make_fleet(n_units=10), "ridge", n_conditions=1, n_splits=5)
        self.assertEqual(len(result["folds"]), 5)
        self.assertGreater(result["cv_rmse_mean"], 0)


if __name__ == "__main__":
    unittest.main()
