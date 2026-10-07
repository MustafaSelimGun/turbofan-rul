import unittest

from rul.config import RUL_CAP
from rul.data import add_rul_labels, last_cycles
from _helpers import make_fleet


class AddRulLabelsTest(unittest.TestCase):
    def test_last_cycle_has_zero_rul(self):
        labelled = add_rul_labels(make_fleet(), cap=None)
        last = labelled.loc[labelled.groupby("unit")["cycle"].idxmax()]
        self.assertTrue((last["rul"] == 0).all())

    def test_rul_decreases_by_one_each_cycle(self):
        labelled = add_rul_labels(make_fleet(), cap=None)
        for _, unit in labelled.groupby("unit"):
            diffs = unit.sort_values("cycle")["rul"].diff().dropna()
            self.assertTrue((diffs == -1).all())

    def test_cap_is_applied(self):
        fleet = make_fleet(min_len=200, max_len=250)
        labelled = add_rul_labels(fleet, cap=RUL_CAP)
        self.assertEqual(labelled["rul"].max(), RUL_CAP)

    def test_input_is_not_modified(self):
        fleet = make_fleet()
        add_rul_labels(fleet)
        self.assertNotIn("rul", fleet.columns)


class LastCyclesTest(unittest.TestCase):
    def test_one_row_per_unit(self):
        fleet = make_fleet()
        last = last_cycles(fleet)
        self.assertEqual(len(last), fleet["unit"].nunique())
        self.assertTrue((last["cycle"].to_numpy() == fleet.groupby("unit")["cycle"].max().to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
