"""Testler için küçük, elle kurulmuş veri üreticileri."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rul.config import SENSOR_COLUMNS, SETTING_COLUMNS


def make_fleet(n_units: int = 6, min_len: int = 30, max_len: int = 60, n_conditions: int = 1, seed: int = 0) -> pd.DataFrame:
    """Gerçek veriye benzeyen sentetik bir filo: sensörler arızaya yaklaştıkça kayar."""
    rng = np.random.default_rng(seed)
    rows = []
    condition_settings = [(0.0, 0.0, 100.0), (20.0, 0.7, 100.0), (35.0, 0.84, 60.0),
                          (42.0, 0.84, 100.0), (10.0, 0.25, 100.0), (25.0, 0.62, 60.0)]
    for unit in range(1, n_units + 1):
        length = int(rng.integers(min_len, max_len + 1))
        for cycle in range(1, length + 1):
            health = cycle / length
            cond = int(rng.integers(0, n_conditions))
            settings = condition_settings[cond]
            row = {"unit": unit, "cycle": cycle}
            row.update(dict(zip(SETTING_COLUMNS, settings)))
            for i, sensor in enumerate(SENSOR_COLUMNS):
                if i in (0, 4, 9, 15, 17, 18):  # gerçek veride sabit olan sensörler
                    row[sensor] = 1.0
                else:
                    row[sensor] = 100.0 + 10.0 * cond + (i + 1) * health + rng.normal(0, 0.05)
            rows.append(row)
    return pd.DataFrame(rows)
