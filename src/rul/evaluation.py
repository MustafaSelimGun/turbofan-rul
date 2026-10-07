"""Değerlendirme düzenekleri: motor bazlı çapraz doğrulama, test değerlendirmesi
ve sızıntı deneyi."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split

from rul.config import RANDOM_STATE, RUL_CAP
from rul.data import SubsetData, add_rul_labels
from rul.features import FeatureBuilder
from rul.metrics import interval_coverage, mae, mean_interval_width, nasa_score, rmse
from rul.models import make_model, make_quantile_model


def _builder(n_conditions: int, normalize_by_condition: bool) -> FeatureBuilder:
    return FeatureBuilder(n_conditions=n_conditions, normalize_by_condition=normalize_by_condition)


def unit_cross_validation(
    train: pd.DataFrame,
    model_name: str,
    n_conditions: int,
    n_splits: int = 5,
    normalize_by_condition: bool = True,
) -> dict:
    """Motor bazlı (GroupKFold) çapraz doğrulama.

    Her katmanda doğrulama motorları eğitimde HİÇ görülmez. Özellik adımları
    (koşul kümeleri, normalizasyon, sensör seçimi) her katmanda yalnızca o
    katmanın eğitim motorlarından öğrenilir.

    Değerlendirme, doğrulama motorlarının bütün döngüleri üzerinde yapılır
    (hedef 125'te kırpılmış RUL).
    """
    labelled = add_rul_labels(train)
    splitter = GroupKFold(n_splits=n_splits)
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(labelled, groups=labelled["unit"])):
        tr, va = labelled.iloc[tr_idx], labelled.iloc[va_idx]
        assert set(tr["unit"]).isdisjoint(set(va["unit"])), "Motor sızıntısı!"
        builder = _builder(n_conditions, normalize_by_condition).fit(tr)
        x_tr, x_va = builder.transform(tr), builder.transform(va)
        y_tr, y_va = tr.loc[x_tr.index, "rul"], va.loc[x_va.index, "rul"]
        model = make_model(model_name).fit(x_tr, y_tr)
        pred = np.clip(model.predict(x_va), 0, None)
        fold_scores.append({"fold": fold, "rmse": rmse(y_va, pred), "mae": mae(y_va, pred)})
    scores = pd.DataFrame(fold_scores)
    return {
        "model": model_name,
        "cv_rmse_mean": float(scores["rmse"].mean()),
        "cv_rmse_std": float(scores["rmse"].std()),
        "cv_mae_mean": float(scores["mae"].mean()),
        "folds": fold_scores,
    }


@dataclass
class TestPredictions:
    units: np.ndarray
    true_rul: np.ndarray
    pred: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame({"unit": self.units, "true_rul": self.true_rul, "pred": self.pred})
        if self.lower is not None:
            frame["lower"] = self.lower
            frame["upper"] = self.upper
        return frame


def fit_and_predict_test(
    subset: SubsetData,
    model_name: str,
    n_conditions: int,
    normalize_by_condition: bool = True,
    with_intervals: bool = False,
    interval: tuple[float, float] = (0.1, 0.9),
) -> TestPredictions:
    """Bütün eğitim verisiyle eğitir, her test motorunun SON döngüsünde tahmin yapar."""
    labelled = add_rul_labels(subset.train)
    builder = _builder(n_conditions, normalize_by_condition).fit(labelled)
    x_train = builder.transform(labelled)
    y_train = labelled.loc[x_train.index, "rul"]

    # Test özellikleri bütün test serisi üzerinden hesaplanır (pencereler
    # geçmişi kullanır), sonra her motorun son satırı seçilir.
    x_test_all = builder.transform(subset.test)
    test_sorted = subset.test.loc[x_test_all.index]
    last_idx = test_sorted.groupby("unit")["cycle"].idxmax().sort_index()
    x_test = x_test_all.loc[last_idx.to_numpy()]
    units = last_idx.index.to_numpy()
    true_rul = subset.test_rul.loc[units].to_numpy()

    model = make_model(model_name).fit(x_train, y_train)
    pred = np.clip(model.predict(x_test), 0, None)
    result = TestPredictions(units=units, true_rul=true_rul, pred=pred)

    if with_intervals:
        low_q, high_q = interval
        lower = make_quantile_model(low_q).fit(x_train, y_train).predict(x_test)
        upper = make_quantile_model(high_q).fit(x_train, y_train).predict(x_test)
        lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)
        result.lower, result.upper = np.clip(lower, 0, None), np.clip(upper, 0, None)
    return result


def score_test(preds: TestPredictions, cap: int | None = RUL_CAP) -> dict:
    """Test metrikleri. cap verilirse gerçek RUL de aynı sınırla kırpılır.

    Literatürde iki yaygın raporlama var; ikisini de veriyoruz:
      *_capped: gerçek RUL 125'te kırpılmış (modelin öğrendiği hedefle tutarlı)
      *_raw:    gerçek RUL olduğu gibi
    """
    true_raw = preds.true_rul.astype(float)
    true_cap = np.minimum(true_raw, cap) if cap is not None else true_raw
    out = {
        "rmse_capped": rmse(true_cap, preds.pred),
        "rmse_raw": rmse(true_raw, preds.pred),
        "mae_capped": mae(true_cap, preds.pred),
        "nasa_score_capped": nasa_score(true_cap, preds.pred),
        "n_units": int(len(true_raw)),
    }
    if preds.lower is not None:
        out["interval_coverage_capped"] = interval_coverage(true_cap, preds.lower, preds.upper)
        out["interval_mean_width"] = mean_interval_width(preds.lower, preds.upper)
    return out


def leakage_experiment(train: pd.DataFrame, model_name: str = "random_forest", n_conditions: int = 1) -> dict:
    """Satırları rastgele bölmek ile motor bazlı bölmek arasındaki farkı ölçer.

    Rastgele bölmede aynı motorun komşu döngüleri hem eğitimde hem doğrulamada
    olur. Komşu döngüler neredeyse aynı göründüğü için model "ezberleyerek"
    çok iyi skor alır; bu skor yeni bir motorda tekrarlanmaz.
    """
    labelled = add_rul_labels(train)

    # 1) Sızıntılı: satırları rastgele %80/%20 böl.
    tr, va = train_test_split(labelled, test_size=0.2, random_state=RANDOM_STATE)
    builder = _builder(n_conditions, True).fit(labelled)  # sızıntılı senaryoda bilinçli olarak tüm veriyle
    x_all = builder.transform(labelled)
    model = make_model(model_name).fit(x_all.loc[tr.index], labelled.loc[tr.index, "rul"])
    leaky_pred = np.clip(model.predict(x_all.loc[va.index]), 0, None)
    leaky_rmse = rmse(labelled.loc[va.index, "rul"], leaky_pred)

    # 2) Doğru: motor bazlı %80/%20 böl.
    units = labelled["unit"].unique()
    rng = np.random.default_rng(RANDOM_STATE)
    va_units = set(rng.choice(units, size=int(round(0.2 * len(units))), replace=False))
    tr2 = labelled[~labelled["unit"].isin(va_units)]
    va2 = labelled[labelled["unit"].isin(va_units)]
    builder2 = _builder(n_conditions, True).fit(tr2)
    x_tr2, x_va2 = builder2.transform(tr2), builder2.transform(va2)
    model2 = make_model(model_name).fit(x_tr2, tr2.loc[x_tr2.index, "rul"])
    unit_pred = np.clip(model2.predict(x_va2), 0, None)
    unit_rmse = rmse(va2.loc[x_va2.index, "rul"], unit_pred)

    return {
        "model": model_name,
        "random_row_split_rmse": leaky_rmse,
        "unit_split_rmse": unit_rmse,
        "optimism_ratio": unit_rmse / leaky_rmse if leaky_rmse > 0 else float("nan"),
    }
