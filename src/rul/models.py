"""Karşılaştırılan modeller.

Basitten karmaşığa üç model: her biri bir öncekinin neyi eksik yaptığını
göstermek için var. Hiperparametreler bilinçli olarak sade tutuldu; test
setine bakarak ayar yapmak, test sonucunu iyimser hale getirir.
"""

from __future__ import annotations

from sklearn.base import RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rul.config import RANDOM_STATE

MODEL_NAMES = ("ridge", "random_forest", "gradient_boosting")


def make_model(name: str) -> RegressorMixin:
    if name == "ridge":
        # Doğrusal taban çizgisi: sensörler ile RUL arasında doğrusal ilişki varsayar.
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "random_forest":
        # Ağaçların ortalaması: doğrusal olmayan ilişkileri yakalar, varyansı düşürür.
        return RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=5,
            max_features=0.5,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if name == "gradient_boosting":
        # Ağaçları sırayla, bir öncekinin hatasını düzeltecek şekilde ekler.
        return HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Bilinmeyen model: {name}. Seçenekler: {MODEL_NAMES}")


def make_quantile_model(quantile: float) -> RegressorMixin:
    """Belirli bir kantili tahmin eden gradient boosting modeli.

    quantile=0.1 ile eğitilen model, gerçek değerin yaklaşık %10 ihtimalle
    altında kalacağı bir sınır öğrenir. 0.1 ve 0.9 birlikte %80'lik bir
    tahmin aralığı verir.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile 0 ile 1 arasında olmalı.")
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        max_iter=400,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )
