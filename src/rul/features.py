"""Ham sensör verisinden model girdilerini üreten adımlar.

Bütün "öğrenilen" adımlar (çalışma koşulu kümeleri, normalizasyon
istatistikleri, hangi sensörlerin tutulacağı) FeatureBuilder.fit() içinde
YALNIZCA eğitim verisinden öğrenilir. transform() aynı değerleri doğrulama
ve test verisine uygular. Bu ayrım veri sızıntısını önlemenin temelidir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from rul.config import (
    MIN_SENSOR_STD,
    RANDOM_STATE,
    ROLLING_WINDOW,
    SENSOR_COLUMNS,
    SETTING_COLUMNS,
    SLOPE_WINDOW,
)


class FeatureBuilder:
    """Çalışma koşulu normalizasyonu + sensör seçimi + geçmiş pencere özellikleri.

    Parametreler
    ------------
    n_conditions:
        Çalışma koşulu sayısı. 1 ise koşul ayrımı yapılmaz (FD001/FD003).
        6 ise üç ayar sütunu k-means ile 6 kümeye ayrılır (FD002/FD004).
    normalize_by_condition:
        False yapılırsa koşula göre normalizasyon kapatılır. Deneylerde bu
        adımın katkısını ölçmek için kullanılır.
    """

    def __init__(
        self,
        n_conditions: int = 1,
        normalize_by_condition: bool = True,
        rolling_window: int = ROLLING_WINDOW,
        slope_window: int = SLOPE_WINDOW,
    ) -> None:
        self.n_conditions = n_conditions
        self.normalize_by_condition = normalize_by_condition
        self.rolling_window = rolling_window
        self.slope_window = slope_window

    # ------------------------------------------------------------------
    # Öğrenme (sadece eğitim verisi)
    # ------------------------------------------------------------------
    def fit(self, train: pd.DataFrame) -> "FeatureBuilder":
        self._settings_scaler = StandardScaler().fit(train[SETTING_COLUMNS])
        if self.n_conditions > 1:
            self._kmeans = KMeans(
                n_clusters=self.n_conditions, n_init=10, random_state=RANDOM_STATE
            ).fit(self._settings_scaler.transform(train[SETTING_COLUMNS]))
        else:
            self._kmeans = None

        conditions = self._assign_conditions(train)

        # Sensör seçimi: her koşul içinde neredeyse sabit kalan sensörler
        # bilgi taşımaz (örneğin FD001'de s1, s5, s10, s16, s18, s19).
        within_std = train[SENSOR_COLUMNS].groupby(conditions).std().mean()
        self.sensors_ = [s for s in SENSOR_COLUMNS if within_std[s] >= MIN_SENSOR_STD]
        if not self.sensors_:
            raise ValueError("Hiçbir sensör seçilemedi; veri beklenenden farklı.")

        if self.normalize_by_condition:
            grouped = train[self.sensors_].groupby(conditions)
            self._means = grouped.mean()
            # std=0 olan (koşul içinde sabit) sensörlerde sıfıra bölmeyi önle.
            self._stds = grouped.std().replace(0, 1.0).fillna(1.0)
        else:
            self._means = train[self.sensors_].mean().to_frame().T
            self._stds = train[self.sensors_].std().replace(0, 1.0).to_frame().T
        return self

    # ------------------------------------------------------------------
    # Uygulama (eğitim, doğrulama, test)
    # ------------------------------------------------------------------
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Model için özellik tablosu döndürür. Satır sırası korunur."""
        self._check_fitted()
        frame = frame.sort_values(["unit", "cycle"])
        normalized = self._normalize(frame)

        features = pd.DataFrame(index=frame.index)
        features["cycle"] = frame["cycle"]
        units = frame["unit"]
        w, sw = self.rolling_window, self.slope_window

        # Bütün sensörler için tek seferde, motor bazında kayan pencereler.
        # groupby(...).rolling(...) her motoru ayrı ele alır: bir motorun
        # penceresi asla başka bir motorun satırlarına taşmaz.
        roll_mean = _grouped_rolling(normalized, units, w, min_periods=1, how="mean")
        roll_std = _grouped_rolling(normalized, units, w, min_periods=2, how="std").fillna(0.0)
        slopes = _grouped_slope(normalized, units, sw)

        for sensor in self.sensors_:
            features[sensor] = normalized[sensor]
            features[f"{sensor}_mean{w}"] = roll_mean[sensor]
            features[f"{sensor}_std{w}"] = roll_std[sensor]
            features[f"{sensor}_slope{sw}"] = slopes[sensor]
        return features

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train).transform(train)

    # ------------------------------------------------------------------
    # Yardımcılar
    # ------------------------------------------------------------------
    def _assign_conditions(self, frame: pd.DataFrame) -> pd.Series:
        if self._kmeans is None:
            return pd.Series(0, index=frame.index, name="condition")
        labels = self._kmeans.predict(self._settings_scaler.transform(frame[SETTING_COLUMNS]))
        return pd.Series(labels, index=frame.index, name="condition")

    def _normalize(self, frame: pd.DataFrame) -> pd.DataFrame:
        values = frame[self.sensors_]
        if self.normalize_by_condition:
            conditions = self._assign_conditions(frame)
            means = self._means.reindex(conditions).to_numpy()
            stds = self._stds.reindex(conditions).to_numpy()
        else:
            means = self._means.to_numpy()
            stds = self._stds.to_numpy()
        return pd.DataFrame((values.to_numpy() - means) / stds, index=frame.index, columns=self.sensors_)

    def _check_fitted(self) -> None:
        if not hasattr(self, "sensors_"):
            raise RuntimeError("Önce fit() çağrılmalı.")


def _grouped_rolling(values: pd.DataFrame, units: pd.Series, window: int, min_periods: int, how: str) -> pd.DataFrame:
    """Her motor için ayrı kayan pencere istatistiği; satır sırası korunur."""
    rolled = getattr(values.groupby(units, sort=False).rolling(window, min_periods=min_periods), how)()
    return rolled.reset_index(level=0, drop=True).loc[values.index]


def _grouped_slope(values: pd.DataFrame, units: pd.Series, window: int) -> pd.DataFrame:
    """Her motor ve sensör için son `window` döngüdeki en küçük kareler eğimi.

    eğim = kov(t, x) / var(t) = (E[t·x] - E[t]·E[x]) / (E[t²] - E[t]²)
    Beklenen değerler kayan pencerede alınır, yani sadece geçmiş kullanılır.
    """
    t = units.groupby(units, sort=False).cumcount().astype(float)
    t_frame = pd.DataFrame({c: t for c in values.columns}, index=values.index)

    def roll(frame: pd.DataFrame) -> pd.DataFrame:
        return _grouped_rolling(frame, units, window, min_periods=2, how="mean")

    mean_t, mean_x = roll(t_frame), roll(values)
    cov = roll(t_frame * values) - mean_t * mean_x
    var = roll(t_frame * t_frame) - mean_t * mean_t
    return (cov / var.where(var != 0)).fillna(0.0)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Son `window` döngüdeki doğrusal eğim (en küçük kareler).

    slope = kov(t, x) / var(t). Ortalamalar kayan pencerede hesaplanır, böylece
    her satır sadece kendisi ve geçmişi kullanılarak hesaplanmış olur.
    """
    t = pd.Series(np.arange(len(series), dtype=float), index=series.index)
    roll = lambda s: s.rolling(window, min_periods=2).mean()  # noqa: E731
    mean_t, mean_x = roll(t), roll(series)
    cov = roll(t * series) - mean_t * mean_x
    var = roll(t * t) - mean_t * mean_t
    slope = cov / var.replace(0, np.nan)
    return slope.fillna(0.0)
