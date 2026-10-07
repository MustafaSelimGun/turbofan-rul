"""Değerlendirme metrikleri."""

from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_pred - y_true)))


def nasa_score(y_true, y_pred) -> float:
    """PHM08 yarışmasının asimetrik skor fonksiyonu (düşük = iyi).

    d = tahmin - gerçek
      d < 0 (erken tahmin):  exp(-d/13) - 1
      d >= 0 (geç tahmin):   exp(d/10) - 1

    Geç tahmin daha ağır cezalandırılır: motoru "daha çok ömrü var" sanıp
    uçurmak, erken bakıma almaktan çok daha tehlikelidir.
    Kaynak: Saxena ve ark. 2008.
    """
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    penalties = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(penalties))


def interval_coverage(y_true, lower, upper) -> float:
    """Gerçek değerin [lower, upper] aralığına düştüğü örneklerin oranı."""
    y_true = np.asarray(y_true, float)
    return float(np.mean((y_true >= np.asarray(lower)) & (y_true <= np.asarray(upper))))


def mean_interval_width(lower, upper) -> float:
    return float(np.mean(np.asarray(upper, float) - np.asarray(lower, float)))
