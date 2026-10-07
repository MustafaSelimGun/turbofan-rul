"""Rapor grafikleri."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rul.data import add_rul_labels  # noqa: E402
from rul.features import FeatureBuilder  # noqa: E402

INK, MUTED, ACCENT, WARN = "#16202d", "#7a8494", "#0f6f66", "#b4351f"


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def sensor_trends(train: pd.DataFrame, sensors: list[str], path: Path, n_units: int = 20) -> None:
    """Motorların sensör değerleri, arızaya kalan döngüye göre hizalanmış."""
    labelled = add_rul_labels(train, cap=None)
    units = labelled["unit"].unique()[:n_units]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for ax, sensor in zip(axes.ravel(), sensors):
        for unit in units:
            u = labelled[labelled["unit"] == unit]
            ax.plot(-u["rul"], u[sensor], color=ACCENT, alpha=0.25, lw=0.8)
        _style(ax, sensor, "arızaya kalan döngü (0 = arıza)", "değer")
    fig.suptitle("FD001: sensörler arızaya yaklaştıkça sistematik olarak kayıyor", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def condition_effect(train: pd.DataFrame, sensor: str, path: Path, unit: int = 1) -> None:
    """Çok koşullu veride ham sensör ile koşula göre normalize edilmiş sensör."""
    builder = FeatureBuilder(n_conditions=6).fit(train)
    u = train[train["unit"] == unit].sort_values("cycle")
    normalized = builder._normalize(u)[sensor]  # noqa: SLF001 (rapor için iç fonksiyon)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].plot(u["cycle"], u[sensor], ".", ms=3, color=MUTED)
    _style(axes[0], f"Ham {sensor}: 6 uçuş koşulu üst üste", "döngü", "değer")
    axes[1].plot(u["cycle"], normalized, ".", ms=3, color=ACCENT)
    axes[1].plot(u["cycle"], normalized.rolling(15, min_periods=1).mean(), color=INK, lw=1.5)
    _style(axes[1], f"Koşula göre normalize {sensor}: bozulma görünür", "döngü", "z-skoru")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pred_vs_true(frame: pd.DataFrame, title: str, path: Path, cap: int = 125) -> None:
    true_cap = np.minimum(frame["true_rul"], cap)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(true_cap, frame["pred"], s=14, color=ACCENT, alpha=0.8)
    lim = [0, max(true_cap.max(), frame["pred"].max()) + 5]
    ax.plot(lim, lim, color=MUTED, ls="--", lw=1)
    _style(ax, title, "gerçek RUL (125'te kırpılmış)", "tahmin")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def intervals(frame: pd.DataFrame, title: str, path: Path, cap: int = 125) -> None:
    f = frame.assign(true_cap=np.minimum(frame["true_rul"], cap)).sort_values("true_cap").reset_index(drop=True)
    inside = (f["true_cap"] >= f["lower"]) & (f["true_cap"] <= f["upper"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.vlines(f.index, f["lower"], f["upper"], color=ACCENT, alpha=0.35, lw=3, label="%80 aralık")
    ax.scatter(f.index, f["pred"], s=8, color=ACCENT, label="tahmin")
    ax.scatter(f.index[inside], f["true_cap"][inside], s=10, color=INK, marker="x", label="gerçek (aralık içinde)")
    ax.scatter(f.index[~inside], f["true_cap"][~inside], s=14, color=WARN, marker="x", label="gerçek (aralık dışında)")
    _style(ax, title, "test motorları (gerçek RUL'a göre sıralı)", "RUL (döngü)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def leakage_bar(result: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    labels = ["Satırları rastgele böl\n(sızıntılı)", "Motorları böl\n(doğru)"]
    values = [result["random_row_split_rmse"], result["unit_split_rmse"]]
    bars = ax.bar(labels, values, color=[WARN, ACCENT], width=0.55)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.3, f"{value:.1f}", ha="center", color=INK)
    _style(ax, "Aynı model, iki değerlendirme: doğrulama RMSE", "", "RMSE (döngü)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def model_comparison(rows: list[dict], path: Path) -> None:
    frame = pd.DataFrame(rows)
    subsets = list(frame["subset"].unique())
    models = list(frame["model"].unique())
    width = 0.8 / len(models)
    colors = [MUTED, "#5aa69d", ACCENT]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, model in enumerate(models):
        part = frame[frame["model"] == model].set_index("subset").reindex(subsets)
        x = np.arange(len(subsets)) + i * width
        ax.bar(x, part["test_rmse_capped"], width=width, color=colors[i % len(colors)], label=model)
    ax.set_xticks(np.arange(len(subsets)) + width * (len(models) - 1) / 2, subsets)
    _style(ax, "Test RMSE (düşük = iyi)", "", "RMSE (döngü)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
