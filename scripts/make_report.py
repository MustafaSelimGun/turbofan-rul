"""reports/results.json dosyasından okunabilir bir reports/RESULTS.md üretir.

Kullanım:  python scripts/make_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def fmt(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    r = json.loads((REPORTS / "results.json").read_text(encoding="utf-8"))
    lines = ["# Sonuçlar", "", "Bu dosya `python scripts/make_report.py` ile `results.json`'dan üretilir. Elle düzenleme.", ""]

    lines += ["## 1. Model karşılaştırması (tek koşullu alt kümeler)", "",
              "| Alt küme | Model | CV RMSE (motor bazlı, 5 kat) | Test RMSE | Test MAE | NASA skoru |",
              "|---|---|---|---|---|---|"]
    for row in r["model_comparison"]:
        lines.append(
            f"| {row['subset']} | {row['model']} | {fmt(row['cv_rmse_mean'])} ± {fmt(row['cv_rmse_std'])} | "
            f"{fmt(row['test_rmse_capped'])} | {fmt(row['test_mae_capped'])} | {fmt(row['test_nasa_score_capped'], 0)} |"
        )
    lines += ["", "Test metriklerinde gerçek RUL 125'te kırpılmıştır. Kırpılmamış RMSE değerleri `results.json` içinde (`test_rmse_raw`).", ""]

    lines += ["## 2. Çok koşullu alt kümelerde koşula göre normalizasyon", "",
              "| Alt küme | Normalizasyon | CV RMSE | Test RMSE | NASA skoru |", "|---|---|---|---|---|"]
    for row in r["condition_normalization"]:
        lines.append(
            f"| {row['subset']} | {'koşula göre' if row['condition_normalization'] else 'global'} | "
            f"{fmt(row['cv_rmse_mean'])} | {fmt(row['test_rmse_capped'])} | {fmt(row['test_nasa_score_capped'], 0)} |"
        )
    lines.append("")

    leak = r["leakage"]
    lines += ["## 3. Sızıntı deneyi (FD001, Random Forest)", "",
              f"- Satırları rastgele bölünce doğrulama RMSE: **{fmt(leak['random_row_split_rmse'])}**",
              f"- Motorları bölünce doğrulama RMSE: **{fmt(leak['unit_split_rmse'])}**",
              f"- Sızıntılı değerlendirme hatayı **{fmt(leak['optimism_ratio'])} kat** küçük gösteriyor.", ""]

    lines += ["## 4. %80 tahmin aralıkları (gradient boosting, kantil regresyon)", "",
              "| Alt küme | Test RMSE | Kapsama (hedef %80) | Ortalama genişlik (döngü) |", "|---|---|---|---|"]
    for row in r["intervals"]:
        lines.append(
            f"| {row['subset']} | {fmt(row['rmse_capped'])} | %{fmt(100 * row['interval_coverage_capped'], 0)} | "
            f"{fmt(row['interval_mean_width'])} |"
        )
    lines.append("")

    if "maintenance" in r:
        lines += ["## 5. Bakım kararı (FD001)", "", "| Karar sütunu | En iyi eşik | Toplam maliyet | Bakım | Arıza | Ort. boşa giden döngü |",
                  "|---|---|---|---|---|---|"]
        for column, row in r["maintenance"].items():
            lines.append(
                f"| {column} | {row['best_threshold']} | {fmt(row['total_cost'])} | {row['n_maintained']} | "
                f"{row['n_failures']} | {fmt(row['mean_wasted_cycles'])} |"
            )
        lines.append("")
    else:
        lines += ["## 5. Bakım kararı", "", "Henüz hesaplanmadı: `src/rul/maintenance.py` yazıldıktan sonra `rul run-all` tekrar çalıştırılmalı.", ""]

    lines += [f"Toplam çalışma süresi: {r['runtime_seconds']} sn", ""]
    (REPORTS / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(REPORTS / "RESULTS.md")


if __name__ == "__main__":
    main()
