"""Bakım kararı simülasyonu.

=====================================================================
NOT: Bu modülün ilk sürümü zaman kısıtı yüzünden Claude tarafından yazıldı.
Proje sahibi (Mustafa Selim Gündüz) bunu kendi yazdığı sürümle değiştirecek.
Fonksiyon imzaları ve testler (tests/test_maintenance.py) aynı kalır, böylece
değişiklik güvenle yapılabilir.
=====================================================================

Senaryo
-------
Elimizde her test motoru için modelin tahmini var: bir nokta tahmini (pred)
ve %80'lik bir aralık (lower, upper). Her motor için bugün tek bir karar
veriyoruz:

    BAKIMA AL  -> motor bugün bakıma girer. Kalan ömrünün hepsi "boşa
                  harcanmış" olur (daha uçabilecekken söküldü).
    BEKLE      -> motor uçmaya devam eder. Bir sonraki kontrol
                  `inspection_interval` döngü sonra. Gerçek kalan ömür bu
                  aralıktan kısaysa motor kontrolden ÖNCE arızalanır.

Maliyetler:
    - Bakıma alınan motor:      true_rul * cost_per_wasted_cycle
    - Arızalanan motor:         cost_of_failure
    - Güvenle bekleyen motor:   0

Soru: Hangi eşik ve hangi tahmin (nokta tahmini mi, aralığın alt sınırı
mı?) toplam maliyeti en aza indirir?

İpuçları
--------
- Ağırlıklı olarak pandas ve basit if/else yeterli. numpy da kullanabilirsin.
- Önce `should_maintain`, sonra `unit_outcome`, sonra `simulate_policy`,
  en son `best_threshold` fonksiyonunu yaz. Her biri bir öncekini kullanır.
- Takıldığında testlere bak: beklenen davranışı küçük örneklerle gösteriyorlar.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CostModel:
    """Maliyet varsayımları. Birimler göreceli: 1 = bir döngülük ömür kaybı."""

    cost_per_wasted_cycle: float = 1.0
    cost_of_failure: float = 100.0
    inspection_interval: int = 10


def should_maintain(estimate: float, threshold: float) -> bool:
    """Tahmin edilen kalan ömür eşik değere eşit veya altındaysa True döndür.

    Örnek: should_maintain(8, 10) -> True, should_maintain(15, 10) -> False,
           should_maintain(10, 10) -> True
    """
    # Eşit olduğunda da bakıma alıyoruz (<=): sınırdaki motoru riske atmak pahalı.
    return estimate <= threshold


def unit_outcome(true_rul: float, maintain: bool, cost_model: CostModel) -> dict:
    """Tek bir motorun kararına göre sonucunu döndür.

    Dönen sözlük şu anahtarları içermeli:
        "cost":           float, bu motorun maliyeti
        "failed":         bool, motor kontrolden önce arızalandı mı
        "wasted_cycles":  float, bakıma alındıysa boşa giden döngü sayısı, değilse 0

    Kurallar:
        maintain=True  -> cost = true_rul * cost_per_wasted_cycle, failed=False,
                          wasted_cycles = true_rul
        maintain=False ve true_rul <= inspection_interval
                       -> cost = cost_of_failure, failed=True, wasted_cycles=0
        maintain=False ve true_rul >  inspection_interval
                       -> cost = 0, failed=False, wasted_cycles=0
    """
    if maintain:
        # Bakıma alınan motorun kalan ömrünün tamamı boşa gider.
        return {
            "cost": true_rul * cost_model.cost_per_wasted_cycle,
            "failed": False,
            "wasted_cycles": true_rul,
        }
    if true_rul <= cost_model.inspection_interval:
        # Bekleyen motor bir sonraki kontrolden önce arızalanır.
        return {"cost": cost_model.cost_of_failure, "failed": True, "wasted_cycles": 0}
    # Bekleyen motor güvenle bir sonraki kontrole ulaşır.
    return {"cost": 0.0, "failed": False, "wasted_cycles": 0}


def simulate_policy(
    predictions: pd.DataFrame,
    decision_column: str,
    threshold: float,
    cost_model: CostModel = CostModel(),
) -> dict:
    """Bir politikayı bütün motorlar üzerinde çalıştır ve özetle.

    predictions: en az "true_rul" ve `decision_column` sütunlarını içeren tablo
                 (örneğin decision_column="pred" ya da "lower").
    Her satır için should_maintain(satır[decision_column], threshold) ile karar ver,
    unit_outcome ile sonucu hesapla.

    Dönen sözlük:
        "total_cost":          bütün motorların maliyet toplamı
        "n_maintained":        bakıma alınan motor sayısı
        "n_failures":          arızalanan motor sayısı
        "mean_wasted_cycles":  bakıma alınan motorlarda ortalama boşa giden döngü
                               (hiç bakım yoksa 0.0)
    """
    total_cost = 0.0
    n_maintained = 0
    n_failures = 0
    wasted = []
    for _, row in predictions.iterrows():
        maintain = should_maintain(row[decision_column], threshold)
        outcome = unit_outcome(row["true_rul"], maintain, cost_model)
        total_cost += outcome["cost"]
        n_failures += int(outcome["failed"])
        if maintain:
            n_maintained += 1
            wasted.append(outcome["wasted_cycles"])
    return {
        "total_cost": total_cost,
        "n_maintained": n_maintained,
        "n_failures": n_failures,
        "mean_wasted_cycles": sum(wasted) / len(wasted) if wasted else 0.0,
    }


def best_threshold(
    predictions: pd.DataFrame,
    decision_column: str,
    thresholds: list[float],
    cost_model: CostModel = CostModel(),
) -> tuple[float, dict]:
    """Verilen eşikler arasından toplam maliyeti en düşük olanı bul.

    (en_iyi_eşik, o eşikteki simulate_policy sonucu) döndür.
    Eşitlik durumunda listede ÖNCE gelen eşiği seç.
    """
    best_thr = None
    best_result = None
    for thr in thresholds:
        result = simulate_policy(predictions, decision_column, thr, cost_model)
        # Kesin küçüktür kullandığımız için eşitlikte listede önce gelen kalır.
        if best_result is None or result["total_cost"] < best_result["total_cost"]:
            best_thr, best_result = thr, result
    return best_thr, best_result


def is_implemented() -> bool:
    """CLI, modül yazılmadan bakım raporunu atlamak için bunu kullanır."""
    try:
        should_maintain(0.0, 1.0)
    except NotImplementedError:
        return False
    return True
