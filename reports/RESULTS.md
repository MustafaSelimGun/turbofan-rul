# Sonuçlar

Bu dosya `python scripts/make_report.py` ile `results.json`'dan üretilir. Elle düzenleme.

## 1. Model karşılaştırması (tek koşullu alt kümeler)

| Alt küme | Model | CV RMSE (motor bazlı, 5 kat) | Test RMSE | Test MAE | NASA skoru |
|---|---|---|---|---|---|
| FD001 | ridge | 19.2 ± 1.1 | 20.2 | 15.8 | 969 |
| FD001 | random_forest | 13.9 ± 1.0 | 14.7 | 10.8 | 355 |
| FD001 | gradient_boosting | 13.6 ± 0.8 | 14.4 | 10.4 | 415 |
| FD003 | ridge | 18.3 ± 0.9 | 20.0 | 15.4 | 1320 |
| FD003 | random_forest | 12.2 ± 1.1 | 13.8 | 10.0 | 354 |
| FD003 | gradient_boosting | 11.9 ± 1.2 | 13.6 | 9.7 | 345 |

Test metriklerinde gerçek RUL 125'te kırpılmıştır. Kırpılmamış RMSE değerleri `results.json` içinde (`test_rmse_raw`).

## 2. Çok koşullu alt kümelerde koşula göre normalizasyon

| Alt küme | Normalizasyon | CV RMSE | Test RMSE | NASA skoru |
|---|---|---|---|---|
| FD002 | global | 17.2 | 16.5 | 1332 |
| FD002 | koşula göre | 15.2 | 14.6 | 972 |
| FD004 | global | 16.7 | 17.5 | 1643 |
| FD004 | koşula göre | 14.6 | 15.2 | 1006 |

## 3. Sızıntı deneyi (FD001, Random Forest)

- Satırları rastgele bölünce doğrulama RMSE: **9.1**
- Motorları bölünce doğrulama RMSE: **13.6**
- Sızıntılı değerlendirme hatayı **1.5 kat** küçük gösteriyor.

## 4. %80 tahmin aralıkları (gradient boosting, kantil regresyon)

| Alt küme | Test RMSE | Kapsama (hedef %80) | Ortalama genişlik (döngü) |
|---|---|---|---|
| FD001 | 14.4 | %70 | 34.2 |
| FD002 | 14.6 | %80 | 32.4 |
| FD003 | 13.6 | %67 | 29.6 |
| FD004 | 15.2 | %75 | 32.2 |

## 5. Bakım kararı (FD001)

| Karar sütunu | En iyi eşik | Toplam maliyet | Bakım | Arıza | Ort. boşa giden döngü |
|---|---|---|---|---|---|
| pred | 15 | 116.0 | 11 | 0 | 10.5 |
| lower | 15 | 116.0 | 11 | 0 | 10.5 |

Toplam çalışma süresi: 742.4 sn
