"""Proje genelinde kullanılan sabitler.

Sabitleri tek bir yerde tutmak, aynı sayının (örneğin RUL üst sınırı) farklı
dosyalarda farklı yazılmasını önler.
"""

from __future__ import annotations

from pathlib import Path

# C-MAPSS metin dosyalarında 26 sütun var: motor numarası, döngü,
# 3 çalışma ayarı ve 21 sensör.
ID_COLUMNS = ["unit", "cycle"]
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS = [f"s{i}" for i in range(1, 22)]
RAW_COLUMNS = ID_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS

SUBSETS = ("FD001", "FD002", "FD003", "FD004")

# FD002 ve FD004 altı farklı uçuş koşulunda (irtifa, Mach, gaz kolu) kaydedildi.
# FD001 ve FD003 tek koşullu (deniz seviyesi).
N_OPERATING_CONDITIONS = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}

# Parçalı doğrusal RUL hedefi: motor sağlıklıyken kalan ömrü
# "125 döngüden fazla" olarak kabul edip 125'te sabitliyoruz.
# Literatürde 120-130 arası değerler yaygın (Heimes 2008).
RUL_CAP = 125

# Özellik pencereleri (döngü sayısı). Sadece GEÇMİŞ döngüler kullanılır.
ROLLING_WINDOW = 10
SLOPE_WINDOW = 20

# Eğitim verisinde standart sapması bu değerin altında kalan sensörler
# bilgi taşımadığı için atılır.
MIN_SENSOR_STD = 1e-4

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "CMAPSSData"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
