"""C-MAPSS dosyalarını okuma ve RUL etiketlerini üretme."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rul.config import DEFAULT_DATA_DIR, RAW_COLUMNS, RUL_CAP, SUBSETS


@dataclass(frozen=True)
class SubsetData:
    """Bir alt kümenin (örneğin FD001) üç parçası."""

    name: str
    train: pd.DataFrame  # arızaya kadar giden tam seriler
    test: pd.DataFrame   # arızadan önce kesilmiş seriler
    test_rul: pd.Series  # test motorlarının son döngüden sonraki gerçek kalan ömrü (index = unit)


def read_cmapss_file(path: str | Path) -> pd.DataFrame:
    """Boşlukla ayrılmış bir C-MAPSS dosyasını sütun adlarıyla okur."""
    frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if frame.shape[1] != len(RAW_COLUMNS):
        raise ValueError(
            f"{path}: {len(RAW_COLUMNS)} sütun bekleniyordu, {frame.shape[1]} bulundu."
        )
    frame.columns = RAW_COLUMNS
    frame[["unit", "cycle"]] = frame[["unit", "cycle"]].astype(int)
    return frame.sort_values(["unit", "cycle"]).reset_index(drop=True)


def load_subset(name: str, data_dir: str | Path = DEFAULT_DATA_DIR) -> SubsetData:
    """train_<ad>.txt, test_<ad>.txt ve RUL_<ad>.txt dosyalarını yükler."""
    if name not in SUBSETS:
        raise ValueError(f"Bilinmeyen alt küme: {name}. Seçenekler: {SUBSETS}")
    data_dir = Path(data_dir)
    train = read_cmapss_file(data_dir / f"train_{name}.txt")
    test = read_cmapss_file(data_dir / f"test_{name}.txt")
    rul_values = pd.read_csv(data_dir / f"RUL_{name}.txt", header=None).iloc[:, 0].astype(int)
    test_units = np.sort(test["unit"].unique())
    if len(test_units) != len(rul_values):
        raise ValueError("Test motoru sayısı ile RUL dosyasındaki satır sayısı uyuşmuyor.")
    # RUL dosyasının i. satırı, test dosyasındaki i. motora aittir.
    test_rul = pd.Series(rul_values.to_numpy(), index=test_units, name="true_rul")
    return SubsetData(name=name, train=train, test=test, test_rul=test_rul)


def add_rul_labels(train: pd.DataFrame, cap: int | None = RUL_CAP) -> pd.DataFrame:
    """Eğitim verisine her satır için kalan ömrü (RUL) ekler.

    Eğitim serileri arızaya kadar sürdüğü için bir satırın kalan ömrü:
        RUL = o motorun son döngüsü - satırın döngüsü
    Son satırda RUL = 0 olur.

    cap verilirse RUL bu değerde kırpılır (parçalı doğrusal hedef). Sebep:
    motorun ilk döngülerinde sensörlerde bozulma izi yoktur; model "kalan ömür
    300 mü 250 mi" ayrımını veriden öğrenemez. Hedefi sabitlemek modelin
    öğrenebileceği bir şey istemek demektir.
    """
    labelled = train.copy()
    last_cycle = labelled.groupby("unit")["cycle"].transform("max")
    labelled["rul"] = last_cycle - labelled["cycle"]
    if cap is not None:
        labelled["rul"] = labelled["rul"].clip(upper=cap)
    return labelled


def last_cycles(frame: pd.DataFrame) -> pd.DataFrame:
    """Her motorun son gözlenen döngüsündeki satırı döndürür (test tahmin noktası)."""
    idx = frame.groupby("unit")["cycle"].idxmax()
    return frame.loc[idx].sort_values("unit").reset_index(drop=True)
