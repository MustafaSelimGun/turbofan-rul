"""Komut satırı arayüzü.

Örnekler:
    python -m rul.cli evaluate --subset FD001 --model gradient_boosting
    python -m rul.cli run-all            # bütün deneyler + rapor + grafikler
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rul import plots
from rul.config import DEFAULT_DATA_DIR, DEFAULT_REPORT_DIR, N_OPERATING_CONDITIONS, SUBSETS
from rul.data import load_subset
from rul.evaluation import fit_and_predict_test, leakage_experiment, score_test, unit_cross_validation
from rul.models import MODEL_NAMES
from rul import maintenance


def _evaluate(subset_name: str, model: str, data_dir: Path, normalize: bool = True, cv: bool = True) -> dict:
    subset = load_subset(subset_name, data_dir)
    n_cond = N_OPERATING_CONDITIONS[subset_name]
    row = {"subset": subset_name, "model": model, "condition_normalization": normalize}
    if cv:
        cv_result = unit_cross_validation(subset.train, model, n_cond, normalize_by_condition=normalize)
        row.update({k: cv_result[k] for k in ("cv_rmse_mean", "cv_rmse_std", "cv_mae_mean")})
    preds = fit_and_predict_test(subset, model, n_cond, normalize_by_condition=normalize)
    row.update({f"test_{k}": v for k, v in score_test(preds).items()})
    return row


def cmd_evaluate(args: argparse.Namespace) -> None:
    row = _evaluate(args.subset, args.model, Path(args.data_dir), normalize=not args.no_condition_norm)
    print(json.dumps(row, indent=2, ensure_ascii=False))


def cmd_run_all(args: argparse.Namespace) -> None:
    data_dir, out = Path(args.data_dir), Path(args.out)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    t0 = time.time()

    # Deney 1: model karşılaştırması (tek koşullu alt kümeler)
    comparison = []
    for name in ("FD001", "FD003"):
        for model in MODEL_NAMES:
            print(f"[1] {name} · {model}")
            comparison.append(_evaluate(name, model, data_dir))
    results["model_comparison"] = comparison

    # Deney 2: çok koşullu alt kümelerde koşul normalizasyonunun etkisi
    ablation = []
    for name in ("FD002", "FD004"):
        for normalize in (False, True):
            print(f"[2] {name} · normalizasyon={normalize}")
            ablation.append(_evaluate(name, "gradient_boosting", data_dir, normalize=normalize))
    results["condition_normalization"] = ablation

    # Deney 3: sızıntı
    print("[3] sızıntı deneyi")
    fd001 = load_subset("FD001", data_dir)
    results["leakage"] = leakage_experiment(fd001.train, "random_forest", 1)

    # Deney 4: tahmin aralıkları
    interval_rows, interval_frames = [], {}
    for name in SUBSETS:
        print(f"[4] {name} · aralıklar")
        subset = load_subset(name, data_dir)
        preds = fit_and_predict_test(subset, "gradient_boosting", N_OPERATING_CONDITIONS[name], with_intervals=True)
        interval_rows.append({"subset": name, **score_test(preds)})
        interval_frames[name] = preds.to_frame()
        preds.to_frame().to_csv(out / f"test_predictions_{name}.csv", index=False)
    results["intervals"] = interval_rows

    # Deney 5: bakım kararı (senin modülün yazıldıysa)
    if maintenance.is_implemented():
        print("[5] bakım politikası")
        frame = interval_frames["FD001"]
        thresholds = list(range(0, 61, 5))
        results["maintenance"] = {}
        for column in ("pred", "lower"):
            thr, res = maintenance.best_threshold(frame, column, thresholds)
            results["maintenance"][column] = {"best_threshold": thr, **res}
    else:
        print("[5] bakım modülü henüz yazılmadı, atlanıyor (src/rul/maintenance.py)")

    # Grafikler
    print("grafikler")
    plots.sensor_trends(fd001.train, ["s2", "s4", "s11", "s15"], figures / "sensor_trends_fd001.png")
    plots.condition_effect(load_subset("FD002", data_dir).train, "s2", figures / "condition_effect_fd002.png")
    plots.pred_vs_true(interval_frames["FD001"], "FD001 test: tahmin ve gerçek", figures / "pred_vs_true_fd001.png")
    plots.intervals(interval_frames["FD001"], "FD001 test: %80 tahmin aralıkları", figures / "intervals_fd001.png")
    plots.leakage_bar(results["leakage"], figures / "leakage_fd001.png")
    plots.model_comparison(comparison, figures / "model_comparison.png")

    results["runtime_seconds"] = round(time.time() - t0, 1)
    (out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Bitti: {out / 'results.json'} ({results['runtime_seconds']} sn)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rul", description="C-MAPSS kalan ömür tahmini")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("evaluate", help="Tek alt küme ve model için CV + test")
    p_eval.add_argument("--subset", choices=SUBSETS, default="FD001")
    p_eval.add_argument("--model", choices=MODEL_NAMES, default="gradient_boosting")
    p_eval.add_argument("--no-condition-norm", action="store_true")
    p_eval.set_defaults(func=cmd_evaluate)

    p_all = sub.add_parser("run-all", help="Bütün deneyler, sonuçlar ve grafikler")
    p_all.add_argument("--out", default=str(DEFAULT_REPORT_DIR))
    p_all.set_defaults(func=cmd_run_all)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
