#!/usr/bin/env python3
"""03_stats_from_gee.py - Statistik dari prediksi out-of-fold hasil 01_gee_classification.js.

Masukan : oof_pixel.csv dan oof_polygon.csv (ekspor GEE)
Keluaran: table_overall.csv, table_perclass.csv, confusion_long.csv,
          mcnemar_pairs.csv, inflation.csv

  python 03_stats_from_gee.py --pixel oof_pixel.csv --polygon oof_polygon.csv --out hasil_gee

Kebutuhan: numpy pandas scipy scikit-learn
"""
import argparse
import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("reanalysis", HERE / "02_reanalysis.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

ORDER = R.ORDER
CLASS_NAMES = ["BU", "F", "VC", "NVC"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pixel", required=True)
    ap.add_argument("--polygon", required=True)
    ap.add_argument("--out", default="hasil_gee")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    overall, perclass, confs, tests = [], [], [], []
    for design, path in [("pixel", a.pixel), ("polygon", a.polygon)]:
        d = pd.read_csv(path).dropna(subset=["class", "poly_id"] + ORDER)
        y = d["class"].astype(int).to_numpy()
        groups = d["poly_id"].to_numpy()
        fold = d["fold"].astype(int).to_numpy()
        K = len(CLASS_NAMES)
        preds = {m: d[m].astype(int).to_numpy() for m in ORDER}
        correct = np.column_stack([preds[m] == y for m in ORDER])
        boot = R.cluster_bootstrap(correct.astype(float), groups, a.boot)
        print(f"{design}: {len(y):,} piksel, {len(np.unique(groups))} poligon")

        for j, m in enumerate(ORDER):
            cm, ua, pa, f1, iou = R.class_metrics(y, preds[m], K)
            folds = np.unique(fold)
            fold_oa = [correct[fold == k, j].mean() for k in folds]
            overall.append({
                "design": design, "model": m, "n": len(y),
                "OA": 100 * correct[:, j].mean(),
                "OA_ci_low": 100 * np.percentile(boot[:, j], 2.5),
                "OA_ci_high": 100 * np.percentile(boot[:, j], 97.5),
                "OA_fold_sd": 100 * np.std(fold_oa, ddof=1),
                "kappa": cohen_kappa_score(y, preds[m]),
                "UA_macro": 100 * np.nanmean(ua), "PA_macro": 100 * pa.mean(),
                "F1_macro": 100 * f1.mean(), "IoU_macro": 100 * iou.mean(),
            })
            for k, c in enumerate(CLASS_NAMES):
                perclass.append({"design": design, "model": m, "class": c,
                                 "UA": 100 * ua[k], "PA": 100 * pa[k],
                                 "F1": 100 * f1[k], "IoU": 100 * iou[k]})
                for k2, c2 in enumerate(CLASS_NAMES):
                    confs.append({"design": design, "model": m, "reference": c,
                                  "predicted": c2, "n": int(cm[k, k2])})

        rows = []
        for i, j in itertools.combinations(range(len(ORDER)), 2):
            b, c, chi2, p = R.mcnemar(correct[:, i], correct[:, j])
            diff = boot[:, i] - boot[:, j]
            rows.append({"design": design, "model_a": ORDER[i], "model_b": ORDER[j],
                         "only_a_correct": b, "only_b_correct": c, "chi2": chi2, "p": p,
                         "OA_diff": 100 * (correct[:, i].mean() - correct[:, j].mean()),
                         "diff_ci_low": 100 * np.percentile(diff, 2.5),
                         "diff_ci_high": 100 * np.percentile(diff, 97.5)})
        t = pd.DataFrame(rows)
        t["p_holm"] = R.holm(t["p"])
        t["significant_holm_0.05"] = t["p_holm"] < 0.05
        t["ci_excludes_zero"] = (t["diff_ci_low"] > 0) | (t["diff_ci_high"] < 0)
        tests.append(t)

    ov = pd.DataFrame(overall)
    ov.round(3).to_csv(out / "table_overall.csv", index=False)
    pd.DataFrame(perclass).round(2).to_csv(out / "table_perclass.csv", index=False)
    pd.DataFrame(confs).to_csv(out / "confusion_long.csv", index=False)
    pd.concat(tests).round(5).to_csv(out / "mcnemar_pairs.csv", index=False)

    wide = ov.pivot(index="model", columns="design", values="OA").loc[ORDER]
    infl = pd.DataFrame({
        "OA_pixel": wide["pixel"], "OA_polygon": wide["polygon"],
        "inflation_points": wide["pixel"] - wide["polygon"],
        "rank_pixel": wide["pixel"].rank(ascending=False, method="min").astype(int),
        "rank_polygon": wide["polygon"].rank(ascending=False, method="min").astype(int),
    })
    infl.round(2).to_csv(out / "inflation.csv")
    rho = stats.spearmanr(infl["OA_pixel"], infl["OA_polygon"]).statistic
    print("\nOA dan selang 95% (bootstrap per poligon):")
    print(ov[["design", "model", "OA", "OA_ci_low", "OA_ci_high", "kappa", "F1_macro"]]
          .round(2).to_string(index=False))
    print("\nInflasi akurasi (piksel-acak dikurangi per-poligon):")
    print(infl.round(2).to_string())
    print(f"\nKorelasi peringkat Spearman antar desain: {rho:.3f}")
    print(f"Tabel tersimpan di {out}/")


if __name__ == "__main__":
    main()
