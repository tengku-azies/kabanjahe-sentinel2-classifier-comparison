#!/usr/bin/env python3
"""02_reanalysis.py - Analisis ulang perbandingan tujuh algoritma klasifikasi.

Semua algoritma dijalankan di satu platform (scikit-learn) pada data yang sama:
  RF, CART, SVM (RBF), Naive Bayes (Gaussian), MLC (= QDA, prior sama),
  Minimum Distance (centroid Euclidean), Spectral Angle Mapper.

Tiga perintah:

  cv        Validasi silang 5-lipat dengan DUA desain pada sampel yang sama:
              pixel   : piksel diacak (desain lama, rawan bocor spasial)
              polygon : seluruh piksel satu poligon masuk lipatan yang sama
            Keluaran: akurasi, selang kepercayaan (bootstrap per poligon),
            uji McNemar 21 pasangan (koreksi Holm), inflasi akurasi.

  map       Melatih model akhir, mengklasifikasi raster, menghitung luas, dan
            membuat titik validasi acak terstratifikasi untuk diinterpretasi.

  validate  Dari titik yang sudah Anda beri label: akurasi dan luas
            terkoreksi galat (penduga terstratifikasi, Stehman 2014;
            Olofsson dkk. 2014) dengan selang kepercayaan 95%.

Contoh:
  python 02_reanalysis.py cv --samples samples.csv --out hasil
  python 02_reanalysis.py map --samples samples.csv --raster composite_10band.tif --out hasil --points 100
  python 02_reanalysis.py validate --labeled hasil/points_for_interpretation.csv --out hasil

Kebutuhan: numpy pandas scipy scikit-learn rasterio
"""
import argparse
import itertools
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
ORDER = ["RF", "CART", "SVM", "NB", "MLC", "MD", "SAM"]
SEED = 42


class SpectralAngleMapper(ClassifierMixin, BaseEstimator):
    """Kelas dengan sudut spektral terkecil terhadap rerata spektrum kelas."""

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.classes_ = np.unique(y)
        self.means_ = np.vstack([X[y == c].mean(axis=0) for c in self.classes_])
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        num = X @ self.means_.T
        den = np.linalg.norm(X, axis=1, keepdims=True) * np.linalg.norm(self.means_, axis=1)
        ang = np.arccos(np.clip(num / np.maximum(den, 1e-12), -1.0, 1.0))
        return self.classes_[ang.argmin(axis=1)]


def build_models(n_classes, fast=False):
    """Model dasar dan kisi hiperparameter."""
    equal = np.full(n_classes, 1.0 / n_classes)
    return {
        "RF": (RandomForestClassifier(n_estimators=100 if fast else 500, random_state=SEED),
               {"max_features": [2, 3, 5], "min_samples_leaf": [1, 3, 5]}),
        "CART": (DecisionTreeClassifier(random_state=SEED),
                 {"max_depth": [5, 10, 20, None], "min_samples_leaf": [1, 5, 10, 20]}),
        "SVM": (make_pipeline(StandardScaler(), SVC(kernel="rbf")),
                {"svc__C": [1, 10, 100, 1000], "svc__gamma": ["scale", 0.01, 0.1, 1]}),
        "NB": (GaussianNB(priors=equal), {"var_smoothing": [1e-9, 1e-6, 1e-3]}),
        "MLC": (QuadraticDiscriminantAnalysis(priors=equal), {"reg_param": [0.0, 1e-4, 1e-2]}),
        "MD": (NearestCentroid(), {}),
        "SAM": (SpectralAngleMapper(), {}),
    }


def load_samples(path):
    df = pd.read_csv(path)
    need = BANDS + ["class", "poly_id"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        sys.exit(f"Kolom tidak ditemukan di {path}: {missing}")
    df = df.dropna(subset=need).reset_index(drop=True)
    df["class"] = df["class"].astype(str).str.replace(r"\.0$", "", regex=True)
    classes = sorted(df["class"].unique())
    y = df["class"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    X = df[BANDS].to_numpy(float)
    groups = df["poly_id"].to_numpy()
    per_class = df.groupby("class")["poly_id"].nunique()
    print("Piksel per kelas:\n", df["class"].value_counts().sort_index().to_string())
    print("Poligon per kelas:\n", per_class.to_string())
    if per_class.min() < 5:
        sys.exit("Ada kelas dengan < 5 poligon. Validasi per poligon butuh minimal 5 "
                 "(disarankan >= 30) poligon per kelas. Tambah poligon lalu ekspor ulang.")
    if per_class.min() < 20:
        print("PERINGATAN: ada kelas dengan < 20 poligon; hasil validasi per poligon akan "
              "tidak stabil. Disarankan >= 30 poligon per kelas.")
    return df, X, y, groups, classes


def fit_model(est, grid, X, y, groups, design, tune):
    if not (tune and grid):
        return clone(est).fit(X, y), None
    if design == "polygon":
        inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
        fit_kw = {"groups": groups}
    else:
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        fit_kw = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs = GridSearchCV(clone(est), grid, cv=inner, scoring="accuracy", n_jobs=-1)
        gs.fit(X, y, **fit_kw)
    return gs.best_estimator_, gs.best_params_


def oof_predictions(X, y, groups, design, models, tune):
    """Prediksi out-of-fold: setiap piksel diprediksi tepat satu kali."""
    if design == "pixel":
        splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
    else:
        splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, groups))
    preds = {m: np.full(len(y), -1) for m in models}
    best = {m: [] for m in models}
    fold = np.empty(len(y), int)
    for k, (tr, te) in enumerate(splits):
        fold[te] = k
        if len(np.unique(y[tr])) < len(np.unique(y)):
            sys.exit(f"Lipatan {k} ({design}): ada kelas yang tidak muncul di data latih. "
                     "Tambah jumlah poligon per kelas.")
        for name, (est, grid) in models.items():
            model, params = fit_model(est, grid, X[tr], y[tr], groups[tr], design, tune)
            preds[name][te] = model.predict(X[te])
            best[name].append(params)
        print(f"  [{design}] lipatan {k + 1}/5 selesai", flush=True)
    return preds, fold, best


def class_metrics(y, p, n_classes):
    cm = confusion_matrix(y, p, labels=range(n_classes))  # baris = referensi, kolom = prediksi
    tp = np.diag(cm).astype(float)
    ref, pred = cm.sum(1), cm.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pa = tp / ref
        ua = np.where(pred > 0, tp / pred, np.nan)
        f1 = np.where(ua + pa > 0, 2 * ua * pa / (ua + pa), 0.0)
        iou = tp / (ref + pred - tp)
    return cm, ua, pa, np.nan_to_num(f1), iou


def cluster_bootstrap(correct, groups, n_boot=2000):
    """OA hasil bootstrap dengan poligon sebagai unit resampling.

    correct: array (n_piksel, n_model) bernilai 0/1. Mengembalikan (n_boot, n_model).
    """
    rng = np.random.default_rng(SEED)
    _, inv = np.unique(groups, return_inverse=True)
    n_poly = inv.max() + 1
    c_sum = np.zeros((n_poly, correct.shape[1]))
    np.add.at(c_sum, inv, correct)
    n_sum = np.bincount(inv, minlength=n_poly).astype(float)
    idx = rng.integers(0, n_poly, size=(n_boot, n_poly))
    return c_sum[idx].sum(1) / n_sum[idx].sum(1)[:, None]


def holm(p):
    p = np.asarray(p, float)
    adj, running, m = np.empty(len(p)), 0.0, len(p)
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def mcnemar(ci, cj):
    b = int(np.sum(ci & ~cj))
    c = int(np.sum(~ci & cj))
    if b + c == 0:
        return b, c, 0.0, 1.0
    if b + c < 25:
        return b, c, np.nan, stats.binomtest(min(b, c), b + c, 0.5).pvalue
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return b, c, chi2, stats.chi2.sf(chi2, 1)


def cmd_cv(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df, X, y, groups, classes = load_samples(a.samples)
    K = len(classes)
    models = build_models(K, a.fast)
    overall, perclass, confs, tests, params, oof = [], [], [], [], {}, df[["class", "poly_id"]].copy()

    for design in ["pixel", "polygon"]:
        print(f"Validasi silang, desain: {design}")
        preds, fold, best = oof_predictions(X, y, groups, design, models, not a.no_tune)
        params[design] = best
        oof[f"fold_{design}"] = fold
        correct = np.column_stack([preds[m] == y for m in ORDER])
        boot = cluster_bootstrap(correct.astype(float), groups, a.boot)
        for j, m in enumerate(ORDER):
            oof[f"{m}_{design}"] = [classes[i] for i in preds[m]]
            cm, ua, pa, f1, iou = class_metrics(y, preds[m], K)
            fold_oa = [correct[fold == k, j].mean() for k in range(5)]
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
            for k, c in enumerate(classes):
                perclass.append({"design": design, "model": m, "class": c,
                                 "UA": 100 * ua[k], "PA": 100 * pa[k],
                                 "F1": 100 * f1[k], "IoU": 100 * iou[k]})
                for k2, c2 in enumerate(classes):
                    confs.append({"design": design, "model": m, "reference": c,
                                  "predicted": c2, "n": int(cm[k, k2])})
        rows = []
        for i, j in itertools.combinations(range(len(ORDER)), 2):
            b, c, chi2, p = mcnemar(correct[:, i], correct[:, j])
            d = boot[:, i] - boot[:, j]
            rows.append({"design": design, "model_a": ORDER[i], "model_b": ORDER[j],
                         "only_a_correct": b, "only_b_correct": c, "chi2": chi2, "p": p,
                         "OA_diff": 100 * (correct[:, i].mean() - correct[:, j].mean()),
                         "diff_ci_low": 100 * np.percentile(d, 2.5),
                         "diff_ci_high": 100 * np.percentile(d, 97.5)})
        t = pd.DataFrame(rows)
        t["p_holm"] = holm(t["p"])
        t["significant_holm_0.05"] = t["p_holm"] < 0.05
        t["ci_excludes_zero"] = (t["diff_ci_low"] > 0) | (t["diff_ci_high"] < 0)
        tests.append(t)

    ov = pd.DataFrame(overall)
    ov.round(3).to_csv(out / "table_overall.csv", index=False)
    pd.DataFrame(perclass).round(2).to_csv(out / "table_perclass.csv", index=False)
    pd.DataFrame(confs).to_csv(out / "confusion_long.csv", index=False)
    pd.concat(tests).round(5).to_csv(out / "mcnemar_pairs.csv", index=False)
    oof.to_csv(out / "oof_predictions.csv", index=False)
    (out / "best_params.json").write_text(json.dumps(params, indent=1, default=str))

    wide = ov.pivot(index="model", columns="design", values="OA").loc[ORDER]
    infl = pd.DataFrame({
        "OA_pixel": wide["pixel"], "OA_polygon": wide["polygon"],
        "inflation_points": wide["pixel"] - wide["polygon"],
        "rank_pixel": wide["pixel"].rank(ascending=False, method="min").astype(int),
        "rank_polygon": wide["polygon"].rank(ascending=False, method="min").astype(int),
    })
    infl.round(2).to_csv(out / "inflation.csv")
    rho = stats.spearmanr(infl["OA_pixel"], infl["OA_polygon"]).statistic
    print("\nInflasi akurasi (OA piksel-acak dikurangi OA per-poligon):")
    print(infl.round(2).to_string())
    print(f"\nKorelasi peringkat Spearman antar desain: {rho:.3f}")
    print(f"Semua tabel tersimpan di: {out}/")


def cmd_map(a):
    import rasterio
    from rasterio.warp import transform as warp_transform

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df, X, y, groups, classes = load_samples(a.samples)
    K = len(classes)
    models = build_models(K, a.fast)

    with rasterio.open(a.raster) as src:
        arr = src.read().astype("float32")
        profile, tf, crs = src.profile, src.transform, src.crs
        nodata = src.nodata
    if arr.shape[0] != len(BANDS):
        sys.exit(f"Raster harus {len(BANDS)} band dengan urutan {BANDS}; ditemukan {arr.shape[0]}.")
    valid = np.all(np.isfinite(arr), axis=0) & (arr.sum(axis=0) > 0)
    if nodata is not None:
        valid &= ~np.all(arr == nodata, axis=0)
    px_ha = abs(tf.a * tf.e) / 10000.0
    pix = arr[:, valid].T.astype(float)
    print(f"Piksel valid: {valid.sum():,} ({valid.sum() * px_ha:,.2f} ha)")

    profile.update(count=1, dtype="uint8", nodata=0, compress="deflate")
    maps, areas = {}, []
    for m in ORDER:
        est, grid = models[m]
        model, _ = fit_model(est, grid, X, y, groups, "polygon", not a.no_tune)
        lab = np.zeros(valid.shape, "uint8")
        lab[valid] = model.predict(pix) + 1  # kode 1..K, 0 = nodata
        maps[m] = lab
        with rasterio.open(out / f"map_{m}.tif", "w", **profile) as dst:
            dst.write(lab, 1)
        counts = np.bincount(lab[valid], minlength=K + 1)[1:]
        areas.append({"model": m, **{c: counts[i] * px_ha for i, c in enumerate(classes)},
                      "total": counts.sum() * px_ha})
        print(f"  peta {m} selesai", flush=True)
    pd.DataFrame(areas).round(2).to_csv(out / "area_mapped_ha.csv", index=False)
    pd.DataFrame({"code": range(1, K + 1), "class": classes}).to_csv(out / "class_codes.csv", index=False)

    if a.points > 0:
        rng = np.random.default_rng(SEED)
        strata = maps[a.strata_model]
        rows, cols, weights = [], [], []
        for h in range(1, K + 1):
            r, c = np.nonzero(strata == h)
            weights.append({"stratum": classes[h - 1], "n_pixels": len(r), "px_area_ha": px_ha})
            take = rng.choice(len(r), size=min(a.points, len(r)), replace=False)
            rows.extend(r[take])
            cols.extend(c[take])
        rows, cols = np.array(rows), np.array(cols)
        xs, ys = rasterio.transform.xy(tf, rows, cols)
        lon, lat = warp_transform(crs, "EPSG:4326", list(xs), list(ys))
        order = rng.permutation(len(rows))  # acak urutan agar interpreter tidak bisa menebak strata
        ids = np.arange(1, len(rows) + 1)
        key = pd.DataFrame({"id": ids, "stratum": [classes[strata[r, c] - 1] for r, c in zip(rows[order], cols[order])]})
        for m in ORDER:
            key[m] = [classes[maps[m][r, c] - 1] for r, c in zip(rows[order], cols[order])]
        pts = pd.DataFrame({"id": ids, "longitude": np.array(lon)[order], "latitude": np.array(lat)[order],
                            "reference": "", "confidence": "", "note": ""})
        pts.to_csv(out / "points_for_interpretation.csv", index=False)
        key.to_csv(out / "points_key_DO_NOT_OPEN_WHILE_LABELING.csv", index=False)
        pd.DataFrame(weights).to_csv(out / "strata_weights.csv", index=False)
        print(f"{len(pts)} titik validasi dibuat. Isi kolom 'reference' dengan salah satu dari: {classes}")


def strat_mean(yv, h, W):
    """Penduga rerata terstratifikasi dan ragamnya."""
    est = var = 0.0
    for s, w in W.items():
        v = yv[h == s]
        est += w * v.mean()
        var += w ** 2 * (v.var(ddof=1) if len(v) > 1 else 0.0) / len(v)
    return est, var


def strat_ratio(yv, xv, h, W):
    """Penduga rasio terstratifikasi (untuk UA dan PA) dan ragamnya (Stehman 2014)."""
    Y, _ = strat_mean(yv, h, W)
    Xm, _ = strat_mean(xv, h, W)
    if Xm == 0:
        return np.nan, np.nan
    R, var = Y / Xm, 0.0
    for s, w in W.items():
        ys_, xs_ = yv[h == s], xv[h == s]
        if len(ys_) > 1:
            cov = np.cov(ys_, xs_, ddof=1)
            var += w ** 2 * (cov[0, 0] + R ** 2 * cov[1, 1] - 2 * R * cov[0, 1]) / len(ys_)
    return R, max(var, 0.0) / Xm ** 2


def cmd_validate(a):
    out = Path(a.out)
    lab = pd.read_csv(a.labeled, dtype={"reference": str})
    key = pd.read_csv(out / "points_key_DO_NOT_OPEN_WHILE_LABELING.csv", dtype=str)
    key["id"] = key["id"].astype(int)
    wdf = pd.read_csv(out / "strata_weights.csv", dtype={"stratum": str})
    lab["reference"] = lab["reference"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    d = key.merge(lab[["id", "reference"]], on="id")
    classes = list(wdf["stratum"])
    bad = d[~d["reference"].isin(classes)]
    if len(bad):
        print(f"{len(bad)} titik tanpa label sah dikeluarkan (id: {bad['id'].tolist()[:20]} ...)")
        d = d[d["reference"].isin(classes)]
    N = wdf["n_pixels"].sum()
    W = dict(zip(wdf["stratum"], wdf["n_pixels"] / N))
    total_ha = N * wdf["px_area_ha"].iloc[0]
    h, ref = d["stratum"].to_numpy(), d["reference"].to_numpy()
    print("Titik per strata:\n", d["stratum"].value_counts().to_string())

    ov, pc = [], []
    for m in ORDER:
        mp = d[m].to_numpy()
        oa, v = strat_mean((mp == ref).astype(float), h, W)
        ov.append({"model": m, "n": len(d), "OA": 100 * oa,
                   "ci_low": 100 * (oa - 1.96 * v ** 0.5), "ci_high": 100 * (oa + 1.96 * v ** 0.5)})
        for c in classes:
            both = ((mp == c) & (ref == c)).astype(float)
            ua, vu = strat_ratio(both, (mp == c).astype(float), h, W)
            pa, vp = strat_ratio(both, (ref == c).astype(float), h, W)
            pc.append({"model": m, "class": c, "UA": 100 * ua, "UA_se": 100 * vu ** 0.5,
                       "PA": 100 * pa, "PA_se": 100 * vp ** 0.5})
    pd.DataFrame(ov).round(2).to_csv(out / "validation_overall.csv", index=False)
    pd.DataFrame(pc).round(2).to_csv(out / "validation_perclass.csv", index=False)

    mapped = pd.read_csv(out / "area_mapped_ha.csv").set_index("model")
    rows = []
    for c in classes:
        p, v = strat_mean((ref == c).astype(float), h, W)
        row = {"class": c, "area_adjusted_ha": p * total_ha,
               "ci95_halfwidth_ha": 1.96 * v ** 0.5 * total_ha}
        for m in ORDER:
            row[f"mapped_{m}_ha"] = mapped.loc[m, c]
        rows.append(row)
    adj = pd.DataFrame(rows)
    adj.round(2).to_csv(out / "area_adjusted_ha.csv", index=False)
    print("\nAkurasi pada titik validasi independen:")
    print(pd.DataFrame(ov).round(2).to_string(index=False))
    print("\nLuas terkoreksi galat vs luas hasil peta (ha):")
    print(adj.round(1).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("cv")
    p.add_argument("--samples", required=True)
    p.add_argument("--out", default="hasil")
    p.add_argument("--no-tune", action="store_true", help="pakai nilai bawaan, tanpa tuning")
    p.add_argument("--fast", action="store_true", help="RF 100 pohon (uji coba cepat)")
    p.add_argument("--boot", type=int, default=2000)
    p.set_defaults(func=cmd_cv)

    p = sub.add_parser("map")
    p.add_argument("--samples", required=True)
    p.add_argument("--raster", required=True)
    p.add_argument("--out", default="hasil")
    p.add_argument("--points", type=int, default=100, help="titik validasi per strata (0 = tidak dibuat)")
    p.add_argument("--strata-model", default="RF", choices=ORDER)
    p.add_argument("--no-tune", action="store_true")
    p.add_argument("--fast", action="store_true")
    p.set_defaults(func=cmd_map)

    p = sub.add_parser("validate")
    p.add_argument("--labeled", required=True)
    p.add_argument("--out", default="hasil")
    p.set_defaults(func=cmd_validate)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
