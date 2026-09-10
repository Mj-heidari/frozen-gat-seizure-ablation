"""Tuned re-analysis of the prepared CHB-MIT cache.

Two changes versus `eegstudy.cli run`, both confined to training folds:

  1. The decision threshold is chosen on inner-CV out-of-fold predictions
     instead of being fixed at 0.5. At ~2 % prevalence a fixed 0.5 is a poor
     operating point and depresses balanced accuracy for every variant.
  2. The prior strengths alpha (node reliability) and beta (edge stability)
     and the readout regularisation C are selected by an inner grouped CV
     inside each outer training fold.

The outer folds, groups, corruption seeds and the primary contrast are
unchanged. No test-fold information is used for any choice. alpha = 0 and
beta = 0 are in the grid, so the tuning can legitimately conclude that the
priors are worthless.

Features (Welch spectra, correlations, stability) are computed once per
corruption condition and reused, which is why this is faster than the
untuned run despite the grid search.

    python sweep.py --config configs/windows_chb_full.json --out runs/sweep_001
"""
from __future__ import annotations

import argparse, json, platform, time
from pathlib import Path

import numpy as np
import scipy, sklearn
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

from eegstudy import data as D
from eegstudy.model import features, corrupt, EPS

CONDITIONS = ["clean", "drop2", "drop4", "noise"]
TEST_SEED_BASE = 70000
CHUNK = 2048


# ---------------------------------------------------------------- features
def feature_pass(X, kind, seed_base):
    """features() over every window, with `kind` corruption applied per sample."""
    n = len(X)
    z = np.empty((n, X.shape[1], 7), np.float32)
    q = np.empty((n, X.shape[1]), np.float32)
    a = np.empty((n, X.shape[1], X.shape[1]), np.float32)
    s = np.empty_like(a)
    for i0 in range(0, n, CHUNK):
        i1 = min(i0 + CHUNK, n)
        blk = np.stack([corrupt(X[i][None], kind, seed_base + int(i))[0] for i in range(i0, i1)])
        zz, qq, aa, ss = features(blk)
        z[i0:i1], q[i0:i1], a[i0:i1], s[i0:i1] = zz, qq, aa, ss
    return dict(z=z, q=q, a=a, st=s)


# ---------------------------------------------------------------- encoder
class Frozen:
    def __init__(self, seed=17, d=24):
        r = np.random.default_rng(seed)
        self.W = r.normal(0, 1 / np.sqrt(7), (7, d))
        self.V = r.normal(0, 1 / np.sqrt(d), (d, d))
        self.F = r.normal(0, 1 / np.sqrt(d), (d, d))
        self.d = d


def encode(F_, feat, mu, sd, alpha, beta, use_quality, idx=None):
    """Return the 31-d representation for the selected samples."""
    sl = slice(None) if idx is None else idx
    z, q, a, st = feat["z"][sl], feat["q"][sl], feat["a"][sl], feat["st"][sl]
    valid = q > 0
    zs = np.clip((z.astype(np.float64) - mu) / sd, -5, 5)
    zs[~valid] = 0
    C = zs.shape[1]

    h = np.tanh(zs @ F_.W)
    prior = (a.astype(np.float64) * (st.astype(np.float64) ** beta if beta else 1.0)) + np.eye(C)[None]
    logits = (h @ h.transpose(0, 2, 1)) / np.sqrt(F_.d) + np.log(prior + 1e-4)
    if alpha:
        logits = logits + alpha * np.log(q.astype(np.float64)[:, None, :] + EPS)
    logits = np.where(valid[:, None, :], logits, -1e9)
    att = softmax(logits, axis=-1)
    h = np.tanh(h + att @ (h @ F_.V))
    h = np.tanh(h + h @ F_.F)

    w = q.astype(np.float64) if use_quality else valid.astype(np.float64)
    den = w.sum(1)[:, None]
    if np.any(den < EPS):
        raise ValueError("All channels invalid in an epoch")
    u = (h * w[..., None]).sum(1) / den
    raw = (zs * w[..., None]).sum(1) / den
    return np.concatenate([u, raw], 1)          # 24 + 7 = 31


def moments(feat, idx):
    z = feat["z"][idx].astype(np.float64)
    v = feat["q"][idx] > 0
    zz = z[v]
    if not len(zz):
        raise ValueError("No valid training nodes")
    return zz.mean(0), np.maximum(zz.std(0), 0.1)


# ---------------------------------------------------------------- metrics
def ba_at(y, p, t):
    return 0.5 * (np.mean(p[y == 1] >= t) + np.mean(p[y == 0] < t))


def best_threshold(y, p):
    grid = np.unique(np.quantile(p, np.linspace(0.001, 0.999, 400)))
    scores = [ba_at(y, p, t) for t in grid]
    return float(grid[int(np.argmax(scores))]), float(np.max(scores))


def metrics_at(y, p, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        balanced_accuracy=float(ba_at(y, p, t)),
        auc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        f1=float(f1_score(y, pred, zero_division=0)),
        sensitivity=float(tp / max(tp + fn, 1)),
        specificity=float(tn / max(tn + fp, 1)),
        brier=float(np.mean((p - y) ** 2)),
        threshold=float(t),
    )


def weights(y, g):
    w = np.array([1 / np.sum(g == v) for v in g], float)
    for cls in (0, 1):
        m = w[y == cls].sum()
        if m == 0:
            raise ValueError("Training requires both classes")
        w[y == cls] /= m
    return w * len(w) / w.sum()


def cluster_ci(y, p, g, q=None, thr=0.5, thr_q=None, seed=900, repeats=1000):
    rng = np.random.default_rng(seed)
    uniq = np.unique(g)
    ixs = {k: np.flatnonzero(g == k) for k in uniq}
    out = []
    for _ in range(repeats):
        ix = np.concatenate([ixs[k] for k in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[ix])) < 2:
            continue
        s = ba_at(y[ix], p[ix], thr)
        out.append(s if q is None else s - ba_at(y[ix], q[ix], thr_q if thr_q is not None else thr))
    return np.quantile(out, [0.025, 0.975]).tolist() if out else None


# ---------------------------------------------------------------- variants
def grids(name):
    Cs = [0.03, 0.1, 0.3]
    if name == "spectral":
        return [(0.0, 0.0, c) for c in Cs]
    if name == "plain":
        return [(0.0, 0.0, c) for c in Cs]
    if name == "quality":
        return [(a, 0.0, c) for a in (0.25, 0.5, 1.0) for c in Cs]
    return [(a, b, c) for a in (0.0, 0.5, 1.0) for b in (0.0, 0.5, 1.0) for c in Cs]


USE_QUALITY = {"spectral": False, "plain": False, "quality": True, "proposed": True, "proposed_aug": True}


def fit_predict(F_, feat_tr, feat_te, ytr, gtr, tr, te, name, alpha, beta, C, aug=None):
    mu, sd = moments(feat_tr, tr)
    uq = USE_QUALITY[name]
    Xtr = encode(F_, feat_tr, mu, sd, alpha, beta, uq, tr)
    ytr_f, gtr_f = ytr, gtr
    if aug is not None:
        extra = [encode(F_, f, mu, sd, alpha, beta, uq, tr) for f in aug]
        Xtr = np.concatenate([Xtr] + extra)
        ytr_f = np.concatenate([ytr] * (1 + len(extra)))
        gtr_f = np.concatenate([gtr] * (1 + len(extra)))
    if name == "spectral":
        Xtr = Xtr[:, 24:31]
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs").fit(
        sc.transform(Xtr), ytr_f, sample_weight=weights(ytr_f, gtr_f))

    Xte = encode(F_, feat_te, mu, sd, alpha, beta, uq, te)
    if name == "spectral":
        Xte = Xte[:, 24:31]
    return clf.predict_proba(sc.transform(Xte))[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/windows_chb_full.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--inner", type=int, default=3)
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Output directory must be new/empty")
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    t0 = time.perf_counter()

    recs, y, g, ids = D.load_cache(cfg)
    X = np.stack([r[0] for r in recs]).astype(np.float32)
    del recs
    print(f"loaded {len(y)} windows, {len(set(g))} groups, prevalence {y.mean():.4f}", flush=True)

    feats = {}
    for c in CONDITIONS:
        t = time.perf_counter()
        feats[c] = feature_pass(X, c, TEST_SEED_BASE)
        print(f"  features[{c}] {time.perf_counter()-t:.0f}s", flush=True)
    aug_feats = []
    for c in ("drop2", "noise"):
        t = time.perf_counter()
        aug_feats.append(feature_pass(X, c, int(cfg["seed"])))
        print(f"  features[aug:{c}] {time.perf_counter()-t:.0f}s", flush=True)
    del X

    F_ = Frozen(seed=17)
    variants = ["spectral", "plain", "quality", "proposed", "proposed_aug"]
    pred = {v: {c: np.zeros(len(y)) for c in CONDITIONS} for v in variants}
    thr = {v: [] for v in variants}
    picks = {v: [] for v in variants}

    outer = StratifiedGroupKFold(cfg["folds"], shuffle=True, random_state=cfg["seed"])
    splits = []
    for fold, (tr, te) in enumerate(outer.split(np.zeros(len(y)), y, g)):
        if set(g[tr]) & set(g[te]):
            raise AssertionError("Group leakage")
        splits.append({"fold": fold, "train_groups": sorted(set(g[tr])), "test_groups": sorted(set(g[te]))})
        print(f"fold {fold}: {len(tr)} train / {len(te)} test", flush=True)

        for name in variants:
            aug = aug_feats if name == "proposed_aug" else None
            cands = grids(name)
            best, best_score = None, -1
            if len(cands) > 1:
                inner = StratifiedGroupKFold(a.inner, shuffle=True, random_state=cfg["seed"] + 1)
                for (al, be, C) in cands:
                    oof = np.zeros(len(tr))
                    ok = True
                    for itr, ite in inner.split(np.zeros(len(tr)), y[tr], g[tr]):
                        if len(np.unique(y[tr][itr])) < 2 or len(np.unique(y[tr][ite])) < 2:
                            ok = False
                            break
                        oof[ite] = fit_predict(F_, feats["clean"], feats["clean"], y[tr][itr], g[tr][itr],
                                               tr[itr], tr[ite], name, al, be, C, aug)
                    if not ok:
                        continue
                    _, sc = best_threshold(y[tr], oof)
                    if sc > best_score:
                        best_score, best, best_oof = sc, (al, be, C), oof
            else:
                al, be, C = cands[0]
                inner = StratifiedGroupKFold(a.inner, shuffle=True, random_state=cfg["seed"] + 1)
                oof = np.zeros(len(tr))
                for itr, ite in inner.split(np.zeros(len(tr)), y[tr], g[tr]):
                    oof[ite] = fit_predict(F_, feats["clean"], feats["clean"], y[tr][itr], g[tr][itr],
                                           tr[itr], tr[ite], name, al, be, C, aug)
                best, best_oof = (al, be, C), oof
                best_score = best_threshold(y[tr], oof)[1]

            t_star, _ = best_threshold(y[tr], best_oof)
            thr[name].append(t_star)
            picks[name].append({"fold": fold, "alpha": best[0], "beta": best[1], "C": best[2],
                                "threshold": t_star, "inner_ba": best_score})
            for c in CONDITIONS:
                pred[name][c][te] = fit_predict(F_, feats["clean"], feats[c], y[tr], g[tr],
                                                tr, te, name, best[0], best[1], best[2], aug)
            print(f"   {name:13s} alpha={best[0]} beta={best[1]} C={best[2]} "
                  f"thr={t_star:.3f} innerBA={best_score:.4f}", flush=True)

    T = {v: float(np.mean(thr[v])) for v in variants}
    summary = {v: {c: {**metrics_at(y, pred[v][c], T[v]),
                       "ba_group_ci95": cluster_ci(y, pred[v][c], g, thr=T[v])}
                   for c in CONDITIONS} for v in variants}
    paired = {c: {"delta_ba": float(ba_at(y, pred["proposed"][c], T["proposed"])
                                    - ba_at(y, pred["plain"][c], T["plain"])),
                  "ci95": cluster_ci(y, pred["proposed"][c], g, pred["plain"][c],
                                     thr=T["proposed"], thr_q=T["plain"])}
              for c in CONDITIONS}

    result = {"data_kind": "real", "clinical_validation": False, "study": "chb",
              "analysis": "tuned: inner-CV threshold and prior strengths",
              "n_samples": int(len(y)), "n_groups": int(len(set(g))),
              "prevalence": float(y.mean()), "seconds": time.perf_counter() - t0,
              "summary": summary, "paired_proposed_minus_plain": paired,
              "selected": picks, "mean_threshold": T, "config": cfg,
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "scipy": scipy.__version__, "sklearn": sklearn.__version__}}
    (out / "results.json").write_text(json.dumps(result, indent=2))
    (out / "splits.json").write_text(json.dumps(splits, indent=2))
    np.savez_compressed(out / "predictions.npz", y=y, g=g, ids=ids,
                        **{f"{v}__{c}": pred[v][c] for v in variants for c in CONDITIONS})
    print(json.dumps({k: result[k] for k in ["n_samples", "n_groups", "prevalence", "seconds"]}))
    for c in CONDITIONS:
        print(f"  proposed-plain {c}: {paired[c]['delta_ba']:+.4f} {paired[c]['ci95']}")


if __name__ == "__main__":
    main()
