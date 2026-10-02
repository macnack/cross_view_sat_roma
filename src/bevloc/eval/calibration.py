"""Per-frame confidence: calibrators, ranking and calibration metrics, abstention curves, split conformal (task 06).

numpy only (sklearn is not in the Eagle container; `gbm_available` guards the optional gradient-boosted model).
Convention: a *score* / *confidence* is higher when the frame is more likely correct; the label y is 1 when the pose
error is below the target radius (a frame with no pose is a failure, error = inf).
"""
from __future__ import annotations

import math

import numpy as np


# ---- ranking metrics -------------------------------------------------------------------------------------------

def _ranks(x):
    """Average ranks (1-based) with ties sharing their mean rank."""
    x = np.asarray(x, np.float64)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    r = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def auroc(score, y):
    """Area under the ROC curve (Mann-Whitney; ties count one half). nan when y has one class only."""
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = _ranks(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def average_precision(score, y):
    """Average precision (step-wise, sklearn's definition): sum over thresholds of (R_k - R_{k-1}) P_k, tied scores
    forming one threshold."""
    y = np.asarray(y, bool)
    if y.sum() == 0:
        return float("nan")
    s = np.asarray(score, np.float64)
    order = np.argsort(-s, kind="mergesort")
    s, yy = s[order], y[order]
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]         # end index of every distinct threshold
    tp = np.cumsum(yy)[last]
    prec = tp / (last + 1.0)
    rec = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


# ---- calibration metrics ---------------------------------------------------------------------------------------

def reliability(prob, y, n_bins=10):
    """Equal-width bins of predicted probability: list of dict(lo, hi, n, conf (mean prob), acc (fraction correct)),
    and the expected calibration error sum_b n_b / N |acc_b - conf_b|."""
    p = np.clip(np.asarray(prob, np.float64), 0.0, 1.0)
    y = np.asarray(y, np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    bins, ece = [], 0.0
    for i in range(n_bins):
        sel = b == i
        n = int(sel.sum())
        conf = float(p[sel].mean()) if n else None
        acc = float(y[sel].mean()) if n else None
        bins.append(dict(lo=float(edges[i]), hi=float(edges[i + 1]), n=n, conf=conf, acc=acc))
        if n:
            ece += n / len(p) * abs(acc - conf)
    return bins, float(ece)


def brier(prob, y):
    return float(np.mean((np.asarray(prob, np.float64) - np.asarray(y, np.float64)) ** 2))


# ---- abstention ------------------------------------------------------------------------------------------------

def abstention_curve(conf, err, coverages=None, gross_m=10.0):
    """Frames kept in decreasing confidence (stable order for ties); at each coverage c the first ceil(c N) are kept.
    Returns dict(coverage, median_m, gross_frac (error > gross_m), n) lists. err may hold inf (no pose)."""
    conf = np.asarray(conf, np.float64)
    err = np.asarray(err, np.float64)
    if coverages is None:
        coverages = np.round(np.arange(0.05, 1.0001, 0.01), 4)
    order = np.argsort(-conf, kind="mergesort")
    e = err[order]
    out = dict(coverage=[], median_m=[], gross_frac=[], n=[])
    for c in coverages:
        k = max(1, int(math.ceil(float(c) * len(e) - 1e-9)))
        out["coverage"].append(float(c))
        out["median_m"].append(float(np.median(e[:k])))
        out["gross_frac"].append(float(np.mean(e[:k] > gross_m)))
        out["n"].append(k)
    return out


def at_coverage(curve, c):
    """The point of an abstention curve at coverage c (nearest grid point)."""
    i = int(np.argmin(np.abs(np.asarray(curve["coverage"]) - c)))
    return dict(coverage=curve["coverage"][i], median_m=curve["median_m"][i], gross_frac=curve["gross_frac"][i],
                n=curve["n"][i])


# ---- calibrators -----------------------------------------------------------------------------------------------

class Isotonic:
    """Monotone non-decreasing P(y | x) by pool-adjacent-violators on one feature; piecewise-linear between the
    pooled block centres, constant beyond the ends."""

    def fit(self, x, y):
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        order = np.argsort(x, kind="mergesort")
        xs, ys = x[order], y[order]
        ux, inv = np.unique(xs, return_inverse=True)            # ties pooled first
        w = np.bincount(inv).astype(np.float64)
        v = np.bincount(inv, weights=ys) / w
        blocks = []                                             # [value, weight, x-weighted-sum]
        for xi, vi, wi in zip(ux, v, w):
            blocks.append([vi, wi, xi * wi])
            while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
                v2, w2, s2 = blocks.pop()
                v1, w1, s1 = blocks.pop()
                blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, s1 + s2])
        self.x_ = np.array([b[2] / b[1] for b in blocks])
        self.y_ = np.array([b[0] for b in blocks])
        return self

    def predict(self, x):
        return np.interp(np.asarray(x, np.float64), self.x_, self.y_)


class Logistic:
    """L2-regularised logistic regression by Newton's method on standardised features (the fit set's mean / std;
    missing values imputed with the fit set's median). l2 applies to the weights, not the intercept."""

    def __init__(self, l2=1.0, iters=100):
        self.l2, self.iters = float(l2), int(iters)

    def _prep(self, X):
        X = np.asarray(X, np.float64)
        X = np.where(np.isfinite(X), X, self.med_[None])
        return np.c_[np.ones(len(X)), (X - self.mu_) / self.sd_]

    def fit(self, X, y):
        X = np.asarray(X, np.float64)
        y = np.asarray(y, np.float64)
        med = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0.0 for c in X.T])
        self.med_ = med
        Xi = np.where(np.isfinite(X), X, med[None])
        self.mu_, sd = Xi.mean(0), Xi.std(0)
        self.sd_ = np.where(sd > 1e-12, sd, 1.0)
        A = self._prep(X)
        w = np.zeros(A.shape[1])
        R = np.eye(A.shape[1]) * self.l2
        R[0, 0] = 0.0
        for _ in range(self.iters):
            p = 1.0 / (1.0 + np.exp(-np.clip(A @ w, -40, 40)))
            g = A.T @ (p - y) + R @ w
            Hm = (A * (p * (1 - p))[:, None]).T @ A + R + 1e-9 * np.eye(len(w))
            step = np.linalg.solve(Hm, g)
            w -= step
            if np.max(np.abs(step)) < 1e-8:
                break
        self.w_ = w
        return self

    def predict(self, X):
        return 1.0 / (1.0 + np.exp(-np.clip(self._prep(X) @ self.w_, -40, 40)))


def gbm_available():
    try:
        import sklearn.ensemble  # noqa: F401
        return True
    except ImportError:
        return False


class GBM:
    """sklearn's HistGradientBoostingClassifier (handles missing values natively); only when sklearn is installed."""

    def __init__(self, seed=0):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.m = HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.05, l2_regularization=1.0,
                                                min_samples_leaf=20, random_state=seed)

    def fit(self, X, y):
        y = np.asarray(y, int)
        self.const_ = None if len(np.unique(y)) > 1 else float(y[0])
        if self.const_ is None:
            self.m.fit(np.asarray(X, np.float64), y)
        return self

    def predict(self, X):
        if self.const_ is not None:
            return np.full(len(X), self.const_)
        return self.m.predict_proba(np.asarray(X, np.float64))[:, 1]


# ---- split conformal -------------------------------------------------------------------------------------------

def conformal_quantile(scores, alpha):
    """The ceil((n + 1)(1 - alpha))-th smallest calibration score (inf when that exceeds n): split conformal's q-hat."""
    s = np.sort(np.asarray(scores, np.float64))
    n = len(s)
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    return float("inf") if k > n else float(s[k - 1])


def conformal_sets(prob_cal, y_cal, prob_test, alpha):
    """Split-conformal prediction sets for the binary event "correct" (error below the radius) from P(correct):
    nonconformity of the true label = 1 - P(true label). Guarantee: P(true label in set) >= 1 - alpha on exchangeable
    test frames. Returns (q-hat, accept (set = {correct}), reject (= {wrong}), abstain (both), empty) bool arrays and
    the covered-indicator function to be applied to the test labels."""
    p_cal = np.asarray(prob_cal, np.float64)
    y_cal = np.asarray(y_cal, bool)
    s_cal = np.where(y_cal, 1.0 - p_cal, p_cal)
    q = conformal_quantile(s_cal, alpha)
    p = np.asarray(prob_test, np.float64)
    has_ok = (1.0 - p) <= q
    has_bad = p <= q
    return q, has_ok & ~has_bad, has_bad & ~has_ok, has_ok & has_bad, ~has_ok & ~has_bad, (has_ok, has_bad)


def conformal_report(prob_cal, y_cal, prob_test, y_test, alpha):
    """Numbers of `conformal_sets` on labelled test frames: achieved coverage (true label in the set), set-type
    fractions, and the accuracy among accepted frames (set = {correct})."""
    q, acc, rej, abst, empty, (has_ok, has_bad) = conformal_sets(prob_cal, y_cal, prob_test, alpha)
    y = np.asarray(y_test, bool)
    covered = np.where(y, has_ok, has_bad)
    return dict(alpha=float(alpha), target_coverage=1.0 - float(alpha), qhat=q, n_cal=int(len(prob_cal)),
                n_test=int(len(y)), coverage=float(covered.mean()), accept=float(acc.mean()),
                reject=float(rej.mean()), abstain=float(abst.mean()), empty=float(empty.mean()),
                accuracy_accepted=float(y[acc].mean()) if acc.any() else None)
