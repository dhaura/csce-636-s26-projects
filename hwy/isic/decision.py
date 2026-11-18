"""Decision layers fitted on out-of-fold (OOF) predictions, targeting balanced accuracy.

All functions work on log-probabilities z of shape (n, C) and integer labels y.
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

EPS = 1e-7


def log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def balanced_accuracy(y, pred, n_classes):
    cm = np.bincount(y * n_classes + pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    return float(np.mean(np.diag(cm) / np.maximum(cm.sum(axis=1), 1)))


def fit_temperature(z, y):
    """Scalar temperature T minimising the NLL of softmax(z / T)."""
    idx = np.arange(len(y))

    def nll(t):
        return -log_softmax(z / t[0])[idx, y].mean()

    return float(minimize(nll, [1.0], bounds=[(0.05, 20.0)]).x[0])


def fit_tau(z, y, log_prior, taus=np.arange(-1.0, 3.01, 0.1)):
    """Logit-adjustment strength tau maximising balanced accuracy of argmax(z - tau * log_prior)."""
    C = z.shape[1]
    scores = [balanced_accuracy(y, (z - t * log_prior).argmax(1), C) for t in taus]
    return float(taus[int(np.argmax(scores))])


def fit_class_bias(z, y, init, passes=4, step=0.05, span=2.0):
    """Per-class additive offsets b maximising balanced accuracy of argmax(z + b), by coordinate ascent.

    Starts from `init` (e.g. -tau * log_prior); each pass scans every class offset over init +- span.
    Offsets are only identified up to a constant, so the result is centred.
    """
    C = z.shape[1]
    b = np.array(init, dtype=float)
    best = balanced_accuracy(y, (z + b).argmax(1), C)
    grid = np.arange(-span, span + 1e-9, step)
    for _ in range(passes):
        improved = False
        for c in range(C):
            base, best_val = b[c], b[c]
            for d in grid:
                b[c] = base + d
                s = balanced_accuracy(y, (z + b).argmax(1), C)
                if s > best + 1e-9:
                    best, best_val, improved = s, b[c], True
            b[c] = best_val
        if not improved:
            break
    return b - b.mean()


def fit_stacker(X, y, C_reg=0.1):
    """Multinomial logistic regression on concatenated member log-probs, class-balanced weights."""
    return LogisticRegression(C=C_reg, class_weight="balanced", max_iter=2000).fit(X, y)
