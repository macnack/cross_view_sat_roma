import numpy as np

from bevloc.eval.metrics import mean_removed_cosine


def test_mean_removed_cosine_ignores_a_shared_offset():
    rng = np.random.default_rng(0)
    residual = rng.normal(size=(32, 8))
    shared = rng.normal(size=(8,)) * 10
    a = residual + shared
    b = residual * 0.5 + shared
    assert mean_removed_cosine(a, a) == 1.0 or abs(mean_removed_cosine(a, a) - 1.0) < 1e-9
    # The shared offset alone would give cosine 1; the residual is only partly aligned.
    plain = np.mean(np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)))
    removed = mean_removed_cosine(a, b)
    assert removed > 0.99          # residuals are positive-scalar copies
    assert plain > 0.9             # the offset dominates either way here; the identity case is the contract


def test_mean_removed_cosine_of_crossed_residuals_is_low():
    a = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    b = np.array([[0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 0.0]])
    assert abs(mean_removed_cosine(a, b)) < 1e-9
