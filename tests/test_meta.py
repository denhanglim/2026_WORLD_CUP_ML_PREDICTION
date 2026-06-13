import numpy as np
from wc2026.models.meta import MetaStacker
from wc2026.eval.scoring import rps

def _good_and_noise(n=1500, seed=2):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    # good base: high prob on the true class
    good = np.full((n, 3), 0.1); good[np.arange(n), y] = 0.8
    good = good / good.sum(axis=1, keepdims=True)
    # noise base: random simplex
    noise = rng.dirichlet([1, 1, 1], size=n)
    return good, noise, y

def test_meta_outputs_valid_probs():
    good, noise, y = _good_and_noise()
    meta = MetaStacker().fit([good, noise], y)
    P = meta.predict_proba([good, noise])
    assert P.shape == (len(y), 3)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)

def test_meta_ignores_noise_base():
    good, noise, y = _good_and_noise()
    tr = slice(0, 1000); te = slice(1000, None)
    meta = MetaStacker().fit([good[tr], noise[tr]], y[tr])
    P = meta.predict_proba([good[te], noise[te]])
    rps_meta = np.mean(rps(P, y[te]))
    rps_good = np.mean(rps(good[te], y[te]))
    rps_noise = np.mean(rps(noise[te], y[te]))
    assert rps_meta < rps_noise                 # learned to discard noise
    assert rps_meta <= rps_good + 0.02           # nearly as good as the good base
