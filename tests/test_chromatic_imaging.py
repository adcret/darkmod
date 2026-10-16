import numpy as np
import pytest
from darkmod.resolution import PentaGauss
from darkmod.chromatic_imaging import angular_posterior


def resolution():
    t = .2275
    r = PentaGauss(np.array([np.cos(2*t), 0., np.sin(2*t)]),
                  3e-5, 2e-5, 2e-4, 2e-4, .64575, .64575*2.5e-5)
    r.compile()
    return r


def test_conditional_angles_and_intensity():
    r = resolution()
    q = r.mean_Q_lab[:, None] + np.array([[0., 1e-4], [0., -2e-4], [0., 1e-4]])
    weights, mean, cov = angular_posterior(r, q)
    np.testing.assert_allclose(weights, r(q))
    np.testing.assert_allclose(mean[:, 0], 0., atol=1e-15)
    assert np.linalg.eigvalsh(cov).min() > -1e-16
    assert np.all(np.diag(cov) <= np.diag(r._cov_x)[3:])
    # Independent conditional joint-Gaussian formula.
    cross = r._cov_x[3:, :] @ r._get_M().T
    np.testing.assert_allclose(mean, cross @ np.linalg.solve(r.cov_Q_lab, q-r.mean_Q_lab[:, None]))


def test_prior_mean_is_retained_in_physical_ray_angles():
    r = resolution()
    prior = np.array([[1e-4], [-2e-4]])
    q = r.mean_Q_lab[:, None] + r._get_M()[:, 3:] @ prior
    w, mean, cov = angular_posterior(r, q, prior)
    np.testing.assert_allclose(w, 1.)
    np.testing.assert_allclose(mean, prior)
