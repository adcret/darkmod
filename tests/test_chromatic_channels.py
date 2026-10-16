"""CPU checks of the angular integration used by the CUDA renderer."""
import numpy as np
import pytest
from darkmod import chromatic_imaging as imaging


def channels(*args, **kwargs):
    assert hasattr(imaging, 'angular_channels'), 'angular integration is missing'
    return list(imaging.angular_channels(*args, **kwargs))


def test_channels_preserve_each_voxel_and_conditional_centroids():
    # A single global blur would lose these opposite, voxel-dependent shifts.
    means = np.array([[-.7, .9], [.2, -.3]])
    covariance = np.array([[.04, .012], [.012, .09]])
    parts = channels(means, covariance, max_angle_step=.025, max_channels=30000)
    positions = np.array([p for p, w in parts])
    weights = np.array([w for p, w in parts])
    np.testing.assert_allclose(weights.sum(axis=0), 1., atol=1e-13)
    np.testing.assert_allclose(positions.T @ weights, means, atol=.001)
    for i in range(2):
        delta = positions - means[:, i]
        np.testing.assert_allclose((delta.T*weights[:, i]) @ delta,
                                   covariance, atol=.0002)


def test_deterministic_angles_and_projection_budget():
    parts = channels(np.array([[0., 1.], [0., 0.]]), np.zeros((2,2)),
                     max_angle_step=.1)
    np.testing.assert_allclose(sum(w for _, w in parts), 1.)
    assert all(np.all(w >= 0) for _, w in parts)
    with pytest.raises(ValueError, match='channels'):
        channels(np.array([[0., 100.], [0., 100.]]), np.eye(2),
                 max_angle_step=.001, max_channels=10)


@pytest.mark.parametrize('step', [0., -1., np.nan])
def test_bad_angular_step_rejected(step):
    with pytest.raises(ValueError):
        channels(np.zeros((2,1)), np.eye(2), max_angle_step=step)


def test_spatial_prior_matches_original_pentagauss_sign():
    from darkmod.resolution import PentaGauss
    r = PentaGauss(np.array([np.cos(.45),0.,np.sin(.45)]),
                  3e-5, 2e-5, 2e-4, 2e-4, .64575, .64575*2.5e-5)
    r.compile()
    shifts = np.array([[1e-4, -2e-4], [-1e-4, 2e-4]])
    q = r.mean_Q_lab[:, None] - r._get_M()[:, 3:] @ shifts
    weight, mean, cov = imaging.angular_posterior(r, q, prior_angles=-shifts)
    np.testing.assert_allclose(weight, r(q, angular_crl_shifts=shifts))
    np.testing.assert_allclose(mean, -shifts, atol=1e-15)
