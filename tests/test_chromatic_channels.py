"""CPU checks of the angular integration used by the CUDA renderer."""

import unittest

import numpy as np

from darkmod import chromatic_imaging as imaging


def channels(*args, **kwargs):
    return list(imaging.angular_channels(*args, **kwargs))


class TestAngularChannels(unittest.TestCase):
    def test_channels_preserve_each_voxel_and_conditional_centroids(self):
        # A single global blur would lose these opposite, voxel-dependent shifts.
        means = np.array([[-0.7, 0.9], [0.2, -0.3]])
        covariance = np.array([[0.04, 0.012], [0.012, 0.09]])
        parts = channels(means, covariance, max_angle_step=0.025, max_channels=30000)
        positions = np.array([p for p, w in parts])
        weights = np.array([w for p, w in parts])
        np.testing.assert_allclose(weights.sum(axis=0), 1.0, atol=1e-13)
        np.testing.assert_allclose(positions.T @ weights, means, atol=0.001)
        for i in range(2):
            delta = positions - means[:, i]
            np.testing.assert_allclose(
                (delta.T * weights[:, i]) @ delta, covariance, atol=0.0002
            )

    def test_deterministic_angles_and_projection_budget(self):
        parts = channels(
            np.array([[0.0, 1.0], [0.0, 0.0]]), np.zeros((2, 2)), max_angle_step=0.1
        )
        np.testing.assert_allclose(sum(w for _, w in parts), 1.0)
        self.assertTrue(all((np.all(w >= 0) for _, w in parts)))
        with self.assertRaisesRegex(ValueError, "channels"):
            channels(
                np.array([[0.0, 100.0], [0.0, 100.0]]),
                np.eye(2),
                max_angle_step=0.001,
                max_channels=10,
            )

    def test_bad_angular_step_rejected_zero(self):
        step = 0.0
        with self.assertRaises(ValueError):
            channels(np.zeros((2, 1)), np.eye(2), max_angle_step=step)

    def test_bad_angular_step_rejected_negative(self):
        step = -1.0
        with self.assertRaises(ValueError):
            channels(np.zeros((2, 1)), np.eye(2), max_angle_step=step)

    def test_bad_angular_step_rejected_nan(self):
        step = np.nan
        with self.assertRaises(ValueError):
            channels(np.zeros((2, 1)), np.eye(2), max_angle_step=step)

    def test_spatial_prior_matches_original_pentagauss_sign(self):
        from darkmod.resolution import PentaGauss

        r = PentaGauss(
            np.array([np.cos(0.45), 0.0, np.sin(0.45)]),
            3e-5,
            2e-5,
            2e-4,
            2e-4,
            0.64575,
            0.64575 * 2.5e-5,
        )
        r.compile()
        shifts = np.array([[1e-4, -2e-4], [-1e-4, 2e-4]])
        q = r.mean_Q_lab[:, None] - r._get_M()[:, 3:] @ shifts
        weight, mean, cov = imaging.angular_posterior(r, q, prior_angles=-shifts)
        np.testing.assert_allclose(weight, r(q, angular_crl_shifts=shifts))
        np.testing.assert_allclose(mean, -shifts, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
