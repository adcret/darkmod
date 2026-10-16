import copy
import unittest
from dataclasses import replace

import numpy as np

from darkmod import laue
from darkmod.crl import CompundRefractiveLens
from darkmod.chromatic import FixedGeometryCRL, strain_scan_coordinates


def lens():
    c = CompundRefractiveLens(
        88, 1600.0, 50.0, laue.refractive_decrement(4, 1.845, 9.0121831, 19.2), 11.68
    )
    c.goto(0.2275, 0.0)
    return c


def independent_matrix(c, energy, d1, d2):
    delta = c.delta * (19.2 / energy) ** 2
    half = np.array([[1.0, c.T / 2], [0.0, 1.0]])
    single = half @ np.array([[1.0, 0.0], [-2 * delta / c.R, 1.0]]) @ half
    return (
        np.array([[1.0, d2], [0.0, 1.0]])
        @ np.linalg.matrix_power(single, c.N)
        @ np.array([[1.0, d1], [0.0, 1.0]])
    )


class TestFixedGeometryCRL(unittest.TestCase):
    def test_reference_focus_and_source_unchanged(self):
        c = lens()
        before = copy.deepcopy(c.__dict__)
        fixed = FixedGeometryCRL.from_crl(c, 19.2)
        for e in (19.1808, 19.2, 19.2192):
            fixed.matrix(e)
        for k in before:
            np.testing.assert_array_equal(c.__dict__[k], before[k])
        self.assertEqual(fixed.matrix(19.2)[0, 1], 0)
        np.testing.assert_allclose(fixed.matrix(19.2)[0, 0], -11.68)
        self.assertTrue(fixed.d1 == c.d1 and fixed.d2 == c.d2)

    def test_fixed_matrix_matches_independent_lenslet_rays_below_reference(self):
        energy = 19.1808
        c = lens()
        fixed = FixedGeometryCRL.from_crl(c, 19.2)
        np.testing.assert_allclose(
            fixed.matrix(energy), independent_matrix(c, energy, c.d1, c.d2), atol=2e-6
        )
        rays = np.array([[0.0, 1.0, -2.0], [1e-4, -3e-4, 2e-4]])
        np.testing.assert_allclose(
            fixed.transfer(rays[0], rays[1], energy),
            (independent_matrix(c, energy, c.d1, c.d2) @ rays)[0],
            atol=1e-8,
        )

    def test_fixed_matrix_matches_independent_lenslet_rays_at_reference(self):
        energy = 19.2
        c = lens()
        fixed = FixedGeometryCRL.from_crl(c, 19.2)
        np.testing.assert_allclose(
            fixed.matrix(energy), independent_matrix(c, energy, c.d1, c.d2), atol=2e-6
        )
        rays = np.array([[0.0, 1.0, -2.0], [1e-4, -3e-4, 2e-4]])
        np.testing.assert_allclose(
            fixed.transfer(rays[0], rays[1], energy),
            (independent_matrix(c, energy, c.d1, c.d2) @ rays)[0],
            atol=1e-8,
        )

    def test_fixed_matrix_matches_independent_lenslet_rays_above_reference(self):
        energy = 19.2192
        c = lens()
        fixed = FixedGeometryCRL.from_crl(c, 19.2)
        np.testing.assert_allclose(
            fixed.matrix(energy), independent_matrix(c, energy, c.d1, c.d2), atol=2e-6
        )
        rays = np.array([[0.0, 1.0, -2.0], [1e-4, -3e-4, 2e-4]])
        np.testing.assert_allclose(
            fixed.transfer(rays[0], rays[1], energy),
            (independent_matrix(c, energy, c.d1, c.d2) @ rays)[0],
            atol=1e-8,
        )

    def test_shift_and_blur_from_ray_ensemble(self):
        f = FixedGeometryCRL.from_crl(lens(), 19.2)
        rng = np.random.default_rng(32)
        alpha = rng.normal(2e-4, 5e-5, 200000)
        p = f.transfer(0.0, alpha, 19.2192)
        b = f.matrix(19.2192)[0, 1]
        self.assertNotEqual(b, 0)
        self.assertLess(abs(p.mean() - b * 0.0002), abs(b) * 5e-07)
        self.assertLess(abs(p.std() - abs(b) * 5e-05), abs(b) * 5e-07)
        np.testing.assert_array_equal(f.transfer(0.0, alpha, 19.2), 0.0)

    def test_scan_coordinates_sign_and_endpoints(self):
        eps = np.linspace(-1e-3, 1e-3, 41)
        energy, theta = strain_scan_coordinates(eps, 19.2, 0.2275)
        np.testing.assert_allclose(19.2 / energy - 1, eps, atol=2e-16)
        np.testing.assert_allclose(np.sin(0.2275) / np.sin(theta) - 1, eps, atol=4e-16)
        self.assertTrue(energy[0] > energy[-1] and theta[0] > theta[-1])

    def test_invalid_energy_zero(self):
        energy = 0
        with self.assertRaises(ValueError):
            FixedGeometryCRL.from_crl(lens(), 19.2).matrix(energy)

    def test_invalid_energy_negative(self):
        energy = -1
        with self.assertRaises(ValueError):
            FixedGeometryCRL.from_crl(lens(), 19.2).matrix(energy)

    def test_invalid_energy_nan(self):
        energy = np.nan
        with self.assertRaises(ValueError):
            FixedGeometryCRL.from_crl(lens(), 19.2).matrix(energy)

    def test_invalid_energy_infinite(self):
        energy = np.inf
        with self.assertRaises(ValueError):
            FixedGeometryCRL.from_crl(lens(), 19.2).matrix(energy)

    def test_snapshot_does_not_follow_source_mutation(self):
        c = lens()
        f = FixedGeometryCRL.from_crl(c, 19.2)
        k = f.matrix(19.21)
        c.refractive_decrement *= 2
        c.goto(0.3, 0)
        np.testing.assert_array_equal(f.matrix(19.21), k)

    def test_inconsistent_reference_magnification_rejected(self):
        fixed = FixedGeometryCRL.from_crl(lens(), 19.2)
        with self.assertRaisesRegex(ValueError, "magnification"):
            replace(fixed, magnification=5.0)


if __name__ == "__main__":
    unittest.main()
