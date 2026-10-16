import copy
from dataclasses import replace

import numpy as np
import pytest

from darkmod import laue
from darkmod.crl import CompundRefractiveLens
from darkmod.chromatic import FixedGeometryCRL, strain_scan_coordinates


def lens():
    c = CompundRefractiveLens(88, 1600., 50.,
        laue.refractive_decrement(4, 1.845, 9.0121831, 19.2), 11.68)
    c.goto(0.2275, 0.)
    return c


def independent_matrix(c, energy, d1, d2):
    delta = c.delta * (19.2 / energy)**2
    half = np.array([[1., c.T / 2], [0., 1.]])
    single = half @ np.array([[1., 0.], [-2 * delta / c.R, 1.]]) @ half
    return np.array([[1., d2], [0., 1.]]) @ np.linalg.matrix_power(single, c.N) @ np.array([[1., d1], [0., 1.]])


def test_reference_focus_and_source_unchanged():
    c = lens()
    before = copy.deepcopy(c.__dict__)
    fixed = FixedGeometryCRL.from_crl(c, 19.2)
    for e in (19.1808, 19.2, 19.2192):
        fixed.matrix(e)
    for k in before:
        np.testing.assert_array_equal(c.__dict__[k], before[k])
    assert fixed.matrix(19.2)[0, 1] == 0
    np.testing.assert_allclose(fixed.matrix(19.2)[0, 0], -11.68)
    assert fixed.d1 == c.d1 and fixed.d2 == c.d2


@pytest.mark.parametrize('energy', [19.1808, 19.2, 19.2192])
def test_fixed_matrix_matches_independent_lenslet_rays(energy):
    c = lens()
    fixed = FixedGeometryCRL.from_crl(c, 19.2)
    np.testing.assert_allclose(fixed.matrix(energy), independent_matrix(c, energy, c.d1, c.d2), atol=2e-6)
    rays = np.array([[0., 1., -2.], [1e-4, -3e-4, 2e-4]])
    np.testing.assert_allclose(fixed.transfer(rays[0], rays[1], energy),
                               (independent_matrix(c, energy, c.d1, c.d2) @ rays)[0], atol=1e-8)


def test_shift_and_blur_from_ray_ensemble():
    f = FixedGeometryCRL.from_crl(lens(), 19.2)
    rng = np.random.default_rng(32)
    alpha = rng.normal(2e-4, 5e-5, 200000)
    p = f.transfer(0., alpha, 19.2192)
    b = f.matrix(19.2192)[0, 1]
    assert b != 0
    assert abs(p.mean() - b * 2e-4) < abs(b) * 5e-7
    assert abs(p.std() - abs(b) * 5e-5) < abs(b) * 5e-7
    np.testing.assert_array_equal(f.transfer(0., alpha, 19.2), 0.)


def test_scan_coordinates_sign_and_endpoints():
    eps = np.linspace(-1e-3, 1e-3, 41)
    energy, theta = strain_scan_coordinates(eps, 19.2, 0.2275)
    np.testing.assert_allclose(19.2 / energy - 1, eps, atol=2e-16)
    np.testing.assert_allclose(np.sin(.2275) / np.sin(theta) - 1, eps, atol=4e-16)
    assert energy[0] > energy[-1] and theta[0] > theta[-1]


@pytest.mark.parametrize('energy', [0, -1, np.nan, np.inf])
def test_invalid_energy(energy):
    with pytest.raises(ValueError):
        FixedGeometryCRL.from_crl(lens(), 19.2).matrix(energy)


def test_snapshot_does_not_follow_source_mutation():
    c = lens()
    f = FixedGeometryCRL.from_crl(c, 19.2)
    k = f.matrix(19.21)
    c.refractive_decrement *= 2
    c.goto(.3, 0)
    np.testing.assert_array_equal(f.matrix(19.21), k)


def test_inconsistent_reference_magnification_rejected():
    fixed = FixedGeometryCRL.from_crl(lens(),19.2)
    with pytest.raises(ValueError,match='magnification'):
        replace(fixed,magnification=5.)
