"""Opt-in, fixed-distance paraxial CRL optics (lengths in micrometres).

No existing CRL, detector or scan code is modified. See docs/chromatic.md.
"""
from dataclasses import dataclass
import numpy as np


def _positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


@dataclass(frozen=True)
class FixedGeometryCRL:
    """Snapshot of a CRL focused at energy0. All distances stay fixed.

    delta(E)=delta0*(energy0/E)**2 is the non-resonant approximation used
    by darkmod.laue.refractive_decrement; suitable for the small Be scan here.
    Rays are specified at the reference object plane: y_out=A*y+B*alpha.
    """
    number_of_lenses: int
    lens_space: float
    lens_radius: float
    delta0: float
    energy0: float
    magnification: float
    d1: float
    d2: float
    theta: float
    eta: float

    def __post_init__(self):
        for name in ('lens_space', 'lens_radius', 'delta0', 'energy0',
                     'magnification', 'd1', 'd2'):
            _positive(getattr(self, name), name)
        if not isinstance(self.number_of_lenses, (int, np.integer)) or self.number_of_lenses < 1:
            raise ValueError('number_of_lenses must be a positive integer')
        if not np.isfinite(self.theta) or not np.isfinite(self.eta):
            raise ValueError('angles must be finite')
        reference = self._matrix(self.energy0)
        if abs(reference[0, 1]) > 1e-7 * max(self.d1, self.d2):
            raise ValueError('reference geometry must be in focus')
        if not np.isclose(reference[0,0],-self.magnification,rtol=1e-9,atol=1e-12):
            raise ValueError('reference magnification does not match the optical geometry')

    @classmethod
    def from_crl(cls, crl, energy0):
        """Copy the focused reference geometry without mutating crl."""
        return cls(crl.N, crl.T, crl.R, crl.delta, energy0,
                   crl.magnification, crl.d1, crl.d2, crl.theta, crl.eta)

    def decrement(self, energy):
        return self.delta0 * (self.energy0 / _positive(energy, 'energy'))**2

    def lens_matrix(self, energy):
        f = self.lens_radius / (2 * self.decrement(energy))
        half = np.array([[1., self.lens_space / 2], [0., 1.]])
        single = half @ np.array([[1., 0.], [-1 / f, 1.]]) @ half
        return np.linalg.matrix_power(single, self.number_of_lenses)

    def _matrix(self, energy):
        return (np.array([[1., self.d2], [0., 1.]]) @ self.lens_matrix(energy)
                @ np.array([[1., self.d1], [0., 1.]]))

    def matrix(self, energy):
        """Object-to-fixed-image matrix; B is in micrometres/radian."""
        matrix = self._matrix(energy)
        if float(energy) == self.energy0:
            # Eliminate numerical cancellation at the exact focus energy.
            matrix[0, 0] = -self.magnification
            matrix[0, 1] = 0.
        return matrix

    def transfer(self, position, angle, energy):
        """Transport object-plane positions and physical ray slopes."""
        a, b = self.matrix(energy)[0]
        return a * np.asarray(position) + b * np.asarray(angle)

    def refocused(self, energy):
        """Ideal control: move conjugate planes to refocus at this energy."""
        from darkmod.crl import CompundRefractiveLens
        c = CompundRefractiveLens(self.number_of_lenses, self.lens_space,
             self.lens_radius, self.decrement(energy), self.magnification)
        c.goto(self.theta, self.eta)
        return type(self).from_crl(c, energy)


def strain_scan_coordinates(strain, energy0, theta0):
    """Bragg-equivalent E and theta for epsilon=(d-d0)/d0.

    Angles in radians, energy in keV. CRL optical axis rotates by 2*theta.
    For the angular scan this radial coordinate assumes the rocking centre
    is tracked; a fixed-rocking CRL scan is not a pure radial reciprocal scan.
    """
    _positive(energy0, 'energy0')
    eps = np.asarray(strain, dtype=float)
    if not np.all(np.isfinite(eps)) or np.any(eps <= -1):
        raise ValueError('strain must be finite and greater than -1')
    if not np.isfinite(theta0) or not 0 < theta0 < np.pi/2:
        raise ValueError('theta0 must lie between 0 and pi/2')
    sine = np.sin(theta0) / (1 + eps)
    if np.any(sine > 1):
        raise ValueError('requested reflection is inaccessible')
    return energy0 / (1 + eps), np.arcsin(sine)
