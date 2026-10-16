"""Paraxial CRL optics for energy scans with fixed object and image distances.

The reference geometry is focused at a chosen energy. Distances are in
microns, energies in keV, and angles in radians throughout this module.
"""

from dataclasses import dataclass
import numpy as np


def _positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


@dataclass(frozen=True)
class FixedGeometryCRL:
    """Immutable snapshot of a focused compound refractive lens.

    The refractive decrement follows ``delta(E) = delta0 * (energy0 / E)**2``,
    using the non-resonant approximation in :mod:`darkmod.laue`. Rays are
    specified at the reference object plane, with transverse position
    ``y_image = A * y_object + B * alpha``. The snapshot is immutable so
    changing energy cannot inadvertently change the reference distances.

    Args:
        number_of_lenses (:obj:`int`): Number of lenslets.
        lens_space (:obj:`float`): Lenslet spacing in microns.
        lens_radius (:obj:`float`): Lens curvature radius in microns.
        delta0 (:obj:`float`): Dimensionless refractive decrement at energy0.
        energy0 (:obj:`float`): Reference energy in keV.
        magnification (:obj:`float`): Positive reference magnification.
        d1 (:obj:`float`): Object-to-CRL entrance distance in microns.
        d2 (:obj:`float`): CRL exit-to-image distance in microns.
        theta (:obj:`float`): Reference Bragg angle in radians.
        eta (:obj:`float`): Reference azimuth angle in radians.

    Raises:
        ValueError: If parameters are invalid, the reference is out of focus,
            or the reference magnification disagrees with the ray matrix.
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
        for name in (
            "lens_space",
            "lens_radius",
            "delta0",
            "energy0",
            "magnification",
            "d1",
            "d2",
        ):
            _positive(getattr(self, name), name)
        if (
            not isinstance(self.number_of_lenses, (int, np.integer))
            or self.number_of_lenses < 1
        ):
            raise ValueError("number_of_lenses must be a positive integer")
        if not np.isfinite(self.theta) or not np.isfinite(self.eta):
            raise ValueError("angles must be finite")
        reference = self._matrix(self.energy0)
        if abs(reference[0, 1]) > 1e-7 * max(self.d1, self.d2):
            raise ValueError("reference geometry must be in focus")
        if not np.isclose(reference[0, 0], -self.magnification, rtol=1e-9, atol=1e-12):
            raise ValueError(
                "reference magnification does not match the optical geometry"
            )

    @classmethod
    def from_crl(cls, crl, energy0):
        """Copy the focused reference geometry of an existing CRL.

        Args:
            crl (:obj:`darkmod.crl.CompundRefractiveLens`): Focused lens.
            energy0 (:obj:`float`): Energy in keV at which crl.delta applies.

        Returns:
            FixedGeometryCRL: Independent snapshot of the reference geometry.

        Raises:
            ValueError: If the reference parameters are invalid.
        """
        return cls(
            crl.N,
            crl.T,
            crl.R,
            crl.delta,
            energy0,
            crl.magnification,
            crl.d1,
            crl.d2,
            crl.theta,
            crl.eta,
        )

    def decrement(self, energy):
        """Compute the refractive decrement at a scan energy.

        Args:
            energy (:obj:`float`): Finite positive energy in keV.

        Returns:
            float: Dimensionless refractive decrement.

        Raises:
            ValueError: If energy is nonfinite or nonpositive.
        """
        return self.delta0 * (self.energy0 / _positive(energy, "energy")) ** 2

    def lens_matrix(self, energy):
        """Compute the entrance-to-exit ray matrix of the lens stack.

        Each lenslet consists of propagation over half the lenslet spacing,
        a thin lens, and propagation over the remaining half spacing.

        Args:
            energy (:obj:`float`): Finite positive energy in keV.

        Returns:
            numpy.ndarray: Ray-transfer matrix, shape=(2, 2). It maps
            transverse positions in microns and slopes in radians.

        Raises:
            ValueError: If energy is nonfinite or nonpositive.
        """
        f = self.lens_radius / (2 * self.decrement(energy))
        half = np.array([[1.0, self.lens_space / 2], [0.0, 1.0]])
        single = half @ np.array([[1.0, 0.0], [-1 / f, 1.0]]) @ half
        return np.linalg.matrix_power(single, self.number_of_lenses)

    def _matrix(self, energy):
        return (
            np.array([[1.0, self.d2], [0.0, 1.0]])
            @ self.lens_matrix(energy)
            @ np.array([[1.0, self.d1], [0.0, 1.0]])
        )

    def matrix(self, energy):
        """Compute the ray matrix between the fixed object and image planes.

        Args:
            energy (:obj:`float`): Finite positive energy in keV.

        Returns:
            numpy.ndarray: Ray-transfer matrix K, shape=(2, 2). K[0, 0] is
            the signed transverse magnification A; K[0, 1] is B in microns
            per radian. At energy0 these are exactly -magnification and zero.

        Raises:
            ValueError: If energy is nonfinite or nonpositive.
        """
        matrix = self._matrix(energy)
        if float(energy) == self.energy0:
            # Eliminate numerical cancellation at the exact focus energy.
            matrix[0, 0] = -self.magnification
            matrix[0, 1] = 0.0
        return matrix

    def transfer(self, position, angle, energy):
        """Transport rays from the reference object plane to the image plane.

        Args:
            position (:obj:`numpy array` or :obj:`float`): Transverse object
                positions in microns, broadcastable with angle.
            angle (:obj:`numpy array` or :obj:`float`): Physical ray slopes
                in radians, broadcastable with position.
            energy (:obj:`float`): Finite positive energy in keV.

        Returns:
            numpy.ndarray or numpy.float64: Transverse image positions in
            microns, with the broadcast shape of position and angle.

        Raises:
            ValueError: If energy is invalid or input shapes cannot broadcast.
        """
        a, b = self.matrix(energy)[0]
        return a * np.asarray(position) + b * np.asarray(angle)

    def refocused(self, energy):
        """Construct an ideal refocused control at a scan energy.

        Args:
            energy (:obj:`float`): Finite positive energy in keV.

        Returns:
            FixedGeometryCRL: New snapshot with conjugate distances adjusted
            for focus at energy. Magnification and optical-axis angles are
            retained; the original snapshot is unchanged.

        Raises:
            ValueError: If energy or the resulting reference is invalid.
        """
        from darkmod.crl import CompundRefractiveLens

        c = CompundRefractiveLens(
            self.number_of_lenses,
            self.lens_space,
            self.lens_radius,
            self.decrement(energy),
            self.magnification,
        )
        c.goto(self.theta, self.eta)
        return type(self).from_crl(c, energy)


def strain_scan_coordinates(strain, energy0, theta0):
    """Compute Bragg-equivalent scan coordinates for lattice strain.

    Strain is defined as ``epsilon = (d - d0) / d0``. The CRL optical axis
    rotates by twice theta. An angular scan must track the sample rocking
    centre to follow this radial Bragg trajectory.

    Args:
        strain (:obj:`numpy array` or :obj:`float`): Dimensionless lattice
            strain greater than -1, with arbitrary shape.
        energy0 (:obj:`float`): Finite positive reference energy in keV.
        theta0 (:obj:`float`): Reference Bragg angle in radians, between
            zero and pi/2.

    Returns:
        tuple: Energy in keV and theta in radians, each with the shape of
        strain. Scalar strain produces scalar coordinates.

    Raises:
        ValueError: If inputs are invalid or a reflection is inaccessible.
    """
    _positive(energy0, "energy0")
    eps = np.asarray(strain, dtype=float)
    if not np.all(np.isfinite(eps)) or np.any(eps <= -1):
        raise ValueError("strain must be finite and greater than -1")
    if not np.isfinite(theta0) or not 0 < theta0 < np.pi / 2:
        raise ValueError("theta0 must lie between 0 and pi/2")
    sine = np.sin(theta0) / (1 + eps)
    if np.any(sine > 1):
        raise ValueError("requested reflection is inaccessible")
    return energy0 / (1 + eps), np.arcsin(sine)
