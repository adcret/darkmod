"""Conditional-angle integration for opt-in ASTRA chromatic simulations.

PentaGauss supplies diffraction probabilities and the outgoing-angle
distribution conditional on each voxel's Q. Transport y'=A*y+B*alpha gives
both beam walk and blur, without imposing a uniform shift on the phantom.
Image generation is implemented in chromatic_gpu, using ASTRA, not here.
"""

import numpy as np


def angular_channels(
    mean, covariance, max_angle_step, tail_sigma=6.0, max_channels=4096
):
    """Yield (angle centre, per-voxel probability) for Gaussian angle bins.

    Diagonalise the shared conditional covariance, integrate bin probabilities
    using Gaussian CDFs, and project each bin at its centre. Outer bins include
    the tails, so every voxel's channel weights sum to one. The transport error
    is controlled by decreasing max_angle_step; max_channels is a hard budget,
    never a silent reduction of accuracy. Memory is O(number of voxels).

    Args:
        mean (:obj:`numpy array`): Conditional horizontal/vertical outgoing
            slopes in radians for N voxels, shape=(2, N), with N greater than zero.
        covariance (:obj:`numpy array`): Shared symmetric positive semidefinite
            angular covariance in radians squared, shape=(2, 2).
        max_angle_step (:obj:`float`): Finite positive maximum bin width in
            radians along either eigenvector of covariance.
        tail_sigma (:obj:`float`, optional): Gaussian standard deviations
            covered by the finite bin-centre grid. Defaults to 6; must be at
            least 4. Outer bins include the remaining probability tails.
        max_channels (:obj:`int`, optional): Positive maximum bin count.
            Defaults to 4096.

    Yields:
        tuple: Outgoing angle centre in radians, shape=(2,), and nonnegative
        per-voxel bin probabilities, shape=(N,). Probabilities sum to one per
        voxel over all yielded channels.

    Raises:
        ValueError: If inputs are invalid or the bin count exceeds max_channels.
            Validation occurs when the generator is iterated.
    """
    from scipy.special import ndtr

    mean = np.asarray(mean, float)
    covariance = np.asarray(covariance, float)
    if (
        mean.ndim != 2
        or mean.shape[0] != 2
        or mean.shape[1] == 0
        or not np.all(np.isfinite(mean))
    ):
        raise ValueError("mean must be finite with shape (2, N), N > 0")
    if (
        covariance.shape != (2, 2)
        or not np.all(np.isfinite(covariance))
        or not np.allclose(covariance, covariance.T)
    ):
        raise ValueError("covariance must be finite and symmetric")
    if not np.isfinite(max_angle_step) or max_angle_step <= 0:
        raise ValueError("max_angle_step must be positive")
    if not np.isfinite(tail_sigma) or tail_sigma < 4:
        raise ValueError("tail_sigma must be at least 4")
    if not isinstance(max_channels, (int, np.integer)) or max_channels < 1:
        raise ValueError("max_channels must be a positive integer")
    eig, basis = np.linalg.eigh(covariance)
    if eig.min() < -1e-12 * max(np.max(np.abs(eig)), np.finfo(float).tiny):
        raise ValueError("covariance must be positive semidefinite")
    sigma = np.sqrt(np.maximum(eig, 0.0))
    rotated = basis.T @ mean
    axes = []
    count = 1
    for mu, std in zip(rotated, sigma):
        lo, hi = mu.min() - tail_sigma * std, mu.max() + tail_sigma * std
        n = max(1, int(np.ceil((hi - lo) / max_angle_step)))
        count *= n
        if count > max_channels:
            raise ValueError(
                f"angular integration requires more than {max_channels} channels; "
                "increase max_channels or explicitly relax max_shift_pixels"
            )
        edges = np.linspace(lo, hi, n + 1)
        centres = (edges[1:] + edges[:-1]) / 2
        edges[0], edges[-1] = -np.inf, np.inf
        axes.append((centres, edges))

    def probabilities(axis, i):
        _, edges = axes[axis]
        if sigma[axis] == 0:
            return (
                (rotated[axis] >= edges[i]) & (rotated[axis] < edges[i + 1])
            ).astype(float)
        return np.maximum(
            0.0,
            ndtr((edges[i + 1] - rotated[axis]) / sigma[axis])
            - ndtr((edges[i] - rotated[axis]) / sigma[axis]),
        )

    for i, x in enumerate(axes[0][0]):
        wx = probabilities(0, i)
        for j, y in enumerate(axes[1][0]):
            weights = wx * probabilities(1, j)
            if np.any(weights):
                yield basis @ np.array([x, y]), weights


def angular_posterior(resolution, q, prior_angles=None):
    """Return (relative intensity, conditional mean slopes, covariance).

    Angles are (horizontal, vertical), in radians, relative to the nominal
    optical axis. Only the untruncated Gaussian PentaGauss model is supported.
    The spectral spread affects diffraction selection; optical propagation
    is evaluated at the frame's central energy (narrow-band approximation).

    Args:
        resolution (:obj:`darkmod.resolution.PentaGauss`): Compiled Gaussian
            resolution function. Its linear map and latent covariance supply
            the outgoing-angle conditioning model.
        q (:obj:`numpy array`): Finite lab-frame scattering vectors in inverse
            angstroms, shape=(3, N).
        prior_angles (:obj:`numpy array`, optional): Additive means of the
            outgoing-angle prior in radians, shape=(2, 1) or (2, N), in
            horizontal/vertical order. Defaults to zero. These are physical
            slopes, not the legacy CRL acceptance shifts.

    Returns:
        tuple: Relative diffraction intensities, shape=(N,); conditional mean
        slopes in radians, shape=(2, N); and shared conditional angular
        covariance in radians squared, shape=(2, 2).

    Raises:
        TypeError: If resolution is not PentaGauss.
        ValueError: If resolution is uncompiled or array shapes/values are invalid.

    Note:
        PentaGauss exposes no public latent-distribution accessor. Conditioning
        therefore uses its internal linear map, covariance and prior mean here.
    """
    from darkmod.resolution import PentaGauss

    if type(resolution) is not PentaGauss:
        raise TypeError("chromatic imaging requires PentaGauss")
    if resolution._p_Q is None:
        raise ValueError("compile the resolution function first")
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or q.shape[0] != 3 or not np.all(np.isfinite(q)):
        raise ValueError("q must be a finite (3, N) array")
    m = resolution._get_M()
    c = resolution._cov_x
    residual = q - resolution.mean_Q_lab[:, None]
    prior = (
        np.zeros((2, 1)) if prior_angles is None else np.asarray(prior_angles, float)
    )
    if prior.shape not in ((2, 1), (2, q.shape[1])) or not np.all(np.isfinite(prior)):
        raise ValueError("prior_angles must have shape (2, 1) or (2, N)")
    residual = residual - m[:, 3:] @ prior
    s = m @ c @ m.T
    cross = c[3:, :] @ m.T
    solved = np.linalg.solve(s, residual)
    weight = np.exp(-0.5 * np.maximum(0.0, np.sum(residual * solved, axis=0)))
    mean = resolution._mean_x[3:, None] + prior + cross @ solved
    covariance = c[3:, 3:] - cross @ np.linalg.solve(s, cross.T)
    covariance = (covariance + covariance.T) / 2
    eig, vectors = np.linalg.eigh(covariance)
    covariance = (vectors * np.maximum(eig, 0.0)) @ vectors.T
    return weight, mean, covariance
