# Fixed-geometry energy scans with ASTRA

Import explicitly from `darkmod.chromatic` and `darkmod.chromatic_gpu`.
Existing classes, defaults and scan helpers are unchanged. All image
projections use ASTRA's FP3D_CUDA, with no CPU image-rendering fallback.
Use an allocated NVIDIA GPU node with a compatible ASTRA/CUDA installation.

```python
from darkmod.chromatic import FixedGeometryCRL
from darkmod.chromatic_gpu import ChromaticDetector, render_chromatic

fixed = FixedGeometryCRL.from_crl(crl, energy0=19.2)
# Rebuild/compile PentaGauss AND update beam.energy for every frame energy.
with ChromaticDetector.wall_mount(
    crl, pixel_size=0.65, det_row_count=320, det_col_count=320,
    super_sampling=4, exposure=500, psf_width=1.5,
    noise=False, dynamic_range=float('inf'),
) as detector:
    signal = render_chromatic(crystal, hkl, resolution, beam, fixed, 19.21,
                              detector, max_shift_pixels=0.25)
    image = detector.readout_image(signal)  # original response, once
    print(detector.last_projection_count)
```

The snapshot captures the current focused object/image distances. It does not
modify the original CRL. The lens and detector remain fixed during an energy
scan, with delta(E)=delta0*(E0/E)^2. No initial d1/d2 perturbation is introduced.
`fixed.refocused(E)` is the ideal control: its conjugate distances are moved
to keep the original magnification and achieve focus at the frame energy.
Sample orientation is NOT brought back to nominal Bragg at every energy.
For the ideal control only, remount the detector to `reference_crl(refocused)`.
Use the newer motor convention: after aligning Q to +lab-z, `dmu=+theta`,
as in current `bring_to_bragg`. This extension never changes motor conventions.

## Optical model

For each lenslet use free propagation T/2, a thin lens of f=R/(2*delta), then
free propagation T/2. Multiply N lenslets, then the fixed d1/d2 propagation:
`K(E) = P(d2) @ M_lenslet(E)**N @ P(d1)` (matrix power).
The physical ray position is `y_image = A*y_object + B*alpha_object`.
At E0, A=-magnification and B=0; away from E0, B gives defocus and beam walk.
An imposed mean ray tilt alpha yields shift B*alpha, as in the supplied
`Al1050_red_ccmth_shift_theory.ipynb`, but no deliberate reference defocus.

The new renderer retains Darkmod's PentaGauss variables x, with Q=Mx. For a
given voxel Q, the outgoing slopes have conditional mean
`mu_alpha + C_alpha,Q @ inv(C_Q) @ (Q-mu_Q)` and covariance
`C_alpha - C_alpha,Q @ inv(C_Q) @ C_Q,alpha`. The conditional means give
voxel-dependent image shifts and the covariance gives residual spot widths.
Using the full CRL angular width as a post-image Gaussian would generally
overestimate this conditional blur.

Conditional angles are integrated into bins using Gaussian CDFs. Each bin
weights a scalar voxel volume; ASTRA projects it with magnification -A and
virtual image displacement B*alpha. Their sum retains the local strain/angle
relationship. The original image inversion, wall obliquity, supersampling and
voxel ordering are retained. Detector response follows the sum, once.

The virtual displacement is added AFTER the legacy detector-centre
normalisation, in magnified voxel units, so metre-scale distances do not
suppress beamwalk. It does not move the physical detector. At E0 a single
ordinary `Detector.render` call uses the unchanged original weights.
`readout_image` delegates to original `Detector.readout`; there is no separate
PSF/noise model. None of the original source files is modified.

`max_shift_pixels` (default 0.25) controls angular-bin spacing at the detector,
not a bound on total image error. Repeat with half that value; also test voxel,
detector supersampling and scan sampling convergence. Outer bins include tails,
preserving per-voxel probability. `max_channels` (default 4096) is a hard cap:
exceeding it raises an error, never a silent accuracy reduction. Chromatic
frames can therefore cost many ordinary GPU projections.

## Scope and limitations

- Paraxial, incoherent ray optics and nominal object plane, as in the
  original CRL projection. It does not resolve depth-dependent defocus.
- PentaGauss only, eta=0 centred square wall detector, A<0, small energy excursions in Be away
  from absorption edges. No aperture truncation, absorption, wave optics,
  lens imperfections, or calibrated reflection structure factors.
- Off-energy `spatial_artefact=True` is explicitly unsupported: the inherited
  off-axis acceptance Euler signs are inconsistent with physical horizontal
  and vertical ray slopes. Focused scans retain the original behaviour.
  Keep this option off for energy comparisons. Sample translations likewise
  retain legacy conventions; the supplied baseline uses zero translation.
- Energy bandwidth enters reciprocal-space selection. Optical transfer uses
  the CENTRAL energy of each exposure. Finite-bandwidth chromatic blur even at
  E0 is not included. This is a narrow-band approximation; widening bandwidth
  enough to affect real-space focus requires joint spectral ray integration.
- Baseline exposure is arbitrary model gain, not seconds or dose. Do not infer
  absolute counts, acquisition time, or relative reflection efficiency.
- KAM is a projected angular statistic and is not the CDD total dislocation
  density. No KAM-to-GND calibration is assumed.

The scan helper maps epsilon=(d-d0)/d0 to E0/(1+epsilon) or
asin(sin(theta0)/(1+epsilon)). For a theta scan the sample rocking centre must
track the radial Bragg condition, or be integrated over a rocking scan; merely
changing the CRL angle at fixed rocking also changes transverse Q sensitivity.

Reference: [Simons et al. (2017), CRL ray-transfer formalism](https://doi.org/10.1107/S160057751602049X).
The implemented conditional-angle transport is derived from the local
PentaGauss linear Gaussian model, not a claimed reproduction of the paper's
full absorption and spectral aberration model.

## Verification

```bash
python -m pytest tests/test_chromatic.py tests/test_chromatic_imaging.py tests/test_chromatic_channels.py tests/test_chromatic_gpu.py -v
```

CPU checks cover lenslet rays, focus, strain coordinates, conditional Gaussians,
channel probabilities/moments, virtual ASTRA geometry, original detector
response and invalid inputs. Real CUDA tests check reference equivalence with
`Crystal.diffract`, signed beamwalk, blur and off-energy convergence. They skip
without CUDA. Make missing CUDA/ASTRA fail on the cluster:

```bash
DFXM_REQUIRE_GPU=1 python -m pytest tests/test_chromatic_gpu.py -v
```

The separate study folder supplies `validate_cluster.py` for these tests plus
uniform-strain controls and optionally all eight notebooks. GPU image checks
have not run on the development PC (no GPU). Full legacy pytest collection
also encounters the pre-existing `tests/test_transforms.py` import of the removed
`HighPrecisionRotation`; this unrelated test is not changed or hidden here.
