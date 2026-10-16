Fixed-geometry energy scans
===========================

The chromatic extension simulates an energy scan with a CRL and detector fixed
at the positions that focus the microscope at a reference energy. Image
formation uses the existing ASTRA ``FP3D_CUDA`` projector and detector response.
An NVIDIA GPU and a compatible ASTRA installation are required.

The extension consists of three modules:

* :mod:`darkmod.chromatic` stores the reference optics and computes ray transfer.
* :mod:`darkmod.chromatic_imaging` computes conditional outgoing angles and
  integrates their distributions into weighted angular channels.
* :mod:`darkmod.chromatic_gpu` projects those channels and applies the existing
  detector response to their summed image.

Setting up a scan
-----------------

Start with a discretized ``crystal``, Miller indices ``hkl``, and a ``crl``
focused at 19.2 keV. Use the current motor convention: after aligning the
reflection vector to positive lab-z, apply ``dmu=+theta``. The existing
``Crystal.bring_to_bragg`` method performs this alignment.

The following example uses the spectral and angular widths of the CDD study
baseline. They are configuration values, rather than an independent
calibration of the experimental instrument. Lengths are in microns and angles
are in radians.

.. code-block:: python

    import numpy as np

    from darkmod import laue
    from darkmod.beam import GaussianLineBeam
    from darkmod.chromatic import FixedGeometryCRL
    from darkmod.chromatic_gpu import ChromaticDetector, render_chromatic
    from darkmod.resolution import PentaGauss

    energy0 = 19.2
    fixed = FixedGeometryCRL.from_crl(crl, energy0)
    fwhm_to_sigma = 1 / np.sqrt(8 * np.log(2))
    energies = energy0 / (1 + np.linspace(-1e-3, 1e-3, 5))

    with ChromaticDetector.wall_mount(
        crl,
        pixel_size=0.65,
        det_row_count=320,
        det_col_count=320,
        super_sampling=4,
        exposure=500,
        psf_width=1.5,
        noise=False,
        dynamic_range=np.inf,
    ) as detector:
        images = []
        for energy in energies:
            wavelength = laue.keV_to_angstrom(energy)
            resolution = PentaGauss(
                crl.optical_axis,
                1e-9 * fwhm_to_sigma,
                0.027e-3 * fwhm_to_sigma,
                0.556e-3 * fwhm_to_sigma,
                0.556e-3 * fwhm_to_sigma,
                wavelength,
                wavelength * 6e-5 * fwhm_to_sigma,
            )
            resolution.compile()
            beam = GaussianLineBeam(z_std=0.4, energy=energy)
            signal = render_chromatic(
                crystal,
                hkl,
                resolution,
                beam,
                fixed,
                energy,
                detector,
                max_shift_pixels=0.25,
            )
            images.append(detector.readout_image(signal))

Rebuild and compile PentaGauss at each energy, and set the beam to the same
energy. Keep the sample orientation, CRL direction and physical detector
geometry fixed. The context manager releases the projectors' cached volumes
after the scan. Exposure, PSF, noise and clipping are applied once, after
summing all channels in a frame.

Reference optics and controls
-----------------------------

``FixedGeometryCRL.from_crl`` copies the focused object and image distances,
reference decrement, magnification and optical-axis angles. The immutable
snapshot keeps these reference values together throughout the scan.
The decrement changes with energy according to
``delta(E) = delta0 * (energy0 / E)**2``. No initial defocus is introduced.

``fixed.refocused(energy)`` creates an ideal control with conjugate distances
adjusted to focus at that energy while preserving the reference magnification.
For this control, remount the detector to the reconstructed reference CRL:

.. code-block:: python

    from darkmod.chromatic_gpu import reference_crl

    refocused = fixed.refocused(energy)
    detector.remount_to_crl(reference_crl(refocused))

Use the refocused snapshot in ``render_chromatic`` for that control. The
physical fixed-lens scan retains its original snapshot and detector position.
Neither energy mode brings the sample back to the nominal unstrained Bragg
orientation at every energy.

``strain_scan_coordinates`` maps lattice strain ``epsilon = (d - d0) / d0`` to
``E0 / (1 + epsilon)`` and ``asin(sin(theta0) / (1 + epsilon))``. For a theta
scan, track the rocking centre along the radial Bragg trajectory or integrate
over a rocking scan. Changing CRL theta at fixed sample rocking also changes
transverse reciprocal-space sensitivity.

Ray transfer and GPU integration
--------------------------------

Each lenslet consists of free propagation over T/2, a thin lens with
``f = R / (2 * delta)``, and free propagation over T/2. For N lenslets,
``K(E) = P(d2) @ matrix_power(M_lenslet(E), N) @ P(d1)``. The transverse mapping
is ``y_image = A * y_object + B * alpha``. At the reference energy,
``A = -magnification`` and ``B = 0``. Away from focus, B converts outgoing
angles into image displacement and blur.

PentaGauss represents a linear Gaussian relationship between spectral spread,
incident divergence, outgoing angles and each voxel's scattering vector.
Conditioning the outgoing angles on that vector gives a voxel-dependent mean
and shared residual covariance. Gaussian CDFs integrate this distribution into
angular bins. Each bin supplies a weighted scalar volume for ASTRA projection
with transverse scale -A and virtual image displacement B times the bin angle.

The virtual displacement is added after the existing projector's
detector-centre normalization, in magnified voxel units. It does not move the
physical detector. At the reference energy a single ordinary detector render
uses the original diffraction weights. Image inversion, wall obliquity,
supersampling, volume ordering and detector processing follow the existing
package conventions.

Choosing integration resolution
-------------------------------

``max_shift_pixels`` controls angular-bin spacing at the detector; its default
is 0.25 pixels. This parameter is not a bound on total image error. Repeat a
calculation with smaller values and compare images and recovered quantities.
Also check phantom-grid, detector-supersampling and motor-sampling convergence.

Outer bins include the Gaussian probability tails, preserving the sum of
channel probabilities per voxel. ``max_channels`` defaults to 4096 and is a
hard cap: exceeding it raises an error. The calculation never silently relaxes
the requested integration spacing. A chromatic frame can require many ordinary
GPU projections; ``detector.last_projection_count`` records the count.

Scope and limitations
---------------------

* Paraxial, incoherent ray optics with a nominal object plane. Depth-dependent
  focus, wave optics, lens imperfections, absorption and hard apertures are
  outside this model.
* PentaGauss, eta=0, centred square wall detectors and A<0 are supported.
  Small energy excursions in Be away from absorption edges are assumed.
* Spectral bandwidth affects diffraction selection. Optical transfer uses the
  central energy of each exposure; within-exposure chromatic blur is not
  integrated, including at the reference energy.
* ``spatial_artefact=True`` retains the existing behaviour at focus but is
  rejected away from focus. The inherited acceptance-shift signs are not
  consistent with physical horizontal and vertical ray slopes. Energy
  comparisons therefore use ``False``.
* Translations retain the original projector conventions. The supplied study
  baseline uses zero translation.
* Intensities and exposure are arbitrary model units. Reflection structure
  factors and absolute photon counts are not calibrated. KAM is a projected
  angular statistic, rather than total CDD dislocation density or a calibrated
  GND estimate.

For CRL matrix theory see `Simons et al. (2017)
<https://doi.org/10.1107/S160057751602049X>`_. The conditional-angle integration
is derived from the local PentaGauss model and does not reproduce that paper's
full absorption and spectral-aberration model.

Verification
------------

The chromatic tests follow the existing ``unittest.TestCase`` convention and
can be run through either unittest or pytest from the repository root:

.. code-block:: bash

    python -m unittest tests.test_chromatic tests.test_chromatic_imaging tests.test_chromatic_channels tests.test_chromatic_gpu
    python -m pytest tests/test_chromatic.py tests/test_chromatic_imaging.py tests/test_chromatic_channels.py tests/test_chromatic_gpu.py -v

CPU checks cover ray transfer, conditional distributions, channel moments,
virtual projector geometry and the original detector response. CUDA checks
cover focused equivalence, signed beamwalk, blur and off-energy convergence.
GPU tests skip when CUDA is unavailable. On an allocated GPU node, require a
working ASTRA/CUDA installation explicitly:

.. code-block:: bash

    DFXM_REQUIRE_GPU=1 python -m pytest tests/test_chromatic_gpu.py -v

The separate microscope studies folder supplies ``validate_cluster.py`` for
these checks, uniform-strain controls and execution of all eight notebooks.
The existing legacy ``tests/test_transforms.py`` imports the removed
``HighPrecisionRotation`` class; report that collection failure separately.
