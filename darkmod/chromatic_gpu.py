"""Opt-in fixed-lens chromatic imaging through Darkmod's ASTRA CUDA projector.

Existing Detector, Crystal and GpuProjector implementations are unchanged.
Only angle-conditioned scalar volumes are prepared on CPU. All image
projections use FP3D_CUDA; there is deliberately no CPU rendering fallback.
"""
from dataclasses import dataclass
import astra
import numpy as np

from darkmod.crl import CompundRefractiveLens
from darkmod.detector import Detector, _get_det_corners_wall_mount
from darkmod.projector import GpuProjector
from darkmod.chromatic_imaging import angular_posterior, angular_channels


def require_cuda():
    if not astra.use_cuda():
        raise RuntimeError('Chromatic image generation requires ASTRA with CUDA on an NVIDIA GPU; '
                           'run on a GPU node (there is no CPU image fallback).')


def reference_crl(optics):
    """Reconstruct the snapshot's reference CRL without changing its focus."""
    crl = CompundRefractiveLens(optics.number_of_lenses, optics.lens_space,
        optics.lens_radius, optics.delta0, optics.magnification)
    crl.goto(optics.theta, optics.eta)
    return crl


class OffsetGpuProjector(GpuProjector):
    """The existing CUDA projector plus an explicit virtual image displacement.

    The legacy geometry normalises its detector-centre vector. Add the
    transverse shift AFTER that normalisation, in ASTRA voxel units, so small
    beamwalk is not divided by the metre-scale sample-detector distance.
    image_offset is in the sample frame, in micrometres; it never moves the
    physical detector. Zero offset gives the original vector geometry.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_offset = np.zeros(3)

    def _get_astra_vectors(self, voxel_size, ray_direction, detector_corners, super_sampling):
        vectors = super()._get_astra_vectors(voxel_size,ray_direction,detector_corners,super_sampling)
        vectors[0,3:6] += self.image_offset / voxel_size
        return vectors


@dataclass
class _IntegratedImage:
    signal: np.ndarray


class ChromaticDetector(Detector):
    """Ordinary Detector plus chromatic projection and integrated-image readout.

    Use wall_mount exactly like Detector. Its ordinary render path remains
    the inherited one. Call close() (or use a context manager) to release
    the cached ASTRA volumes after a scan, including on failure.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._offset_projector = OffsetGpuProjector(self.pixel_size,
            self.det_row_count,self.det_col_count,self.super_sampling)
        self.last_projection_count = 0

    def render(self, voxel_volume, *args, **kwargs):
        if isinstance(voxel_volume, _IntegratedImage):
            return voxel_volume.signal.copy()
        return super().render(voxel_volume,*args,**kwargs)

    def readout_image(self, signal):
        """Apply the ORIGINAL Detector.readout response once to a summed image."""
        signal = np.asarray(signal)
        if signal.shape != (self.det_row_count,self.det_col_count) or not np.all(np.isfinite(signal)) or np.any(signal < 0):
            raise ValueError('signal must be a finite, nonnegative detector-shaped image')
        return super().readout(_IntegratedImage(signal.astype(np.float32,copy=False)),
                               1.,np.array([1.,0.,0.]),1.)

    def project_channels(self, channels, voxel_size, optical_axis, magnification,
                         sample_rotation, sample_translation, imaging_system, b):
        """Project (angle, ASTRA-order weighted volume) channels and sum.

        ASTRA uses the existing nominal-axis/nominal-object-plane model. The
        virtual image displacement is B*angle; transverse magnification is
        -A. No post-image interpolation or global-blur approximation is used.
        """
        require_cuda()
        if self.det_row_count != self.det_col_count:
            raise ValueError('chromatic integration currently requires a square detector')
        if not np.isfinite(magnification) or magnification <= 0:
            raise ValueError('chromatic integration requires A < 0')
        corners = sample_rotation.T @ self.detector_corners - sample_translation
        direction = sample_rotation.T @ optical_axis
        image = np.zeros((self.det_row_count,self.det_col_count),np.float64)
        self.last_projection_count = 0
        for angles, volume in channels:
            shift_lab = imaging_system[:,1:] @ (b*np.asarray(angles))
            self._offset_projector.image_offset = sample_rotation.T @ shift_lab
            partial = self._offset_projector(volume,voxel_size*magnification,direction,corners)
            image += self._invert(partial)
            self.last_projection_count += 1
        return image.astype(np.float32)

    def close(self):
        for projector in (self._projector,self._offset_projector):
            if projector._vol_id is not None:
                astra.data3d.delete(projector._vol_id)
                projector._vol_id = None
                projector._vol_shape_key = None
                projector._vol_geom = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def render_chromatic(crystal, hkl, resolution, beam, optics, energy, detector,
                     spatial_artefact=False, max_shift_pixels=.25, max_channels=4096):
    """Noiseless ASTRA image at E, with lens and detector fixed at E0 focus.

    Rebuild/compile PentaGauss for energy before calling. Use readout_image()
    once afterwards for exposure, PSF, noise and clipping. Decrease
    max_shift_pixels to check angular-channel convergence. At reference
    energy there is one ordinary Detector.render call, with unchanged weights.

    Supported: PentaGauss, eta=0, centred wall-mounted square detectors, A<0.
    Finite bandwidth affects diffraction selection, but the optical transfer
    is evaluated at the frame's central energy (narrow-band approximation).
    """
    from darkmod import laue
    from darkmod.resolution import PentaGauss
    if not isinstance(detector,ChromaticDetector):
        raise TypeError('use ChromaticDetector.wall_mount for chromatic scans')
    if type(resolution) is not PentaGauss:
        raise TypeError('chromatic imaging requires PentaGauss')
    if not np.isfinite(max_shift_pixels) or max_shift_pixels <= 0:
        raise ValueError('max_shift_pixels must be positive')
    if abs(optics.eta) > 1e-12:
        raise ValueError('chromatic wall integration currently supports eta=0 only')
    crl = reference_crl(optics)
    if not np.allclose(resolution.optical_axis_0,crl.optical_axis,rtol=0,atol=1e-12):
        raise ValueError('resolution and reference optics must share their axis')
    if not np.isclose(resolution._mean_wavelength,laue.keV_to_angstrom(energy),rtol=1e-10,atol=0):
        raise ValueError('rebuild PentaGauss for the frame energy')
    if not np.isclose(beam.energy,energy,rtol=1e-10,atol=0):
        raise ValueError('beam.energy must match the frame energy')
    expected = _get_det_corners_wall_mount(crl,detector.pixel_size,detector.det_row_count,detector.det_col_count)
    if not detector._wall_mount or not np.allclose(detector.detector_corners,expected,rtol=0,atol=1e-7):
        raise ValueError('detector must remain at the reference wall geometry')
    if detector.det_row_count != detector.det_col_count:
        raise ValueError('chromatic integration currently requires a square detector')
    a,b = optics.matrix(energy)[0]
    if a >= 0:
        raise ValueError('chromatic integration requires A < 0')
    if b != 0. and spatial_artefact:
        raise NotImplementedError('spatial_artefact=True is not supported away from focus: '
            'the legacy acceptance shifts have inconsistent physical angle signs. '
            'Use spatial_artefact=False for chromatic energy scans.')
    cos = np.cos(2*optics.theta)
    if cos <= 1e-6:
        raise ValueError('wall detector is grazing or behind the ray bundle')
    require_cuda()
    detector.last_projection_count = 0
    xyz = crystal._get_x_lab_flat()
    shifts = crl.get_angular_shifts(xyz) if spatial_artefact else None
    q = crystal._get_Q_lab_flat(hkl)
    p = resolution(q,angular_crl_shifts=shifts)
    illumination = beam(xyz)
    if b == 0.:
        volume = crystal._get_voxel_volume_astra_order(p,illumination)
        signal = detector.render(volume,crystal.voxel_size,crl.optical_axis,
            optics.magnification,crystal.goniometer.R,crystal.goniometer.u)
        detector.last_projection_count = 1
        return signal

    active = (p > 0) & (illumination > 0) & (crystal.density.ravel() > 0)
    if not np.any(active):
        return np.zeros((detector.det_row_count,detector.det_col_count),np.float32)
    # Off-energy spatial acceptance is rejected above; no geometric prior.
    _,means,covariance = angular_posterior(resolution,q[:,active])
    # Two eigen-axis bin errors combine in quadrature; include wall obliquity.
    angle_step = max_shift_pixels*detector.pixel_size*cos/(np.sqrt(2)*abs(b))

    def volumes():
        for angle, fractions in angular_channels(means,covariance,angle_step,max_channels=max_channels):
            channel_p = np.zeros_like(p)
            channel_p[active] = p[active]*fractions
            yield angle,crystal._get_voxel_volume_astra_order(channel_p,illumination)

    return detector.project_channels(volumes(),crystal.voxel_size,crl.optical_axis,
        -a,crystal.goniometer.R,crystal.goniometer.u,crl.imaging_system,b)
