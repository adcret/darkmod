"""Geometry tests run on CPU; image tests require a real ASTRA CUDA device.

DFXM_REQUIRE_GPU=1 makes missing CUDA a failure, not a successful skipped run.
"""
import importlib
import os
import numpy as np
import pytest


@pytest.fixture
def gpu_module():
    assert importlib.util.find_spec('darkmod.chromatic_gpu') is not None, 'ASTRA integration is missing'
    try:
        import astra
    except ImportError:
        if os.environ.get('DFXM_REQUIRE_GPU') == '1':
            pytest.fail('ASTRA is required for cluster validation but could not be imported')
        pytest.skip('ASTRA is not installed')
    return importlib.import_module('darkmod.chromatic_gpu')


@pytest.fixture
def cuda(gpu_module):
    import astra
    if not astra.use_cuda():
        if os.environ.get('DFXM_REQUIRE_GPU') == '1':
            pytest.fail('CUDA is required for cluster validation but ASTRA cannot use it')
        pytest.skip('requires ASTRA with an NVIDIA GPU')
    return gpu_module


def test_virtual_offset_survives_long_detector_distance(gpu_module):
    from darkmod.projector import GpuProjector
    p = gpu_module.OffsetGpuProjector(.5, 32, 32, 1)
    old = GpuProjector(.5, 32, 32, 1)
    corners = np.array([[1e6,1e6,1e6],[-8,8,-8],[-8,-8,8.]])
    axis = np.array([1.,0.,0.])
    baseline = old._get_astra_vectors(2.,axis,corners,1)
    np.testing.assert_array_equal(p._get_astra_vectors(2.,axis,corners,1), baseline)
    p.image_offset = np.array([0.,3.,-4.])
    vectors = p._get_astra_vectors(2.,axis,corners,1)
    np.testing.assert_allclose(vectors[0,3:6]-baseline[0,3:6], [0.,1.5,-2.])
    np.testing.assert_array_equal(vectors[0,6:],baseline[0,6:])


def test_response_uses_original_noise_psf_and_clipping(gpu_module):
    from darkmod.detector import Detector
    # No image projection: exercise the real Detector.readout response on CPU.
    class GivenImage(Detector):
        def render(self, volume, *args):
            return volume.copy()
    corners = np.array([[100.,100.,100.],[-8,8,-8],[-8,-8,8.]])
    kwargs = dict(exposure=10.,psf_width=1.2,noise=True,dynamic_range=64000)
    detector = gpu_module.ChromaticDetector(corners,.5,32,32,**kwargs)
    original = GivenImage(corners,.5,32,32,**kwargs)
    signal = np.arange(1024,dtype=np.float32).reshape(32,32)/10
    np.random.seed(123)
    expected = original.readout(signal,1.,np.array([1.,0,0]),1.)
    np.random.seed(123)
    actual = detector.readout_image(signal)
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(signal,np.arange(1024,dtype=np.float32).reshape(32,32)/10)
    detector.close()


def scene(gpu_module, strain=0.):
    from darkmod import laue
    from darkmod.crystal import Crystal
    from darkmod.crl import CompundRefractiveLens
    from darkmod.chromatic import FixedGeometryCRL
    from darkmod.beam import GaussianLineBeam
    from darkmod.resolution import PentaGauss
    c = Crystal([4.05]*3+[90.]*3,np.eye(3))
    x = (np.arange(16)-7.5)*.1
    f = np.broadcast_to(np.eye(3)*(1+strain),(16,16,16,3,3)).copy()
    c.discretize(*np.meshgrid(x,x,x,indexing='ij'),f)
    hkl = np.array([2,-2,0])
    theta,_ = c.bring_to_bragg(hkl,19.2)
    crl = CompundRefractiveLens(88,1600.,50.,laue.refractive_decrement(4,1.845,9.0121831,19.2),11.68)
    crl.goto(theta,0.)
    fixed = FixedGeometryCRL.from_crl(crl,19.2)
    detector = gpu_module.ChromaticDetector.wall_mount(crl,.65,64,64,
        super_sampling=2,exposure=2.,psf_width=0.,noise=False,dynamic_range=np.inf)
    def resolution(energy):
        wave = laue.keV_to_angstrom(energy)
        r = PentaGauss(crl.optical_axis,3e-5,2e-5,2e-4,2e-4,wave,wave*2.5e-5)
        r.compile()
        return r
    return c,hkl,crl,fixed,detector,GaussianLineBeam(.4,19.2),resolution


def test_no_cpu_image_fallback(gpu_module):
    import astra
    if astra.use_cuda():
        pytest.skip('this check covers a machine without CUDA')
    c,hkl,crl,fixed,d,beam,resolution = scene(gpu_module)
    try:
        with pytest.raises(RuntimeError,match='CUDA'):
            gpu_module.render_chromatic(c,hkl,resolution(19.2),beam,fixed,19.2,d)
    finally:
        d.close()


def test_defocused_spatial_acceptance_rejected_before_rendering(gpu_module):
    c,hkl,crl,fixed,d,beam,resolution = scene(gpu_module)
    beam.energy = 19.21
    try:
        with pytest.raises(NotImplementedError,match='spatial_artefact'):
            gpu_module.render_chromatic(c,hkl,resolution(19.21),beam,fixed,19.21,d,
                                       spatial_artefact=True)
    finally:
        d.close()


def test_wrong_energy_or_moved_detector_rejected(gpu_module):
    c,hkl,crl,fixed,d,beam,resolution = scene(gpu_module)
    try:
        with pytest.raises(ValueError,match='PentaGauss'):
            gpu_module.render_chromatic(c,hkl,resolution(19.2),beam,fixed,19.21,d)
        d.detector_corners[1] += 1.
        with pytest.raises(ValueError,match='reference wall geometry'):
            gpu_module.render_chromatic(c,hkl,resolution(19.2),beam,fixed,19.2,d)
    finally:
        d.close()


@pytest.mark.parametrize('spatial', [False,True])
def test_gpu_reference_matches_existing_diffract(cuda, spatial):
    from darkmod.detector import Detector
    c,hkl,crl,fixed,d,beam,resolution = scene(cuda)
    old = Detector.wall_mount(crl,.65,64,64,super_sampling=2,
        exposure=2.,psf_width=0.,noise=False,dynamic_range=np.inf)
    corners = d.detector_corners.copy()
    try:
        r = resolution(19.2)
        expected = c.diffract(hkl,r,crl,old,beam,spatial_artefact=spatial)
        signal = cuda.render_chromatic(c,hkl,r,beam,fixed,19.2,d,spatial_artefact=spatial)
        actual = d.readout_image(signal)
        assert actual.sum() > 0
        np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=1e-8)
        np.testing.assert_array_equal(d.detector_corners,corners)
        assert d.last_projection_count == 1
    finally:
        d.close()
        if old._projector._vol_id is not None:
            import astra
            astra.data3d.delete(old._projector._vol_id)


def moments(image):
    coords = np.indices(image.shape)
    mean = (coords*image).sum(axis=(1,2))/image.sum()
    var = (((coords-mean[:,None,None])**2)*image).sum(axis=(1,2))/image.sum()
    return mean,var


def test_gpu_signed_beamwalk_and_blur_from_independent_rays(cuda):
    c,hkl,crl,fixed,d,beam,resolution = scene(cuda)
    volume = np.zeros((16,16,16),np.float32)
    volume[6:10,6:10,6:10] = 1.
    # Hand-chosen transport B=10000 um, mean angle (0.00013,-0.00026).
    # On a wall detector, image rows run along -z and columns along -y.
    kwargs = dict(voxel_size=.1,optical_axis=crl.optical_axis,magnification=11.68,
        sample_rotation=np.eye(3),sample_translation=np.zeros(3),imaging_system=crl.imaging_system,b=10000.)
    try:
        reference = d.project_channels([(np.zeros(2),volume)],**kwargs)
        shift = np.array([.00013,-.00026])
        shifted = d.project_channels([(shift,volume)],**kwargs)
        expected = np.array([-10000*shift[1]/(.65*np.cos(2*crl.theta)), -10000*shift[0]/.65])
        np.testing.assert_allclose(moments(shifted)[0]-moments(reference)[0],expected,atol=.12)
        spread = np.array([0.,.00013])
        blurred = d.project_channels([(shift-spread,volume*.5),(shift+spread,volume*.5)],**kwargs)
        np.testing.assert_allclose(moments(blurred)[0],moments(shifted)[0],atol=.12)
        expected_variance = (10000*spread[1]/(.65*np.cos(2*crl.theta)))**2
        assert abs(moments(blurred)[1][0]-moments(shifted)[1][0]-expected_variance) < .7
    finally:
        d.close()


def test_gpu_off_energy_convergence_and_fixed_geometry(cuda):
    c,hkl,crl,fixed,d,beam,resolution = scene(cuda,strain=4e-4)
    energy = 19.2/(1+4e-4)
    beam.energy = energy
    corners = d.detector_corners.copy()
    try:
        a = cuda.render_chromatic(c,hkl,resolution(energy),beam,fixed,energy,d,max_shift_pixels=.2)
        b = cuda.render_chromatic(c,hkl,resolution(energy),beam,fixed,energy,d,max_shift_pixels=.1)
        assert b.sum() > 0 and np.isfinite(b).all()
        assert np.linalg.norm(a-b)/np.linalg.norm(b) < .08
        assert d.last_projection_count > 1
        np.testing.assert_array_equal(d.detector_corners,corners)
        assert fixed.d1 == crl.d1 and fixed.d2 == crl.d2
    finally:
        d.close()
