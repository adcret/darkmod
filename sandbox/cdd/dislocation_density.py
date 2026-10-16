import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from darkmod.crystal import Crystal
from darkmod import laue

# Load dislocation density volume (same grid as defgrad)
dislocation_density = np.load(
    "/dtu/3d-imaging-center/projects/2022_QIM_PMP/analysis/Adam/CDD/Microstructures/dislocation_density_1560.npy"
)

# Rotate volume to the same lab frame as in mosa_batch.py, then take central z-layer
# Replicate crystal setup and goniometer moves
unit_cell = [3.52, 3.52, 3.52, 90.0, 90.0, 90.0]
H = np.array([[1, -1, 0], [1, 1, 0], [0, 0, 1]]).T
orientation = H / np.linalg.norm(H, axis=0)
hkl = np.array([2, -2, 0])
energy = 19.2  # keV

crystal = Crystal(unit_cell, orientation)
d = np.pi * 2 / np.linalg.norm(crystal.U @ crystal.B @ hkl)
theta = np.arcsin(laue.keV_to_angstrom(energy) / (2 * d))
crystal.align(hkl, axis=np.array([0, 0, 1]))
crystal.align(np.array([1, 1, 0]), axis=np.array([1, 0, 0]), transformation_hkl=hkl)
phi_0 = -theta
crystal.goniometer.relative_move(dphi=0, dchi=0, domega=0, dmu=phi_0)

# Get lab rotation matrix and rotate volume around its center
R = crystal.goniometer.R  # maps crystal -> lab
M = R.T  # output(lab) -> input(crystal) for affine_transform

nx, ny, nz = dislocation_density.shape[:3]
center = (np.array([nx, ny, nz]) - 1) / 2.0
offset = center - M @ center

rho_rot = ndi.affine_transform(
    dislocation_density,
    M,
    offset=offset,
    order=1,
    mode="nearest",
)

cz = int(center[2])
rho_slice = rho_rot[:, :, cz]

# Ensure units are m^-2 (assume source is in µm^-2; convert by 1e12)
# If your source is already m^-2, set unit_scale = 1.0 instead.
unit_scale = 1e12  # µm^-2 -> m^-2
rho_slice_si = rho_slice * unit_scale

# Output filenames next to this script (used when running headless)
out_dir = os.path.dirname(__file__)
out_linear = os.path.join(out_dir, "dislocation_density_slice_linear.png")
out_log = os.path.join(out_dir, "dislocation_density_slice_log.png")

# Linear scale image
aspect_xy = 1 / np.cos(theta)  # elongation along x due to rotation about y
plt.figure(figsize=(8, 7))
im = plt.imshow(rho_slice_si, origin="lower", aspect=aspect_xy)
plt.colorbar(im, fraction=0.046, pad=0.04, label="dislocation density [m$^{-2}$]")
plt.title(f"Central z-layer (index {cz}) - linear scale")
plt.axis("off")
plt.tight_layout()
if os.environ.get("DISPLAY") and matplotlib.get_backend().lower() != "agg":
    plt.show()
else:
    plt.savefig(out_linear, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()
# Log scale image (avoid log(0) by adding small epsilon)
positive = rho_slice_si[rho_slice_si > 0]
eps = 1e-20 if positive.size == 0 else max(1e-20, np.percentile(positive, 0.1))
rho_log = np.log10(np.maximum(rho_slice_si, eps))

plt.figure(figsize=(8, 7))
im = plt.imshow(rho_log, origin="lower", aspect=aspect_xy)
plt.colorbar(
    im, fraction=0.046, pad=0.04, label="log10(dislocation density [m$^{-2}$])"
)
plt.title(f"Central z-layer (index {cz}) - log scale")
plt.axis("off")
plt.tight_layout()
if os.environ.get("DISPLAY") and matplotlib.get_backend().lower() != "agg":
    plt.show()
else:
    plt.savefig(out_log, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()
