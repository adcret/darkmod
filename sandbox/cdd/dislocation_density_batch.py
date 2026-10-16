import os
import re
import argparse
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import ndimage as ndi

from darkmod.crystal import Crystal
from darkmod import laue


def rotate_to_lab_and_central_slice(rho_vol, energy_keV=19.2):
    unit_cell = [3.52, 3.52, 3.52, 90.0, 90.0, 90.0]
    H = np.array([[1, -1, 0], [1, 1, 0], [0, 0, 1]]).T
    orientation = H / np.linalg.norm(H, axis=0)
    hkl = np.array([2, -2, 0])

    crystal = Crystal(unit_cell, orientation)
    d = np.pi * 2 / np.linalg.norm(crystal.U @ crystal.B @ hkl)
    theta = np.arcsin(laue.keV_to_angstrom(energy_keV) / (2 * d))

    crystal.align(hkl, axis=np.array([0, 0, 1]))
    crystal.align(np.array([1, 1, 0]), axis=np.array([1, 0, 0]), transformation_hkl=hkl)
    phi_0 = -theta
    crystal.goniometer.relative_move(dphi=0, dchi=0, domega=0, dmu=phi_0)

    R = crystal.goniometer.R
    M = R.T

    nx, ny, nz = rho_vol.shape[:3]
    center = (np.array([nx, ny, nz]) - 1) / 2.0
    offset = center - M @ center

    rho_rot = ndi.affine_transform(
        rho_vol,
        M,
        offset=offset,
        order=1,
        mode="nearest",
    )

    cz = int(center[2])
    rho_slice = rho_rot[:, :, cz]
    return rho_slice, theta


def _process_one(path, out_dir_global, pattern, assume_um2):
    base = os.path.basename(path)
    try:
        rho = np.load(path)

        rho_slice, theta = rotate_to_lab_and_central_slice(rho, energy_keV=19.2)

        # Convert to m^-2 if needed
        if assume_um2:
            rho_slice = rho_slice * 1e12

        vmin, vmax = 1e11, 1e15  # m^-2

        # Avoid zeros for LogNorm by clipping
        rho_slice_clipped = np.clip(rho_slice, vmin, vmax)

        plt.figure(figsize=(8, 7))
        aspect_xy = 1 / np.cos(theta)
        im = plt.imshow(
            rho_slice_clipped,
            origin="lower",
            aspect=aspect_xy,
            norm=LogNorm(vmin=vmin, vmax=vmax),
            cmap="inferno",
        )
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label("dislocation density [m$^{-2}$]")
        plt.axis("off")
        plt.tight_layout()

        out_name = os.path.splitext(base)[0] + "_log.png"
        if out_dir_global:
            out_dir = out_dir_global
        else:
            parent = os.path.dirname(path)
            out_dir = os.path.join(parent, "dils_dens_slices")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, out_name)
        plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close()
        return (path, out_path, None)
    except Exception as e:
        return (path, None, str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument(
        "--out",
        dest="out_dir",
        default=None,
        help="[optional] global output dir; if omitted, saves next to each .npy under dils_dens_slices/",
    )
    ap.add_argument("--pattern", default=r"dislocation_density_(\d+)\.npy")
    ap.add_argument(
        "--assume_um2",
        action="store_true",
        help="Assume input units are µm^-2 (convert to m^-2)",
    )
    ap.add_argument(
        "--workers", type=int, default=6, help="Number of parallel worker processes"
    )
    args = ap.parse_args()

    # out_dir is optional now; when None we will save next to each .npy under dils_dens_slices/
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    rx = re.compile(args.pattern)
    files = []
    for fn in os.listdir(args.in_dir):
        if rx.fullmatch(fn):
            files.append((fn, os.path.join(args.in_dir, fn)))
    files.sort()

    if not files:
        print("No files matched.")
        return

    worker = partial(
        _process_one,
        out_dir_global=args.out_dir,
        pattern=args.pattern,
        assume_um2=args.assume_um2,
    )
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {ex.submit(worker, pth): pth for _, pth in files}
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                src, out, err = fut.result()
                if err is None:
                    print("Saved:", out)
                else:
                    print(f"Failed on {src}: {err}")
            except Exception as e:
                print(f"Failed on {path}: {e}")


if __name__ == "__main__":
    main()
