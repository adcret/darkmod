# batch_simulate_mosa.py
import os, re, argparse, numpy as np
import matplotlib
import scipy.ndimage
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from matplotlib.ticker import FixedLocator
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullFormatter
plt.style.use("seaborn-v0_8-white")

import darling
from darkmod import laue
from darkmod.beam import GaussianLineBeam
from darkmod.crl import CompundRefractiveLens
from darkmod.crystal import Crystal
from darkmod.detector import Detector
from darkmod.resolution import PentaGauss

plt.style.use("seaborn-v0_8-white")
fontsize = 16  # General font size for all text
ticksize = 16  # tick size
plt.rcParams["font.size"] = fontsize
plt.rcParams["xtick.labelsize"] = ticksize
plt.rcParams["ytick.labelsize"] = ticksize
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.dpi"] = 300

def get_roi( diffmax, threshold, pad_row=20, pad_col=20, filter_size=5):
    mask = diffmax > threshold
    mask = scipy.ndimage.median_filter(mask.astype(float), filter_size)
    mask = scipy.ndimage.binary_dilation(mask.astype(float), np.ones((3,3)), iterations=filter_size)
    r, c = np.where(mask)
    r1, r2, c1, c2 = (r.min()-pad_row, r.max()+pad_row+1, c.min()-pad_col, c.max()+pad_col+1)
    return r1, r2, c1, c2, mask

def choose_ticks(vmin_axis, vmax_axis, step=0.05):

    r_min = np.ceil(vmin_axis / step) * step
    r_max = np.floor(vmax_axis / step) * step

    if r_min >= r_max:
        return np.array([vmin_axis, vmax_axis])

    return np.array([r_min, r_max])


def add_color_encoding_legend(
    ax,
    colorgrid,
    colorkey,
    *,
    xlabel=r"$\mathbf{\phi}$",       
    ylabel=r"$\mathbf{\chi}$",        
    size_inches=2.0,                 # Absolute size in inches (Square Box)
    loc=2,
    dx=0.02,
    dy=0.02,
    fontsize=20,
    scale=1e4,
    scale_text=r"$\mathbf{\times 10^{-4}\,rad}$", # Bold math text
    nticks=3,
):
    # 1. Anchor Logic
    anchors = {1: (1, 1), 2: (0, 1), 3: (0, 0), 4: (1, 0)}
    x0, y0 = anchors[loc]
    
    if loc == 1: dx_ = -dx; dy_ = -dy
    elif loc == 2: dx_ = dx;  dy_ = -dy
    elif loc == 3: dx_ = dx;  dy_ = dy
    elif loc == 4: dx_ = -dx; dy_ = dy

    # 2. Create Inset (Physical Square)
    # width=size_inches, height=size_inches guarantees the box is square
    # regardless of the parent image's aspect ratio.
    ax_inset = inset_axes(
        ax,
        width=size_inches,   
        height=size_inches, 
        loc=loc,
        bbox_to_anchor=(x0 + dx_, y0 + dy_),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )

    # 3. Plot Data
    # shading='auto' handles the grid. 
    # We rely on default aspect='auto' so it STRETCHES to fill the square box.
    ax_inset.pcolormesh(*colorgrid, colorkey, shading="auto")
    
    # 4. Remove Margins
    # This ensures the color touches the axis spines exactly.
    ax_inset.margins(0)

    # 5. Styling
    for spine in ax_inset.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")

    # Ticks formatting
    def _fmt(v, pos):
        val = v * scale
        if abs(val) < 1e-9: val = 0
        return f"{val:.1f}"
    
    tickfmt = FuncFormatter(_fmt)
    ax_inset.xaxis.set_major_formatter(tickfmt)
    ax_inset.yaxis.set_major_formatter(tickfmt)
    vals_x = colorgrid[0] * scale
    vals_y = colorgrid[1] * scale
    vmin_x, vmax_x = vals_x.min(), vals_x.max()
    vmin_y, vmax_y = vals_y.min(), vals_y.max()

    xticks = choose_ticks(vmin_x, vmax_x)
    yticks = choose_ticks(vmin_y, vmax_y)

    ax_inset.xaxis.set_major_locator(FixedLocator(xticks / scale))
    ax_inset.yaxis.set_major_locator(FixedLocator(yticks / scale))

    thick_stroke = [pe.withStroke(linewidth=0.1, foreground="black")]


    # 6. Bold Labels & Title
    # Using \mathbf{} inside LaTeX ensures it uses the bold math font.
    ax_inset.set_xlabel(xlabel, fontsize=fontsize, labelpad=5, path_effects=thick_stroke)
    ax_inset.set_ylabel(ylabel, fontsize=fontsize, labelpad=5, path_effects=thick_stroke)

    ax_inset.xaxis.label.set_path_effects(thick_stroke)
    ax_inset.yaxis.label.set_path_effects(thick_stroke)

    # Bold Tick Labels
    # We iterate through ticks to set their weight
    ax_inset.tick_params(axis='both', labelsize=fontsize-14)
    for t in ax_inset.get_xticklabels() + ax_inset.get_yticklabels():
        t.set_fontweight("bold")

    # Bold Top Label
    ax_inset.text(
        0.5, 1.1, 
        scale_text,
        transform=ax_inset.transAxes,
        ha="center", va="bottom",
        fontsize=fontsize -6,
    )

    return ax_inset

thick_stroke = [pe.withStroke(linewidth=0.1, foreground="black")]


bar_height = 6                       # px thickness on screen
bar_color = "white"

fontprops = fm.FontProperties(size=22, weight="bold")

mu_min_data  = -0.0005
mu_max_data  =  0.0005
chi_min_data = -0.0015
chi_max_data =  0.0015

buffer = 1e-4  # small extra margin in radians

mu_min  = mu_min_data  - buffer
mu_max  = mu_max_data  + buffer
chi_min = chi_min_data - buffer
chi_max = chi_max_data + buffer

# keep the same number of steps as before
n_mu  = 40   # was: len(mu_steps)
n_chi = 6   # was: len(chi_steps)

_mu_steps  = np.linspace(mu_min,  mu_max,  n_mu)
_chi_steps = np.linspace(chi_min, chi_max, n_chi)
_mu_grid, _chi_grid = np.meshgrid(_mu_steps, _chi_steps, indexing="ij")

# motors_array with shape (2, n_mu, n_chi)
motors_array = np.stack([_mu_grid, _chi_grid], axis=0)

# ---------- CLI ----------
p = argparse.ArgumentParser()
p.add_argument("--in", dest="in_dir", required=True)
p.add_argument("--out", dest="out_dir", required=True)
p.add_argument("--parts", type=int, default=1)  # total partitions
p.add_argument("--part", type=int, default=1)  # this partition, 1-based
args = p.parse_args()
IN_DIR, OUT_DIR = args.in_dir, args.out_dir
assert 1 <= args.part <= args.parts
os.makedirs(OUT_DIR, exist_ok=True)


# ---------- discover and order files ----------
rx = re.compile(r"defgrad_(\d+)\.npy$")
files = []
for fn in os.listdir(IN_DIR):
    m = rx.fullmatch(fn)
    if m:
        files.append((int(m.group(1)), os.path.join(IN_DIR, fn)))
files.sort(key=lambda t: t[0])

# keep only this partition but retain global indices for correct step numbers
part_files = [
    (k, pth) for i, (k, pth) in enumerate(files) if i % args.parts == (args.part - 1)
]

print("PID", os.getpid(), "parts", args.parts, "part", args.part)
print("LSB_JOBINDEX", os.environ.get("LSB_JOBINDEX"))
print(
    "files:",
    len(files),
    "part_files:",
    len(part_files),
    "first5:",
    [k for k, _ in part_files[:5]],
)


os.makedirs(OUT_DIR, exist_ok=True)

MOSA_DIR = os.path.join(OUT_DIR, "mosa_scans")
os.makedirs(MOSA_DIR, exist_ok=True)

r1, r2, c1, c2 = 94, 226, 64, 256

# ------------------ everything below is your pipeline ------------------
# constants …
voxel_size = 0.0664  # micron
number_of_lenses = 88
lens_space = 1600
lens_radius = 50
magnification = 11.68
energy = 19.2
Z, rho, A = 4, 1.845, 9.0121831
hkl = np.array([2, -2, 0])
eta = np.radians(0)
unit_cell = [3.52, 3.52, 3.52, 90.0, 90.0, 90.0]
H = np.array([[1, -1, 0], [1, 1, 0], [0, 0, 1]]).T
orientation = H / np.linalg.norm(H, axis=0)
z_std = 0.3

beam_FWHM_vertical = 0.027e-3
beam_FWHM_horizontal = 1e-9
sigma_e = (6e-5) / (2 * np.sqrt(2 * np.log(2)))
FWHM_CRL_vertical = 0.556e-3
FWHM_CRL_horizontal = FWHM_CRL_vertical

det_row_count = 320
det_col_count = 320
pixel_size = 0.65
super_sampling = 4
dynamic_range = 2**16 - 1
exposure = 500
psf_width = 1.5
noise = True
spatial_artefact = False

mu_steps = np.linspace(-0.0005, 0.0005, n_mu)
chi_steps = np.linspace(-0.0015, 0.0015, n_chi)

mu_grid, chi_grid = np.meshgrid(mu_steps, chi_steps, indexing="ij")


bar_um = 4.0                        # micrometers
bar_px = bar_um / (pixel_size/11.68)          # ≈ 30.12 px
bar_px = int(round(bar_px)) 

# motors_array with shape (2, n_mu, n_chi)
#motors_array = np.stack([mu_grid, chi_grid], axis=0)


_crystal = Crystal(unit_cell, orientation)
d = np.pi * 2 / np.linalg.norm(_crystal.U @ _crystal.B @ hkl)
theta = np.arcsin(laue.keV_to_angstrom(energy) / (2 * d))
delta = laue.refractive_decrement(Z, rho, A, energy)
crl = CompundRefractiveLens(
    number_of_lenses, lens_space, lens_radius, delta, magnification
)
crl.goto(theta, eta)

lambda_0 = laue.keV_to_angstrom(energy)
beam = GaussianLineBeam(z_std=z_std, energy=energy)

epsilon = np.random.normal(0, sigma_e, size=(20000,))
random_energy = energy + epsilon * energy
sigma_lambda = laue.keV_to_angstrom(random_energy).std()

resolution_function = PentaGauss(
    crl.optical_axis,
    beam_FWHM_horizontal / (2 * np.sqrt(2 * np.log(2))),
    beam_FWHM_vertical / (2 * np.sqrt(2 * np.log(2))),
    FWHM_CRL_horizontal / (2 * np.sqrt(2 * np.log(2))),
    FWHM_CRL_vertical / (2 * np.sqrt(2 * np.log(2))),
    lambda_0,
    sigma_lambda,
)
resolution_function.compile()

detector = Detector.wall_mount(
    crl,
    pixel_size,
    det_row_count,
    det_col_count,
    super_sampling=super_sampling,
    exposure=exposure,
    dynamic_range=dynamic_range,
    psf_width=psf_width,
    noise=noise,
)

median_bck = None

# process this partition
for global_idx, path in part_files:
    try:
        print(f"[part {args.part}] starting {path}", flush=True)
        unit_cell = [4.05, 4.05, 4.05, 90.0, 90.0, 90.0]
        H = np.array([[1, -1, 0], [1, 1, 0], [0, 0, 1]]).T
        orientation = H / np.linalg.norm(H, axis=0)
        crystal = Crystal(unit_cell, orientation)
        d = np.pi * 2 / np.linalg.norm(crystal.U @ crystal.B @ hkl)
        aspect = 1 / np.tan(theta * 2)
        defgrad = np.load(path).transpose((2, 3, 4, 0, 1))

        # grid from defgrad shape (unchanged)
        nx, ny, nz = defgrad.shape[-3:]

        xg = np.arange(nx, dtype=float) * voxel_size
        yg = np.arange(ny, dtype=float) * voxel_size
        zg = np.arange(nz, dtype=float) * voxel_size

        xg -= np.median(xg)
        yg -= np.median(yg)
        zg -= np.median(zg)

        Xgrid, Ygrid, Zgrid = np.meshgrid(xg, yg, zg, indexing="ij")

        Fgrid = np.moveaxis(defgrad, (0, 1), (-2, -1))
        crystal.discretize(Xgrid, Ygrid, Zgrid, Fgrid)

        # NEW: fresh crystal per file
        crystal.align(hkl, axis=np.array([0, 0, 1]))
        crystal.align(
            np.array([1, 1, 0]), axis=np.array([1, 0, 0]), transformation_hkl=hkl
        )
        phi_0 = -theta
        crystal.goniometer.relative_move(dphi=0, dchi=0, domega=0, dmu=phi_0)

        mu0 = crystal.goniometer.mu
        chi0 = crystal.goniometer.chi
        omega0 = crystal.goniometer.omega
        phi0 = crystal.goniometer.phi

        # darks computed once is fine
        if median_bck is None:
            darks = np.zeros((detector.det_col_count, detector.det_row_count, 10))
            crystal.goniometer.goto(
                phi=phi0 + 1, chi=chi0 + 1, omega=omega0 + 1, mu=mu0 + 1
            )
            for i in range(10):
                darks[..., i] = crystal.diffract(
                    hkl,
                    resolution_function,
                    crl,
                    detector,
                    beam,
                    spatial_artefact=spatial_artefact,
                )
            crystal.goniometer.goto(phi=phi0, chi=chi0, omega=omega0, mu=mu0)
            dark_reader = darling.reader.Darks("nowhere")
            darkdset = darling.DataSet(dark_reader)
            darkdset.data = darks.astype(np.uint16)
            median_bck = np.median(darkdset.data, axis=-1)

        mosa = np.zeros(
            (
                detector.det_col_count,
                detector.det_row_count,
                len(mu_steps),
                len(chi_steps),
            )
        )
        for i, dmu in enumerate(mu_steps):
            for j, dchi in enumerate(chi_steps):
                crystal.goniometer.goto(
                    phi=phi0, chi=chi0 + dchi, omega=omega0, mu=mu0 + dmu
                )
                mosa[..., i, j] = crystal.diffract(
                    hkl,
                    resolution_function,
                    crl,
                    detector,
                    beam,
                    spatial_artefact=spatial_artefact,
                )

        # initial intensity mask
        mask = np.sum(mosa, axis=(-1, -2)) > 105 * len(mu_steps) * len(chi_steps)
        masked_mosa = mosa.copy()
        masked_mosa[~mask] = 0

        # mosaics dataset for darling
        reader = darling.reader.MosaScan("nowhere")
        dset = darling.DataSet(reader)
        dset.data = masked_mosa.astype(np.uint16)
        dset.motors = motors_array
        dset.subtract(median_bck[..., None, None].astype(np.uint16))

        # ROI from max projection
        diffmax = dset.data.max(axis=(-1, -2))
        _r1, _r2, _c1, _c2, roi_mask = get_roi(
            diffmax, threshold=220, pad_row=10, pad_col=10, filter_size=9
        )

        # Gaussian mixture features
        feats = darling.properties.gaussian_mixture(
            dset.data,
            k=4,
            coordinates=dset.motors,
        )

        # ---------- 1) RGB map of first component mean (μ, χ) ----------
        first_mean = np.concatenate(
            (
                feats["mean_motor1"][..., 0, None],
                feats["mean_motor2"][..., 0, None],
            ),
            axis=-1,
        )
        first_mean[~mask] = np.nan

        rgbmap1, colorkey, colorgrid = darling.properties.rgb(
            first_mean,
            norm="dynamic",
            coordinates=dset.motors,
        )

        rgb_display = rgbmap1.copy()
        rgb_display[~mask] = np.nan

        fig_rgb, ax_rgb = plt.subplots(1, 1, figsize=(10, 10))
        im_rgb = ax_rgb.imshow(rgb_display[r1:r2, c1:c2], aspect=aspect)

        #add_color_encoding_legend(
        #    ax_rgb,
        #      colorgrid,
        #    colorkey,
        #    fontsize=30,
        #    size_inches=1.3,  # <--- Use this instead of width/height
        #    loc=2,
        #    dx=0.18,
        #    dy=0.15,
        #)
        scalebar = AnchoredSizeBar(
                ax_rgb.transData,
                bar_px,                     # length in data units (pixels)
                r"4 µm",                    # label
                loc='lower center',
                bbox_to_anchor=(0.6, 0.15),   # (x,y) in axes coordinates
                bbox_transform=ax_rgb.transAxes,        # placement
                pad=0.1,
                borderpad=0.1,
                sep=5,
                color='black',
                frameon=False,
                size_vertical=1,            # thickness in pixels (visually consistent)
                fontproperties=fontprops,
            )
        ax_rgb.add_artist(scalebar)

        for spine in ax_rgb.spines.values():
            spine.set_visible(False)
        ax_rgb.set_xticks([])
        ax_rgb.set_yticks([])


        out_rgb = os.path.join(OUT_DIR, f"part{args.part}_k{global_idx}_rgb_no_color_legend.png") # no color legend change this is want image with the square box
        fig_rgb.savefig(out_rgb, dpi=300)
        plt.close(fig_rgb)
        print(f"Saved RGB no color legend: {out_rgb}")

        # ---------- 2) KAM map (log colour scale) ----------

        kernel_size = (3, 9)
        kam = darling.properties.kam(first_mean, size=kernel_size)

        kam_display = kam.copy()
        kam_display[~roi_mask] = np.nan

        # log scale: only strictly positive values
        kam_display = np.where(kam_display <= 0, np.nan, kam_display)

        # FIXED KAM RANGE FOR ALL FIGURES
        vmin_kam = 1e-6
        vmax_kam = 1e-4
        norm_kam = LogNorm(vmin=vmin_kam, vmax=vmax_kam)

        fig_kam, ax_kam = plt.subplots(1, 1, figsize=(10, 10))
        im_kam = ax_kam.imshow(
            kam_display[r1:r2, c1:c2],
            aspect=aspect,
            cmap="plasma",
            norm=norm_kam,
        )
        fig_kam.subplots_adjust(left=0.02, right=0.75, top=0.98, bottom=0.02)

        # thick outline for colorbar label (same style as ρ_GND)
        thick_stroke = [pe.withStroke(linewidth=0.1, foreground="black")]

        cbar_kam = fig_kam.colorbar(im_kam, ax=ax_kam, fraction=0.036, pad=0.01)
        cbar_kam.set_label(
            r"$\mathbf{KAM}$ [rad]",
            fontsize=26,
            path_effects=thick_stroke,
        )

        exps = np.array([-6, -5, -4])
        ticks = 10.0 ** exps

        cbar_kam.set_ticks(ticks)
        cbar_kam.set_ticklabels([rf"$\mathbf{{10^{{{e}}}}}$" for e in exps])

        cbar_kam.ax.tick_params(labelsize=26, which="major", size=6, width=2)
        cbar_kam.ax.tick_params(which="minor", size=3, width=1.5)
        cbar_kam.ax.yaxis.set_minor_formatter(NullFormatter()) # hide minor ticks
        for t in cbar_kam.ax.get_yticklabels():
            t.set_fontweight("bold")

        # scalebar – same style as ρ_GND
        scalebar_kam = AnchoredSizeBar(
            ax_kam.transData,
            bar_px,                      # length in data units (pixels)
            r"4 µm",                     # label
            loc="lower center",
            bbox_to_anchor=(0.6, 0.15),
            bbox_transform=ax_kam.transAxes,
            pad=0.4,
            borderpad=0.5,
            sep=5,
            color="white",
            frameon=False,
            size_vertical=1,
            fontproperties=fontprops,
        )
        ax_kam.add_artist(scalebar_kam)

        for spine in ax_kam.spines.values():
            spine.set_visible(False)
        ax_kam.set_xticks([])
        ax_kam.set_yticks([])

        out_kam = os.path.join(OUT_DIR, f"part{args.part}_k{global_idx}_kam.png")
        fig_kam.savefig(out_kam, dpi=300)
        plt.close(fig_kam)
        print(f"Saved KAM: {out_kam}")

        # ---------- 3) Dislocation density map (GND) ----------
        # sample pixel size from detector pixel size and magnification
        pixel_size_sample_m = (pixel_size / magnification) * 1e-6  # m
        L = kernel_size[1] * pixel_size_sample_m

        # Ni Burgers vector from lattice parameter 3.52 Å (<110>)
        a_ang = 3.52  # Angstrom
        a_m = a_ang * 1e-10
        average_burgers_length = a_m / np.sqrt(2.0)

        # KAM is in mrad -> convert to rad
        kam_rad = kam

        dislocation_density = kam_rad / (average_burgers_length * L)

        rho_display = dislocation_density.copy()
        rho_display[~roi_mask] = np.nan  # same ROI mask as KAM
        rho_display = np.where(rho_display <= 0, np.nan, rho_display)

        # crop to the same ROI window as KAM
        rho_roi = rho_display[r1:r2, c1:c2]

        # --- log scale, FIXED RANGE across all figures ---
        vmin_rho = 1e10
        vmax_rho = 1e13
        norm_rho = LogNorm(vmin=vmin_rho, vmax=vmax_rho)

        fig_rho, ax_rho = plt.subplots(1, 1, figsize=(10, 10))
        im_rho = ax_rho.imshow(
            rho_roi,
            aspect=aspect,
            cmap="plasma",
            norm=norm_rho,
        )

        fig_rho.subplots_adjust(left=0.02, right=0.75, top=0.98, bottom=0.02)   
        cbar_rho = fig_rho.colorbar(im_rho, ax=ax_rho, fraction=0.036, pad=0.01)
        cbar_rho.set_label(
            r"$\boldsymbol{\rho}_\mathrm{GND}$ [m$^{-2}$]",
            fontsize=26,
            path_effects=thick_stroke,
        )

        # pure 10^x ticks, at least 2, bold math
        exps = np.array([10, 11, 12, 13])
        ticks = 10.0 ** exps

        ticks_rho = 10.0**exps
        cbar_rho.set_ticks(ticks_rho)
        cbar_rho.set_ticklabels([rf"$\mathbf{{10^{{{e}}}}}$" for e in exps])

        cbar_rho.ax.tick_params(labelsize=26, which="major", size=6, width=2)
        cbar_rho.ax.tick_params(which="minor", size=3, width=1.5)
        cbar_rho.ax.yaxis.set_minor_formatter(NullFormatter()) # hide minor ticks
        for t in cbar_rho.ax.get_yticklabels():
            t.set_fontweight("bold")

        # scalebar for ρ – same settings as KAM
        scalebar_rho = AnchoredSizeBar(
            ax_rho.transData,
            bar_px,                     # length in data units (pixels)
            r"4 µm",                    # label
            loc="lower center",
            bbox_to_anchor=(0.6, 0.15),
            bbox_transform=ax_rho.transAxes,
            pad=0.4,
            borderpad=0.5,
            sep=5,
            color="white",
            frameon=False,
            size_vertical=1,
            fontproperties=fontprops,
        )
        ax_rho.add_artist(scalebar_rho)

        for spine in ax_rho.spines.values():
            spine.set_visible(False)
        ax_rho.set_xticks([])
        ax_rho.set_yticks([])

        #plt.tight_layout()

        out_rho = os.path.join(OUT_DIR, f"part{args.part}_k{global_idx}_rho.png")
        fig_rho.savefig(out_rho, dpi=300)
        plt.close(fig_rho)
        print(f"Saved ρ: {out_rho}")

    except Exception as e:
        print(f"Failed on {path}: {e}")
