import numpy as np
import darling
import os



OUT_DIR = "/dtu/3d-imaging-center/projects/2022_QIM_PMP/analysis/Adam/CDD/Mosa"

MOSA_DIR = os.path.join(OUT_DIR, "mosa_scans")

motors = np.load(os.path.join(MOSA_DIR, "motors.npy"))

mosa_path = os.path.join(MOSA_DIR, f"mosa_01560.npy")
mosa = np.load(mosa_path)

reader = darling.reader.MosaScan("nowhere")
dset = darling.DataSet(reader)
dset.data = mosa.astype(np.uint16)
dset.motors = motors

feats = darling.properties.gaussian_mixture(
    dset.data, k=4, coordinates=dset.motors
)
first_max = np.concatenate(
    (feats["mean_motor1"][..., 0, None], feats["mean_motor2"][..., 0, None]),
    axis=-1,
)

rgbmap1, colorkey, colorgrid = darling.properties.rgb(
    first_max, norm="dynamic", coordinates=dset.motors
)

print(f"the maxminum value for the first component first motor is {first_max[:,:,0].max()}")
print(f"the maxminum value for the first component second motor is {first_max[:,:,1].max()}")
print(f"the minimum value for the first component first motor is {first_max[:,:,0].min()}")
print(f"the minimum value for the first component second motor is {first_max[:,:,1].min()}")