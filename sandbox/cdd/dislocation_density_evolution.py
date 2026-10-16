import numpy as np
from matplotlib import pyplot as plt

data = np.load("dislocation_density_total.npy")

print(data.shape)
print(data)

plt.plot(data*1e12)
plt.xlabel("Strain index")
plt.ylabel("Dislocation density [m$^{-2}$]")
plt.title("Dislocation density evolution")
plt.grid(True)
plt.tight_layout()
plt.savefig("dislocation_density_evolution.png")

