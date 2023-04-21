import numpy as np
import matplotlib.pyplot as plt
import logging
logging.getLogger().setLevel(logging.INFO)

from scipy.signal import find_peaks

from fenics import *

from uqdamage.fem.experiments import ExpandingRing
from uqdamage.fem.RandomField import RandomLogNormalField

from uqdamage.fem.postprocessing.DamageAveraging import angle_averaged_damage
from uqdamage.fem.DamageBase import load_data


r = 0.8
h = 0.05
ν = 0.1
c = 0.5
Δt = 0.05
tmax = 3

δ = 0.1
ℓ = 0.1
δσ = Constant(0.1)
num_modes = 50

nt = int(tmax / Δt)
nsave = 1

ring = ExpandingRing.RingProblem((r, h), ν, c, Δt, 
                                 α_f=0.2, α_m=0.2, η_m=1e-3, η_k=1e-3)

plot(ring.mesh)
plt.show()

k = lambda x, y: δ * np.exp(-np.linalg.norm(x - y)**2 / (2 * ℓ**2))
mean = Constant(1)
max_range = 3 * ℓ
X = RandomLogNormalField(mean=mean, 
                         kernel=k, 
                         space=ring.Vθ, 
                         num_modes=num_modes,
                         max_range=max_range)

np.random.seed(1)
Σ = X.sample_field()
art = plot(Σ)
plt.colorbar(art)
plt.show()

ring.set_softening_fields(Σ, δσ)

xdmf = XDMFFile("ring.xdmf")
fout = HDF5File(ring.mesh.mpi_comm(), "ring_data.h5", "w")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True
xdmf.parameters["rewrite_function_mesh"] = False
xdmf.write(Σ)
ring.integrate(nt, nsave, xdmf_file=xdmf, data_file=fout)
print("Done integration")

print("Beginning post processing")

_, damage = load_data(ring.mesh, "ring_data.h5", ring.Vθ, "Damage")
final_damage = damage[-1]

φ = np.linspace(0, 2 * np.pi, 100)
δφ = 5 * np.pi / 180        # 5 deg
ω = angle_averaged_damage(final_damage, φ, width=δφ)

fig = plt.figure()
ax = fig.add_subplot(121)
plot(final_damage)
ax.set_title("$\\mathcal{D}$")
ax = fig.add_subplot(122, projection="polar")
ax.plot(φ, ω)
ax.set_title("$\\overline{\\mathcal{D}}(\\phi)$")
plt.show()

peaks, properties = find_peaks(ω, prominence=(1e-2, None))

plt.plot(φ, ω)
plt.plot(φ[peaks], ω[peaks], "^", label="fracture locations")
plt.legend()
plt.xlabel("$\\phi$")
plt.ylabel("$\\overline{\\mathcal{D}}$")
plt.show()

number_fragments = len(peaks)
print("Number of fragments: ", number_fragments)
print("Done")