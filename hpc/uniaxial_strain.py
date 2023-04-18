import numpy as np
import matplotlib.pyplot as plt
import sys
import logging
logging.getLogger().setLevel(logging.INFO)

from fenics import *

from uqdamage.fem.experiments import UniaxialStrain
from uqdamage.fem.RandomField import RandomLogNormalField

from random_field_decomposition import laplacian_decomposition

NOTCHED = True

# domain setup
w = 0.3

notch_width = 0.1
notch_depth = 0.1

nx = 20
ny = 6

ν = 0.1
# time parameters
Δt = 0.01
tmax = 10
nt = int(tmax / Δt)
nsave = 5

# expansion speed
c = 1e-1
f = Expression(
    ("c * t", "0"), 
    c=c, t=0, degree=1
)

# correlation length 
ℓ = 0.2
n_modes = 100

# create a problem
mesh_params = (w, (nx, ny))
problem = UniaxialStrain(mesh_params, ν, Δt, f)
xdmf_name = "uniaxial_strain_sample_{:d}.xdmf"

# instantiate a random field
seed = int(sys.argv[1])
δσ = Constant(0.1)

μ = Constant(1)
# fix correlation length
eig_gen = lambda n_max: laplacian_decomposition(n_max, ℓ)
X = RandomLogNormalField(mean=μ, eig_gen=eig_gen, 
                         space=problem.Vθ, num_modes=n_modes)
np.random.seed(seed)
Σ = X.sample_field()

problem.set_softening_fields(Σ, δσ)

xdmf = XDMFFile(xdmf_name.format(seed))
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True
xdmf.parameters["rewrite_function_mesh"] = False
xdmf.write(Σ)

problem.integrate(nt, nsave, xdmf_file=xdmf)
print("Done")


