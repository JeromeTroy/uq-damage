import numpy as np
import matplotlib.pyplot as plt
import sys
import logging
logging.getLogger().setLevel(logging.INFO)

from fenics import *

from uqdamage.fem.experiments import UniaxialStress, NotchedUniaxialStress
from uqdamage.fem.RandomField import RandomWeibullField

from random_field_decomposition import laplacian_decomposition

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

# forcing parameters
t0 = 1
g0 = 0.18

g_left = Expression(
    ("-g * (t/a)*(t/a) / sqrt(1 + (t/a*t/a))", "0"), 
    g=g0, a=t0, t=0, degree=1
)
g_right = Expression(
    ("g * (t/a)*(t/a) / sqrt(1 + (t/a*t/a))", "0"), 
    g=g0, a=t0, t=0, degree=1
)

# correlation length 
ℓ = 0.2
# shape parameter
κ = 5
n_modes = 100

mesh_params = (
    {"width" : w, 
        "notch_width" : notch_width,
        "notch_depth" : notch_depth}, 
        1/nx
)
problem = NotchedUniaxialStress(mesh_params, ν, Δt, g_left, g_right)
xdmf_name = "notched_uniaxial_stress_{:d}.xdmf"

# instantiate a random field
seed = int(sys.argv[1])
δσ = Constant(0.1)

μ = Constant(1)
# fix correlation length
eig_gen = lambda n_max: laplacian_decomposition(n_max, ℓ)
X = RandomWeibullField(mean=μ, eig_gen=eig_gen, 
                         space=problem.Vθ, num_modes=n_modes, 
                         shape_param=κ)
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

