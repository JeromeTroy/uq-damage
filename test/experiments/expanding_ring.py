import numpy as np
import matplotlib.pyplot as plt
import logging
logging.getLogger().setLevel(logging.INFO)

from fenics import *

from uqdamage.fem.experiments import ExpandingRing
from uqdamage.fem.RandomField import RandomLogNormalField

from random_field_decomposition import laplacian_decomposition

r = 0.8
h = 0.05
ν = 0.1
c = 0.5
Δt = 0.05
tmax = 1

nt = int(tmax / Δt)
nsave = 1

num_modes = 50
correlation_length = 0.5
eig_val_scaling = 0.1
eig_gen = lambda modes: laplacian_decomposition(modes, correlation_length, scaling=eig_val_scaling)

solver_params = {
    "newton_solver" : {
        "linear_solver" : "gmres",
        "krylov_solver" : {
            "monitor_convergence" : True,
            "relative_tolerance" : 1e-8,
        },
        "maximum_iterations" : 100
    }
}
ring = ExpandingRing.RingProblem((r, h), ν, c, Δt, solver_params=solver_params, 
                                 α_f=0.2, α_m=0.2, η_m=1e-3, η_k=1e-3)

δσ = Constant(0.1)
X = RandomLogNormalField(Constant(1), eig_gen=eig_gen, 
                         space=ring.Vθ, num_modes=num_modes)

np.random.seed(1)
Σ = X.sample_field()


ring.set_softening_fields(Σ, δσ)

xdmf = XDMFFile("ring_test.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True
xdmf.parameters["rewrite_function_mesh"] = False
xdmf.write(Σ)
ring.integrate(nt, nsave, xdmf_file=xdmf)
print("Done")