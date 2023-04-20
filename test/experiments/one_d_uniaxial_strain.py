import numpy as np
import matplotlib.pyplot as plt
import logging
logging.getLogger().setLevel(logging.INFO)

from fenics import *

from uqdamage.fem.experiments import OneDUniaxialStrain

nx = 20

ν = 0.1
# time parameters
Δt = 0.01
tmax = 10
nt = int(tmax / Δt)
nsave = 5

# expansion speed
c = 1e-1
f = Expression(
    ("c * t",), 
    c=c, t=0, degree=1
)

xdmf_name = "one_d_uniaxial_strain.xdmf"

problem = OneDUniaxialStrain(nx, ν, Δt, f)

δσ = Constant(0.1)
Σ = Function(problem.Vθ)
cs_vals = Σ.vector()[:]
x = problem.Vθ.tabulate_dof_coordinates().ravel()
cs_vals += 1
indices = np.logical_and(x < 0.6, x > 0.4)
cs_vals[indices] = 0.8
Σ.vector()[:] = cs_vals
problem.set_softening_fields(Σ, δσ)

xdmf = XDMFFile(xdmf_name)
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True
xdmf.parameters["rewrite_function_mesh"] = False
xdmf.write(Σ)

problem.integrate(nt, nsave, xdmf_file=xdmf)
print("Done")


