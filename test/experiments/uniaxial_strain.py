import numpy as np
import matplotlib.pyplot as plt
import logging
logging.getLogger().setLevel(logging.INFO)

from fenics import *

from uqdamage.fem.experiments import UniaxialStrain, NotchedUniaxialStrain

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


if not NOTCHED:
    mesh_params = (w, (nx, ny))
    problem = UniaxialStrain(mesh_params, ν, Δt, f)
    xdmf_name = "uniaxial_strain.xdmf"
else:
    mesh_params = (
        {"width" : w, 
         "notch_width" : notch_width,
         "notch_depth" : notch_depth}, 
         1/nx
    )
    problem = NotchedUniaxialStrain(mesh_params, ν, Δt, f)
    xdmf_name = "notched_uniaxial_strain.xdmf"

δσ = Constant(0.1)
Σ = Function(problem.Vθ)
cs_vals = Σ.vector()[:]
x = problem.Vθ.tabulate_dof_coordinates()
cs_vals += 1
cs_vals[np.linalg.norm(x, axis=1) < 0.1] = 0.8
Σ.vector()[:] = cs_vals
problem.set_softening_fields(Σ, δσ)

xdmf = XDMFFile(xdmf_name)
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True
xdmf.parameters["rewrite_function_mesh"] = False
xdmf.write(Σ)

problem.integrate(nt, nsave, xdmf_file=xdmf)
print("Done")


