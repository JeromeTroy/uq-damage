import numpy as np
import matplotlib.pyplot as plt
import logging
logging.getLogger().setLevel(logging.INFO)

from fenics import *

from uqdamage.fem.experiments import UniaxialStress, NotchedUniaxialStress

NOTCHED = False

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

if not NOTCHED:
    mesh_params = (w, (nx, ny))
    problem = UniaxialStress(mesh_params, ν, Δt, g_left, g_right)
    xdmf_name = "uniaxial_stress.xdmf"
else:
    mesh_params = (
        {"width" : w, 
         "notch_width" : notch_width,
         "notch_depth" : notch_depth}, 
         1/nx
    )
    problem = NotchedUniaxialStress(mesh_params, ν, Δt, g_left, g_right)
    xdmf_name = "notched_uniaxial_stress.xdmf"

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

