import numpy as np
import matplotlib.pyplot as plt
import logging
logging.getLogger().setLevel(logging.INFO)

from fenics import *

from uqdamage.fem.experiments import ExpandingRing


r = 0.8
h = 0.2
ν = 0.1
c = 0.5
Δt = 0.05
tmax = 5

nt = int(tmax / Δt)
nsave = 1

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

plot(ring.mesh)
plt.show()

δσ = Constant(0.1)
Σ = Function(ring.Vθ)

cs_vals = Σ.vector()[:]
x = ring.Vθ.tabulate_dof_coordinates()
cs_vals += 1
cs_vals[np.abs(np.arctan2(x[:, 1], x[:, 0])) < np.pi/3] = 0.5
Σ.vector()[:] = cs_vals

ring.set_softening_fields(Σ, δσ)

xdmf = XDMFFile("ring_test.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True
xdmf.parameters["rewrite_function_mesh"] = False
xdmf.write(Σ)
ring.integrate(nt, nsave, xdmf_file=xdmf)
print("Done")