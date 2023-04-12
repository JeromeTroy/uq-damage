import numpy as np
import matplotlib.pyplot as plt
import logging
logging.getLogger().setLevel(logging.INFO)

from fenics import *

from uqdamage.fem.experiments import ExpandingRing


r = 0.8
h = 0.07
ν = 0.1
c = 1e-10
Δt = 0.01
tmax = 5

nt = int(tmax / Δt)
nsave = 3

ring = ExpandingRing.RingProblem((r, h), ν, c, Δt)

plot(ring.mesh)
plt.show()

δσ = Constant(0.1)
Σ = Function(ring.Vθ)

cs_vals = Σ.vector()[:]
x = ring.Vθ.tabulate_dof_coordinates()
cs_vals += 1
cs_vals[np.abs(np.arctan2(x[:, 1], x[:, 0])) < 0.01] = 0.9
Σ.vector()[:] = cs_vals

ring.set_softening_fields(Σ, δσ)

xdmf = XDMFFile("ring_test.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True
xdmf.parameters["rewrite_function_mesh"] = False
xdmf.write(Σ)
ring.integrate(nt, nsave, xdmf_file=xdmf)
print("Done")