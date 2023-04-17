import numpy as np
from fenics import (
    Constant, UnitSquareMesh, FunctionSpace, plot, Function
)
import matplotlib.pyplot as plt

from uqdamage.fem.RandomField import RandomGaussianField

ℓ = 0.1
def k(x, y):
    return np.exp(-np.linalg.norm(x - y)**2 / (2 * ℓ**2))

Ω = UnitSquareMesh(20, 20, diagonal="crossed")
V = FunctionSpace(Ω, "DG", 0)
n_modes = 100
max_range = 3 * ℓ
μ = Function(V)

X = RandomGaussianField(
    mean=μ, kernel=k, space=V, num_modes=n_modes, max_range=max_range
)

fname = "decomp.npz"
X.save_eigendecomp(fname)

np.random.seed(0)
Xn = X.sample_field()

Y = RandomGaussianField(mean=μ, kernel=k, space=V, num_modes=n_modes)
Y.load_eigendecomp(fname)

np.random.seed(0)
Yn = Y.sample_field()

print(np.linalg.norm(Yn.vector()[:] - Xn.vector()[:]))

art = plot(Xn)
plt.colorbar(art)
plt.show()