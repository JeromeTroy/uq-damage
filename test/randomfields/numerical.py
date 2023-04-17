import numpy as np
from fenics import (
    Constant, UnitSquareMesh, FunctionSpace, plot
)

import matplotlib.pyplot as plt

from uqdamage.fem.RandomField import RandomGaussianField

ℓ = 0.1
def k(x, y):
    return np.exp(-np.linalg.norm(x - y)**2 / (2 * ℓ**2))

μ = Constant(0)
Ω = UnitSquareMesh(20, 20, diagonal="crossed")
V = FunctionSpace(Ω, "DG", 0)
n_modes = 100
max_range = 3 * ℓ

X = RandomGaussianField(
    mean=μ, kernel=k, space=V, num_modes=n_modes, max_range=max_range
)
np.random.seed(0)
Xn = X.sample_field()

art = plot(Xn)
plt.colorbar(art)
plt.show()