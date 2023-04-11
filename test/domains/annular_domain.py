from fenics import plot 
import matplotlib.pyplot as plt

from uqdamage.fem.Domains2D import AnnularDomain

inner_radius = 0.8
h = 0.1

Ω = AnnularDomain(inner_radius, h)

plot(Ω)
plt.show()