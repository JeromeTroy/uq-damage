from fenics import *
import matplotlib.pyplot as plt 

from uqdamage.fem.Domains2D import RectangularDomain 

length = 1
width = 0.3
h = 0.1

nx = int(length / h)
ny = int(width / length * nx)

Ω = RectangularDomain(length, width, nx, ny)

plot(Ω)
plt.show()