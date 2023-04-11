from fenics import plot
import matplotlib.pyplot as plt 

from uqdamage.fem.Domains2D import RectangularDomain 

length = 1
width = 0.3
h = 0.1

nx = int(length / h)
ny = int(width / length * nx)
res = (nx, ny)

Ω = RectangularDomain(length, width, res)

plot(Ω)
plt.show()

Ω = RectangularDomain(length, width, nx)

plot(Ω)
plt.show()