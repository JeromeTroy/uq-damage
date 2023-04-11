from fenics import plot 
import matplotlib.pyplot as plt

from uqdamage.fem.Domains2D import RectangularNotchedDomain

length = 1
width = 0.3
h = 0.1

notch_width = 0.05
notch_depth = 0.1

Ω = RectangularNotchedDomain(length, width, h, notch_width, notch_depth)

plot(Ω)
plt.show()