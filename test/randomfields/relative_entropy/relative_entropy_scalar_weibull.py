import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma as Γ
import seaborn as sns 

plt.rcParams.update({"font.size" : 12})
sns.set_theme(context="paper", style="ticks", palette="colorblind", font="serif")

γ = 0.5772
f = lambda r: -np.log(r) - γ * (1 - r) - 1 + Γ(1 + r)

r_vals = np.linspace(0, 3, 100)[1:]
plt.semilogy(r_vals, f(r_vals))
plt.xlabel("$r$")
plt.ylabel("$D_{KL}(\\omega_1 || \\omega_2)$")
plt.savefig("relent_scalar_weibull.png")
plt.show()