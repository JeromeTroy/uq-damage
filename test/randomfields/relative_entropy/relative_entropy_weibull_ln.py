import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma as Γ
from scipy.optimize import minimize_scalar
import seaborn as sns 
from sympy import EulerGamma

plt.rcParams.update({"font.size" : 12})
sns.set_theme(context="paper", style="ticks", palette="colorblind", font="serif")
plt.set_cmap("viridis")

γ = float(EulerGamma.evalf())

R = lambda λ, k: 0.5 * np.log(2 * np.pi * λ * k**2) - 1 - γ + \
    ((0.5 * λ + γ / k + np.log(λ))**2 + 1./6 * np.pi**2 / k**2) / (2 * λ)

def get_optimal_k(λ):
    objective = lambda k: R(λ, k**2)

    opt_res = minimize_scalar(objective, bounds=[1e-10, 10])
    return opt_res.x**2

λ_vals = np.power(10, np.linspace(-5, 0))

k_vals = np.array(list(map(get_optimal_k, λ_vals)))

fig, axes = plt.subplots(nrows=2, sharex=True)
axes[0].semilogx(λ_vals, k_vals)
axes[0].set_ylabel("$k_{opt}$")

axes[1].loglog(λ_vals, R(λ_vals, k_vals))
axes[1].set_ylabel("$D_{KL}(\\omega || \\ell)$")
axes[1].set_xlabel("$\\lambda$")

plt.savefig("optimal_relent_same_gaus.png")

plt.show()
