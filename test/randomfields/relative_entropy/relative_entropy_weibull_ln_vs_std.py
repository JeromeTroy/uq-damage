import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma as Γ
from scipy.optimize import minimize
import seaborn as sns 
from sympy import EulerGamma

plt.rcParams.update({"font.size" : 12})
sns.set_theme(context="paper", style="ticks", palette="colorblind", font="serif")
plt.set_cmap("viridis")

γ = float(EulerGamma.evalf())

kopt = np.pi / np.sqrt(3)
λ_opt = lambda σ: np.exp(-σ**2 / 2 - np.sqrt(3) * γ / np.pi)

μ = lambda σ: -σ**2 / 2
#μ = lambda σ: -σ
R = lambda σ, λ, k: np.log(np.sqrt(2 * np.pi) * σ * k) - γ - 1 + \
    1 / (2*σ**2) * ((np.log(λ) - μ(σ) + γ/k)**2 + np.pi**2 / (6 * k**2))

def get_optimal_params(σ):
    objective = lambda z: R(σ, z[0]**2, z[1]**2)
    opt_res = minimize(objective, (1, 1))

    λ, k = opt_res.x
    return λ**2, k**2

σ_vals = np.power(10, np.linspace(-5, 0))
λ_vals, k_vals = zip(*list(map(get_optimal_params, σ_vals)))

fig, axes = plt.subplots(nrows=2, sharex=True)
axes[0].semilogx(σ_vals, λ_vals)
axes[0].set_ylabel("$\\lambda$")

axes[1].loglog(σ_vals, k_vals)
axes[1].set_ylabel("$k$")
axes[1].set_xlabel("$\\sigma$")

plt.savefig("diff_gaus_opt_params.png")
plt.show()

plt.loglog(σ_vals, list(map(R, σ_vals, λ_vals, k_vals)))
plt.ylabel("$D_{KL}(\\omega || \\ell)$")
plt.xlabel("$\\sigma$")

plt.savefig("optimal_relent_diff_gaus.png", bbox_inches="tight")

plt.show()
