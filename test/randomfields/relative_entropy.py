# building relative entropy between similar Gaussian fields

import numpy as np 

def λ_ratio(ℓ1, ℓ2, k, m):
    """
    For exact covariance operator, the ratio of the eigenvalues
    for differing correlation lengths

    Input:
        ℓ1, ℓ2 : float > 0, 
            correlation length of each field
        k, m : int >= 1
            current modes
    Output:
        result = λ1 / λ2
    """

    coef = 4 * np.pi**2 * (k**2 + m**2)
    top = 1 + 1 / (ℓ2**2 * coef)
    bot = 1 + 1 / (ℓ1**2 * coef)
    return top / bot 

def λ_ratio_m1(ℓ1, ℓ2, k, m):
    """
    For exact covariance operator, the ratio of the eigenvalues - 1
    for differing correlation lengths
    This function carries through some computations to reduce subtractive
    cancellation errors

    Input:
        ℓ1, ℓ2 : float > 0, 
            correlation length of each field
        k, m : int >= 1
            current modes
    Output:
        result = λ1 / λ2 - 1
    """
    top = ℓ1**2 - ℓ2**2
    coef = 4 * np.pi**2 * ℓ1**2 * ℓ2**2 * (k**2 + m**2)
    bot = ℓ2**2 + coef
    return top / bot

def summand(k, m, ℓ1, ℓ2):
    """
    For exact covariance operator each term in the sum for computing 
    relative entropy

    Input:
        ℓ1, ℓ2 : float > 0, 
            correlation length of each field
        k, m : int >= 1
            current modes
    Output:
        result = λ1 / λ2 - 1 - log(λ1 / λ2)
    """
    return λ_ratio_m1(ℓ1, ℓ2, k, m) - np.log(λ_ratio(ℓ1, ℓ2, k, m))

def do_sum(n_modes, ℓ1, ℓ2, yield_freq=1):
    """
    Compute the partial sum for the relative entropy
    This operates as a generator, so will emit values leading up to the 
    stopping point

    Input:
        n_modes : int or float > 0,
            maximum number of modes to use, 
            taken to be the max of k^2 + m^2
        ℓ1, ℓ2 : float > 0, 
            correlation length of each field
        yield_freq : int, optional
            iterations of k before emitting a value.
            The default is 1, so a value will be emitted each time k changes
    Output:
        partial sum computing relative entropy
    """
    # maximum allowed k
    k_max = np.sqrt(n_modes) + 1
    # copy to m, so k^2 + m^2 <= n_modes
    m_max = k_max

    total = 0
    k = 0
    # generator pattern
    while k < k_max + 1:
        k += 1
        # diagonal sum trick
        for m in range(1, k):
            total += summand(k, m, ℓ1, ℓ2)
        # output for generator at specified intervals
        if k % yield_freq == 0:
            # mult by 4 from sum over j = 1, 2, 3, 4
            yield 4 * total


def compute_sum(τ, ℓ1, ℓ2, yield_freq=1):
    """
    For exact covariance operator, compute the relative entropy
    between fields of differing correlation lengths up to a given tolerance.
    
    Note, if a tolerance of 0 is used, this function will enter an 
    infinite loop. This function relies on the above function being a generator,
    and so takes n_modes = inf.

    Input:
        τ : float > 0, 
            tolerance value for partial sum
        ℓ1, ℓ2 : float > 0, 
            correlation length of each field
        yield_freq : int, optional
            iterations of k before emitting a value.
            The default is 1, so a value will be emitted each time k changes
            This value is used in the do_sum() function above
    Output:
        relative entropy approximation
    """

    # establish an infinite generator
    total = do_sum(np.inf, ℓ1, ℓ2, yield_freq=yield_freq)

    prev = 0
    for index, value in enumerate(total):
        # initialize
        if index == 0:
            prev = value 
            continue

        # compute error and compare
        difference = abs(value - prev)
        # good enough, stop and return
        if difference < τ:
            return value      
        
        # otherwise, update previous value and continue in loop
        prev = value   
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    import seaborn as sns
    import pandas as pd

    plt.rcParams.update({"font.size" : 12})
    sns.set_theme(context="paper", 
                  style="ticks", 
                  font="serif", 
                  palette="colorblind")

    # script for making plot
    ℓ1_vals = [0.1, 0.2, 0.5]

    # stopping parameters for sum computation
    τ = 1e-6
    yield_freq = 10

    # alternative model, note ℓ2 = 0 may be problematic, so remove it
    ℓ2 = np.linspace(0, 1, 101)[1:]

    # compute relative entropy for each ℓ2 value
    sums = [np.array(list(map(
                lambda ℓ: compute_sum(τ, ℓ1, ℓ, yield_freq=yield_freq), 
                ℓ2
            )))
            for ℓ1 in ℓ1_vals]
    
    ℓ1_arr = np.array([ℓ1 * np.ones(len(ℓ2)) for ℓ1 in ℓ1_vals]).ravel()

    relent_str = "$D_\\mathrm{KL}(\\gamma_1 \\mid \\mid \\gamma_2)$"
    df = pd.DataFrame.from_dict({
        "$\\ell_2$" : np.array([ℓ2] * len(ℓ1_vals)).ravel(), 
        relent_str : np.array(sums).ravel(), 
        "$\\ell_1$" : ℓ1_arr
    })
    
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    sns.lineplot(data=df, x="$\\ell_2$", y=relent_str, hue="$\\ell_1$", ax=ax)
    plt.savefig("gaus_field_relent.png")
    plt.show()
