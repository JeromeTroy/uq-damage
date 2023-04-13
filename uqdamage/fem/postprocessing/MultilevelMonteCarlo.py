"""
Methods for performing Multilevel Monte Carlo on data
"""

import numpy as np 
import pandas as pd 
import logging

from scipy.stats import gaussian_kde

from uqdamage.fem.experiments.UniaxialStress import interpolate_stress_on_strain

def stress_strain_standard_mlmc(df : pd.DataFrame):
    """
    Use standard MLMC approaches to compute the estimator
    variance and standard error for data on a stress-strain curve

    Input:
        df : pandas.DataFrame object
            data in question.  
            Must have keys : 
            - "level" indicating refinement level
            - "seed" identifier of sample number e.g. sample 10 on level 1 => seed = 10
            - "strain" uniform grid of strain coordinates
            - "stress" stress values on each strain point, viewed as QoI

    Output:
        dictionary object containing the following fields
        - "strain" the uniform strain grid of the dataframe
        - "estimator" the mean estimator for the stress values
        - "variance" the variance on the stress values as computed by MLMC
        - "standard_error" the standard error on the estimator
    """

    # determine all unique levels
    levels = sorted(list(set(df["level"])))
    samples_per_level = [None] * len(levels)
    ε = None 

    # allocate storage for differences
    gσ = []
    for j, ℓ in enumerate(levels):
        # determine data currently in use
        level_data = df.query(f"level == {ℓ}")
        samples_per_level[j] = len(list(set(level_data["seed"])))

        logging.info(f"Level {ℓ}")
        logging.info(f"No. seeds {samples_per_level[j]}")
        if j == 0:
            # first pass, G = Data
            logging.info("First pass")
            pivotted = level_data.pivot(
                    index="strain", columns="seed", values="stress"
                )
            gσ.append(
                np.array(pivotted)
            )

            # additionally, compute strain
            ε = np.array(pivotted.index)

        else:
            # collect previous data for given seeds
            max_seed = level_data["seed"].max()
            logging.info(f"Maximum seed value {max_seed}")
            prev_data = df.query(f"level == {levels[j-1]}").query(f"seed <= {max_seed}")
            logging.info(f"Prev data, max seed {prev_data['seed'].max()}")
            logging.info(f"No. seeds previous level {len(list(set(prev_data['seed'])))}")

            # assignment of bias
            mat_curr = np.array(level_data.pivot(index="strain", columns="seed", values="stress"))
            mat_prev = np.array(prev_data.pivot(index="strain", columns="seed", values="stress"))
            length = min([mat_curr.shape[1], mat_prev.shape[1]])
            mat_curr = mat_curr[..., :length]
            mat_prev = mat_prev[..., :length]
            logging.info(f"Current matrix shape {mat_curr.shape}")
            logging.info(f"Previous matrix shape {mat_prev.shape}")

            gσ.append(
                mat_curr - mat_prev
            )
    
    # apply mean and variance operators
    gσ_means = list(map(lambda Σ: np.mean(Σ, axis=1), gσ))
    gσ_vars = list(map(lambda Σ: np.std(Σ, axis=1)**2, gσ))

    σ_est_var = np.sqrt(sum(v / n for v, n in zip(gσ_vars, samples_per_level)))
    # compute estimator, variance and standard error
    output = {
        "strain" : ε,
        "estimator" : sum(gσ_means),
        "standard_error" : σ_est_var,
        "variance" : np.power(σ_est_var, 2) * max(samples_per_level)
    }

    return output

def ensure_nondecreasing(sequence):
    """
    Helper function to ensure a sequence of data is non decreasing

    Input:
        sequence : list of numbers
    Output:
        new_seq : list of numbers, which is now non decreasing

    If the next number is lower than the previous, the next number is set to the previous
    """

    has_changed = False 
    new_seq = sequence.copy()

    for j, (v1, v2) in enumerate(zip(sequence[:-1], sequence[1:])):
        
        if v2 < v1:
            # copy value from v1 to next placement
            new_seq[j + 1] = v1
            has_changed = True 
    
    # guard against case of needing to repeat this
    if has_changed:
        return ensure_nondecreasing(new_seq)

    return new_seq
        

def cdf_stopping_time(df : pd.DataFrame, stopping_time : str, t_grid):
    """
    Use conditional probability based MLMC to compute the
    CDF for a stopping time

    Input:
        df : pandas.DataFrame object
            must contain keys
            - "level" indicates the refinement level
            - "seed" identifier for each sample, e.g. sample 10 at level 1 => seed = 10
            - stopping_time : value in question
        stopping_time : string
            name of the stopping time attribute
        t_grid : list or array object
            array of time nodes on which problem data is defined
    
    Output:
        cdf_vals : callable function with signature p = mlmc_cdf(t), and 0 <= p <= 1, 
            non decreasing function
            The CDF computed using MLMC
    """

    levels = sorted(list(set(df["level"])))

    cdf_vals = np.zeros_like(t_grid)

    for j, ℓ in enumerate(levels):
        curr_data = df.query(f"level == {ℓ}")
        kde = gaussian_kde(np.array(curr_data[stopping_time]))

        if j == 0:
            # compute starting CDF from KDE directly
            cdf = lambda t: kde.integrate_box(0, t)
            cdf_vals = np.array(list(map(cdf, t_grid)))

        else:
            max_seed = curr_data["seed"].max()
            prev_data = df.query(f"level == {levels[j-1]}").query(f"seed <= {max_seed}")

            prev_stopping_time = np.array(prev_data[stopping_time])
            curr_stopping_time = np.array(curr_data[stopping_time])

            # ensure sizes are each the same
            length = min([len(prev_stopping_time), len(curr_stopping_time)])
            prev_stopping_time = prev_stopping_time[:length]
            curr_stopping_time = curr_stopping_time[:length]

            # update cdf values
            cdf_vals = ensure_nondecreasing(
                update_cdf_values(t_grid, prev_stopping_time, curr_stopping_time, cdf_vals)
            )

    return cdf_vals

def update_cdf_values(t, prev_data, curr_data, current_cdf):
    
    # conditional probabilities
    def conditional_kde(τ : float, good : bool) -> float:
        """
        Evaluate conditional probablitities for both good and bad from previous data

        Input: τ : input point, scalar, good : boolean
        Output: integral : scalar in [0, 1]
        """
        if good:
            indices = prev_data <= τ
        else:
            indices = prev_data > τ
        # guard clauses for no data
        if np.sum(indices) == 0:
            return 0
        if np.sum(indices) == 1:
            return (curr_data[indices] <= τ).astype(float)
        
        # finally try to build the conditional KDE
        try:
            cond_kde = gaussian_kde(curr_data[indices])
            return cond_kde.integrate_box(0, τ)
        except np.linalg.LinAlgError:
            # repeated values, default to basic approach
            return np.mean(curr_data[indices] <= τ)

    # apply these to input data to build new CDF
    new_cdf_vals = np.array(list(map(lambda τ: conditional_kde(τ, True), t))) * current_cdf + \
        np.array(list(map(lambda τ: conditional_kde(τ, False), t))) * (1 - current_cdf)
    
    return new_cdf_vals