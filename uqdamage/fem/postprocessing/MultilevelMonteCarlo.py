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
        sequence : numpy array size (n,)
    Output:
        new_seq : numpy array size (n,), which is nondecreasing

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

    # fix values larger than 1 for CDF
    new_seq[new_seq > 1] = 1
    
    return new_seq
        

def quick_cdf(arr : np.ndarray, x : float) -> float:
    """
    Quick CDF computation for P(X <= x)

    Input:
        arr : list like of floats
            arr[i] = X_i, i'th sample
        x : float
            comparison value, x in P(X <= x)
    Output:
        p : float in [0, 1]
            evaluation of P(X <= x)
    """

    if len(arr) == 0:
        return 0
    
    return np.sum(arr <= x) / len(arr)

def bayes_update(p : float, conditions):
    """
    Helper function which performs Bayesian update

    Input:
        p : float
            prior probability
        conditions : 2-tuple, of P(E | prev) and P(E | not prev)
            where prev occurs with probability p
    Output:
        updated probability using bayes update proceedure
    """
    for i in range(len(conditions)):
        if np.isnan(conditions[i]) or np.isinf(conditions[i]):
            conditions[i] = 0
    
    return p * conditions[0] + (1 - p) * conditions[1]

def brute_force_cdf_update(x : float, 
                           new_data : np.ndarray, 
                           prev_data : np.ndarray,
                           prev_cdf : float):
    """
    Compute updated CDF using brute force method:
    subsample new_data based on prev_data
    evaluate quick CDF on sub sample

    Input:
        x : float
            comparison value, used in P(X <= x)
        new_data, prev_data : numpy arrays
            arrays of data from previous level and current level
        prev_cdf : float in [0, 1]
            previous level's CDF value
    Output:
        new_cdf : float in [0, 1]
            updated CDF value
    """

    conditions = (
        quick_cdf(new_data[prev_data <= x], x), 
        quick_cdf(new_data[prev_data > x], x)
    )

    return bayes_update(prev_cdf, conditions)

def sum_indicator_cdf_update(x : float, 
                             new_data : np.ndarray,
                             prev_data : np.ndarray,
                             prev_cdf : float):
    """
    Compute updated CDF using sum of indicator method
    given subsampled data on previous level and this level
    X_l, X_{l+1}, 
    p_l(x) = P(X_l <= x)
    p_l(x) = 0 =>
    P(X_{l+1} <= x | X_l <= x) = E[1(X_{l+1} <= x) 1(X_l <= x)] / p_l(x)
    else P(. | .) = 0

    Input:
        x : float
            comparison value, used in P(X <= x)
        new_data, prev_data : numpy arrays
            arrays of data from previous level and current level
        prev_cdf : float in [0, 1]
            previous level's CDF value
    Output:
        new_cdf : float in [0, 1]
            updated CDF value
    """

    denoms = np.array((
        np.sum(prev_data <= x), 
        np.sum(prev_data > x)
    ))
    numers = np.array((
        np.sum(np.logical_and(prev_data <= x, new_data <= x)),
        np.sum(np.logical_and(prev_data > x, new_data <= x))
    ))
    
    ratios = numers / denoms
    # fix div 0 errors
    ratios[denoms == 0] = 0

    return bayes_update(prev_cdf, ratios)

def subsampled_kde_cdf_update(x : float, 
                             new_data : np.ndarray,
                             prev_data : np.ndarray,
                             prev_cdf : float):
    """
    Compute updated CDF using KDE of subsamples
    given samples X_{l+1} and X_l
    first take only X_{l+1} where X_l <= x
    build KDE on these subsamples and integrate

    Input:
        x : float
            comparison value, used in P(X <= x)
        new_data, prev_data : numpy arrays
            arrays of data from previous level and current level
        prev_cdf : float in [0, 1]
            previous level's CDF value
    Output:
        new_cdf : float in [0, 1]
            updated CDF value
    """

    # subsample data
    new_data_sub = (new_data[prev_data <= x],
                    new_data[prev_data > x])
    # check that KDE can be constructed
    n_unique = [len(list(set(data))) for data in new_data_sub]
    conditionals = np.zeros(2)
    for index, (nu, dat) in enumerate(zip(n_unique, new_data_sub)):
        # guard against case not enough entries to build KDE
        if nu == 0:
            conditionals[index] = 0
            continue
        if nu == 1:
            conditionals[index] = float(dat[0] <= x)
            continue

        sub_kde = gaussian_kde(dat)
        conditionals[index] = sub_kde.integrate_box_1d(0, x)
    
    return bayes_update(prev_cdf, conditionals)

def joint_kde_cdf_update(x : float, 
                        new_data : np.ndarray,
                        prev_data : np.ndarray,
                        prev_cdf : float):
    """
    Compute updated CDF using joint KDE
    build KDE on prev data ρ(prev)
    build KDE on joint new and prev data
    π(new, prev)
    conditional : P(new <= x, prev <= x) / P(prev <= x)
    dido but with prev > x.

    Input:
        x : float
            comparison value, used in P(X <= x)
        new_data, prev_data : numpy arrays
            arrays of data from previous level and current level
        prev_cdf : float in [0, 1]
            previous level's CDF value
    Output:
        new_cdf : float in [0, 1]
            updated CDF value
    """

    # build joint kde
    joint_data = np.vstack([new_data, prev_data])
    joint_kde = gaussian_kde(joint_data)

    prev_kde = gaussian_kde(prev_data)

    conditionals = [
        joint_kde.integrate_box([-np.inf, -np.inf], [x, x]) / 
            prev_kde.integrate_box_1d(-np.inf, x),
        joint_kde.integrate_box([-np.inf, x], [x, np.inf]) / 
            prev_kde.integrate_box_1d(x, np.inf)
    ]
    return bayes_update(prev_cdf, conditionals)

def cdf_mlmc(df : pd.DataFrame, x_nodes : np.ndarray, value_name : str, 
             method : str = "joint", ensure_nondec=False):
    """
    Construct the CDF for a distribution using MLMC

    Input:
        df : pandas.DataFrame object
            data in question, must have the following keys
            - "level" refinement level on which sample was generated
            - "seed" identifier of sample number, e.g. sample 10 on level 2 is seed 10
            - value_name value in question on which the CDF will be constructed
                (see below)
        x_nodes : numpy array of size (n,)
            nodes on which to compute the CDF
            the cdf will be computed pointwise at each x node
        value_name : string
            name of value to build CDF for
        method : string, optional
            name of method to use to build conditional updates to CDF
            possible values are
                - "joint" (default) - build joint KDE and integrate
                - "subsample" - subsample data based on previous and do KDE
                - "indicator" - probability as expectation of indicator with
                    bayesian updating
                - "brute" - brute-force type method, subsample data
                    and compute probabilities with divisions by length of array
        ensure_nondec : boolean, optional
            whether to enforce nondecreasing on the CDF
            the default is False
    Output:
        cdf : numpy array of size (n,)
            CDF values at corresponding x nodes
    """

    # preprocess data - unique levels
    levels = sorted(list(set(df["level"])))
    # storage for cdfs computed using each level
    cdfs = []

    # determine method to do Bayesian updating to CDF
    if method == "brute" :
        cdf_update = brute_force_cdf_update
    elif method == "indicator" : 
        cdf_update = sum_indicator_cdf_update
    elif method == "subsample" : 
        cdf_update = subsampled_kde_cdf_update
    else:
        cdf_update = joint_kde_cdf_update

    for index, level in enumerate(levels):
        # guard clause in case no CDF computed yet
        curr_data = df.query(f"level == {level}")
        if index == 0:
            # build the CDF on the first level using standard MC for KDE
            kde = gaussian_kde(curr_data[value_name])
            cdfs.append(np.array(list(map(
                lambda x: kde.integrate_box_1d(0, x), x_nodes
            ))))
            # next iteration in loop
            continue
        
        # take previous level's data which is repeated on current level
        max_seed = max(curr_data["seed"])
        prev_data = np.array(df.query(
            f"level == {levels[index-1]}").query(
            f"seed <= {max_seed}")[value_name])
        
        # cast curr_data to array
        new_data = np.array(curr_data[value_name])
        
        # compute new cdf
        cdfs.append(np.array(list(map(
            lambda x, c: cdf_update(x, new_data, prev_data, c),
            x_nodes, cdfs[-1]
        ))))

    if ensure_nondec:
        cdfs = list(map(ensure_nondecreasing, cdfs))
    
    # return a dictionary of constructed cdfs with each level
    output = {
        level : cdf 
        for (level, cdf) in zip(levels, cdfs)
    }
    return output
        
