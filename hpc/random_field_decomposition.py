import numpy as np 
from itertools import product 

def φ(x : np.ndarray, m : int, n : int):
    """
    Eigenfunction for integers m, n

    Input:
        x : numpy array, size (k, 2)
            input data points
        m, n : ints > 0
            modes for x and y components
    Output:
        4-tuple containing eigenmodes for m, n
        entries are (like)
            sin(.x.) sin(.y.), 
            sin(.x.) cos(.y.),
            cos(.x.) sin(.y.),
            cos(.x.) cos(.y.)
    """

    amplitude = 2

    # group amplitude with x terms
    sinx = amplitude * np.sin(2 * m * np.pi * x[:, 0])
    cosx = amplitude * np.cos(2 * m * np.pi * x[:, 0])
    siny = np.sin(2 * n * np.pi * x[:, 1])
    cosy = np.cos(2 * n * np.pi * x[:, 1])

    return sinx * siny, sinx * cosy, cosx * siny, cosx * cosy

def laplacian_decomposition(n_max : int, corr_len : float):
    """
    Decomposition of laplacian operator with correlation length:
    Eigen expansion generator for 
    λ (∇^2 + 1/ℓ^2) φ = φ, period boundary conditions
    This is for covariance operator
    C : f -> ψ, where 
    (∇^2 + 1/ℓ^2) ψ = f, with periodic boundary conditions

    Input:
        n_max : int > 0
            number of modes
            This will be the number of modes in both x and y
        width : float in (0, 1), optional
            ratio of width of domain to the length.
            The default is 1 (square domain)
        ℓ : float > 0, optional
            correlation length.
            The default is 1.

    Output:
        for j in range(n), yields λ, φ where
            λ is a float corresponding to eigenvalue j
            and φ is a callable corresponding to the j'th eigenfunction 

    Note:
        this object acts as a generator, it "yields" rather than returns
        this is a way to create new entries in a quick(er) way
        so the first values are out before the last values finish
    """

    ℓ = corr_len

    # eigenvalues
    λ = lambda m, n: ℓ**2 / (
        1 + 4 * np.pi**2 * ℓ**2 * (m**2 + n**2)
    )

    # sum over m, n for eigenmodes
    # diagonal sum trick - [0, 0], [0, 1], [1, 0], [0, 2], [1, 1], [2, 0] , ...
    sorting_hat = lambda pair: sum(pair)
    pairs = sorted(list(product(range(n_max), range(n_max))), key=sorting_hat)
    eigen_number = 1
    for (m, n) in pairs:
        funcs = lambda x: φ(x, m, n)

        # skips for zeros
        # constant guard
        if m == 0 and n == 0:
            continue
        # constant x guard
        if m == 0:
            # skip sin(x) modes
            yield λ(m, n), lambda x: funcs(x)[2]
            yield λ(m, n), lambda x: funcs(x)[3]
            continue
        # constant y guard
        if n == 0:
            yield λ(m, n), lambda x: funcs(x)[1]
            yield λ(m, n), lambda x: funcs(x)[3]
            continue

        # standard case
        for k in range(4):
            yield λ(m, n), lambda x: funcs(x)[k]