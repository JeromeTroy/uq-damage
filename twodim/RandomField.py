"""
RandomField.py - Module for random fields

"""


import numpy as np
from fenics import *
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import gamma
from petsc4py import PETSc
from slepc4py import SLEPc
import traceback
import sys
import logging
#from numba import jit
from math import sqrt
from typing import Callable

# deprecated function
def petsc_to_sparse(pet_mat):
    """
    Convert a PETSc matrix to a scipy.sparse.csr_matrix object

    Input:
        pet_mat - input PETSc matrix
    Output:
        sparse_mat - scipy.sparse.crs_matrix
    """

    logging.warning("Call to deprecated function! Instead use sparse.csr_matrix(pet_mat.getValuesCSR()[::-1])")

    # get shape
    m = pet_mat.size(0)
    n = pet_mat.size(1)

    # m x n is the largest this matrix can be

    # build a csr_matrix via element-wise additions
    data = []
    row_indices = []
    col_indices = []

    # temporary variable for data extraction
    tmp = np.empty([1, 1])

    for i in range(m):
        for j in range(n):
            pet_mat.get(tmp, [i], [j])

            if tmp > 0:
                data.append(tmp[0, 0])
                row_indices.append(i)
                col_indices.append(j)

    # now we have a list of data, corresponding row and column indices
    sparse_mat = sp.csr_matrix( (data, (row_indices, col_indices)),
                shape=(m, n))
    return sparse_mat

#@jit
def covariance_matrix_csr(x: list[tuple[float]], cutoff: float, ρ: Callable) -> (list[int], list[int], list[float]):
    """
    Helper function for getting CSR values for covariance matrix

    Input:
        x : list of 2-tuples of floats
            each entry is a coordinate pair (x, y) of a point in the mesh
        cutoff : float > 0
            cutoff distance.  Points which are further in distance than this
            are considered so far that the covariance is zero.
            This is used to enforce sparsity
        ρ : Callable, signature ρ(x, y) -> float
            where x, y are 2-tuples of floats
            This is the covariance kernel.
    Output:
        counts, cols, data : CSR values for matrix
    """
    # csr matrix values
    counts = [0]
    cols = []
    data = []

    # size, and counter for row stops
    n = len(x)
    counter = 0

    for i, xs in enumerate(x):
        for j, ys in enumerate(x):
            if sqrt((xs[0] - ys[0])**2 + (xs[1] - ys[1])**2) < cutoff:
                # points are close enough to include
                counter += 1
                cols.append(j)
                data.append(ρ(xs, ys))

        # end of row
        counts.append(counter)

    return counts, cols, data

class RandomGaussianField:
    """
    Gaussian random field generator for FEniCs in 1D

    μ - mean. FEniCs funciton.  It is assumued that it is constant or μ ∈ V
    ρ - covariance kernel. Python style funcion
    mesh - FEniCs mesh
    V - Scalar function space for the field
    n_max - Number of modes to use in Karhunen-Loeve expansion
    """

    def __init__(self, μ=None, ρ=None, mesh=None, V=None, n_max = 10, l=0.05):
        self.mesh = mesh
        self.V = V
        self.μ = μ
        self.ρ = ρ
        self.n_max= n_max
        self.ℓ = l

        #self.μ_vector = μ.vector()[:]
        if V is not None:
            self.build_fields()

    def load_field(matrix_vals, μ, V, n_max):

        """
        Load in parameters λ and φ

        Abstracted function, returns an instance of RandomGaussianField
        """

        # build blank field
        X = RandomGaussianField()

        # set values
        X.μ = μ
        X.λ_vals = matrix_vals[0, :]
        X.λ_vals[X.λ_vals < 0] = 0
        X.φ_vals = matrix_vals[1:, :]
        X.V = V
        X.mesh = V.mesh()
        X.n_max = n_max

        return X

    def from_eigen_expansion(eigen_gen, μ, V, n_max, scaling=1, log_every=1000):

        """
        Build a random field from a known eigen expansion

        Input:
            eigen_gen : generator object
                outputs eigenvalue and eigenfunction (callable) for each 
                number (eigen number)
            μ : fenics Expression or Constant-like 
                mean value for field
            V : FunctionSpace object
                function space on which random variable lives
            n_max : int > 0
                number of modes to use
            scaling : float > 0, optional
                scaling for randomness, the default is 1
            log_every : int > 0, optional
                how often to log eigenvalue values, the default is every 1000
        """

        logging.debug("Creating Random field from eigenexpansion")
        X = RandomGaussianField()
        X.μ = μ
        X.V = V
        logging.debug("Extracting mesh from function space")
        X.mesh = V.mesh()

        X.λ_vals = []
        X.φ_vals = []

        logging.debug("Collecting eigenexpansion")
        x = V.tabulate_dof_coordinates()
        for j, (λ, φ) in enumerate(eigen_gen(n_max)):
            if j % log_every == 0:
                logging.info("Index: {:d}, eigenvalue: {:e}".format(
                    j, λ
                ))
            X.λ_vals.append(λ)
            X.φ_vals.append(np.array(φ(x)))
        # number of eigenmodes will be larger than nmax in 2D
        X.n_max = j+1

        X.φ_vals = scaling * np.array(X.φ_vals).T

        return X

    def low_memory_generator_sampling(self, eig_gen, μ, V, n_max, 
        scaling=1, log_every=1000):
        
        x = V.tabulate_dof_coordinates()
        # default vector values are zeros
        Σ = Function(V)

        # running sum
        vals = np.zeros(len(Σ.vector()[:]))

        # assume φ and vals are large arrays
        # avoid storing 
        for j, (λ, φ) in enumerate(eig_gen(n_max)):
            if j % log_every == 0:
                logging.info("Index: {:d}, eigenvalue: {:e}".format(
                    j, λ
                ))
            ξ = np.random.randn()
            vals += ξ * np.sqrt(np.real(λ)) * np.array(φ(x))

        # apply scaling, vals is varying with randomness only
        vals *= scaling

        # apply mean
        Σ = project(μ, V)
        Σ.vector()[:] += vals

        return Σ



    def save_fields(self, filename):
        """
        save data to reload field later
        """
        [m, n] = np.shape(self.ϕ_vals)
        mtx = np.zeros([m+1, n])
        k = len(self.λ_vals)
        mtx[0, :k] = self.λ_vals
        mtx[1:, :] = self.φ_vals

        np.savez_compressed(filename, mtx=mtx, n_max=self.n_max)

    def build_covariance_matrix(self):
        # build the covariance matrix as a quasi-sparse matrix
        cutoff_dist = 3 * self.ℓ

        x = self.V.tabulate_dof_coordinates()
        x = list(map(tuple, x))
        n = len(x)

        logging.info("Building Covariance Matrix CSR Values")
        counts, cols, data = covariance_matrix_csr(x, cutoff_dist, self.ρ)

        # convert lists to sparse matrix
        logging.info("Assembling PETSc Covariance Matrix")
        self.C = PETSc.Mat().createAIJWithArrays((n, n), (counts, cols, data))

    def build_fields(self):

        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        cutoff_dist = 3 * self.ℓ

        self.Mform = u * v * dx
        self.MP = PETScMatrix()
        assemble(self.Mform, tensor=self.MP) # compute & assign to MP
        self.M = self.MP.mat()
        self.N = self.MP.size(0) # system size, square so size(0) = size(1)

        # helper function, builds self.C
        self.build_covariance_matrix()

        # assemble and solve eigenvalue system:
        # (M C M) φ = λ M φ
        # A = (M ⋅ C ⋅ M)
        A = self.M.matMult(self.C.matMult(self.M))
        
        problem = SLEPc.EPS()
        problem.create()
        problem.setOperators(A, self.M)
        # General Hermitian Eigenvalue Problem
        problem.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        # number eigenvalues to collect
        problem.setDimensions(self.n_max, PETSc.DECIDE)

        logging.info("Solving Eigenvalue Problem")
        problem.solve()

        # total number of eigenvalues
        n_ev, _, _ = problem.getDimensions()
        if n_ev < self.n_max:
            logging.error("Did not collect all requested eigenvalues!")
        # collect eigenpairs
        self.λ_vals = []
        self.ϕ_vals = []
        logging.info("Extracting Eigenpairs")
        for i in range(self.n_max):
            # storage for eigenvectors
            # vectors must be rebuilt so that they are not all the same
            vr, _ = A.getVecs()
            vi, _ = A.getVecs()
            λ = problem.getEigenpair(i, vr, vi)
            # hermitian problem, disregard imaginary part
            self.λ_vals.append(np.real(λ))
            self.ϕ_vals.append(vr.array)

        self.λ_vals = np.array(self.λ_vals)
        self.ϕ_vals = np.array(self.ϕ_vals).T

    def get_spectrum(self):
        return self.λ_vals

    def sample_field(self, threshold=None):
        """
        Generate a sample as a random field
        """

        # storage for random numbers
        self.ξ = np.zeros(self.n_max)
        # initialize to the mean
        X = Function(self.V)
        X.assign(self.μ)

        use_λ = self.λ_vals[:]
        if threshold is not None:
            use_λ[use_λ < threshold] = 0

        #for n in range(self.N-1, self.N-self.n_max-1,-1):
        for n in range(self.n_max):
            if(use_λ[n]<0):
                print("WARNING: Eigenvalue {:d} is negative".format(n))

            self.ξ[n] = np.random.randn()
            X.vector()[:]+=np.sqrt(use_λ[n].real) * self.ξ[n] * self.ϕ_vals[:,n].real
        return X

    def get_random_numbers(self):
        return self.ξ

    def set_random_numbers(self, ξ):
        self.ξ = ξ

class RandomLogNormalField(RandomGaussianField):

    def sample_field(self, threshold=None):
        logX = super().sample_field(threshold)

        X = Function(self.V)
        X.vector()[:] = np.exp(logX.vector()[:])

        return X

    def low_memory_generator_sampling(self, 
        eig_gen, μ, V, n_max, scaling=1, log_every=1000):
        
        # total variance
        gen_for_λ = eig_gen(n_max)
        λ_sum = sum(v[0] for v in gen_for_λ)
        # log normal RV mean
        log_normal_μ = Constant(np.log(float(μ)) - λ_sum / 2)
        log_Σ = super().low_memory_generator_sampling(eig_gen,
            log_normal_μ, V, n_max, scaling, log_every)

        # convert normal var to log-normal
        Σ = Function(V)
        Σ.vector()[:] = np.exp(log_Σ.vector()[:])
        return Σ

    def load_field(matrix_vals, μ, V, mesh, n_max):
        """
        Load in parameters λ and φ

        Abstracted function, returns an instance of RandomGaussianField
        """

        # build blank field
        X = RandomLogNormalField()

        # set values
        X.μ = μ
        X.λ_vals = matrix_vals[0, :]
        X.λ_vals[X.λ_vals < 0] = 0
        X.φ_vals = matrix_vals[1:, :]
        X.V = V
        X.mesh = mesh
        X.n_max = n_max

        return X

    def from_eigen_expansion(eigen_gen, μ, V, n_max, scaling=1, log_every=1000):
        X_gaus = RandomGaussianField.from_eigen_expansion(
            eigen_gen, Constant(0), V, n_max, scaling=scaling, 
            log_every=log_every)

        # collect the sum of eigenvalues
        eigenval_sum = sum(X_gaus.λ_vals)
        # correction of mean
        log_normal_μ = Constant(np.log(float(μ)) - eigenval_sum / 2)

        # convert gaussian field to log-normal field
        [m, n] = np.shape(X_gaus.ϕ_vals)
        mtx = np.zeros([m+1, n])
        k = len(X_gaus.λ_vals)
        mtx[0, :k] = X_gaus.λ_vals
        mtx[1:, :] = X_gaus.φ_vals

        X = RandomLogNormalField.load_field(mtx, log_normal_μ, V, V.mesh(), X_gaus.n_max)
        return X

class RandomWeibullField(RandomGaussianField):

    def __init__(self, μ=None, ρ=None, mesh=None, V=None, n_max=10, l=0.05, kappa=1):
        super().__init__(μ, ρ, mesh, V, n_max, l)
        
        self.κ = kappa
        self.scale = gamma(1 + 1/kappa)

    def sample_field(self, threshold=None):
        X1 = super().sample_field(threshold=threshold)
        X2 = super().sample_field(threshold=threshold)

        Y = Function(self.V)
        Y.vector()[:] = np.power(0.5 * (X1.vector()[:]**2 + X2.vector()[:]**2), 1/self.κ)

        Z = self.μ * Y / self.scale
        return Z

    def load_field(matrix_vals, μ, V, mesh, n_max):
        """
        Load in parameters λ and φ

        Abstracted function, returns an instance of RandomGaussianField
        """

        # build blank field
        X = RandomWeibullField()

        # set values
        X.μ = μ
        X.λ_vals = matrix_vals[0, :]
        X.λ_vals[X.λ_vals < 0] = 0
        X.φ_vals = matrix_vals[1:, :]
        X.V = V
        X.mesh = mesh
        X.n_max = n_max

        return X

    def low_memory_generator_sampling(self, eig_gen, μ, V, n_max, κ, log_every=1000):

        # enforce scaling = 1, actual scaling is controlled by κ
        scaling = 1
        # build 2 independent gaussian samples, extract numpy arrays
        Σ_1 = super().low_memory_generator_sampling(eig_gen,
            Constant(0), V, n_max, scaling, log_every).vector()[:]
        Σ_2 = super().low_memory_generator_sampling(eig_gen,
            Constant(0), V, n_max, scaling, log_every).vector()[:]

        # assign as ((S1^2 + S2^2) / 2)^1/κ
        Σ = Function(V)
        Σ.vector()[:] = np.power(0.5 * Σ_1**2 + Σ_2**2, 1/κ)
        # apply mean and scaling
        Σ = Σ * μ / gamma(1 + 1/κ)
        return Σ

    
    
