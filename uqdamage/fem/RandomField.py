"""
RandomField.py - Module for random fields

"""


import numpy as np
from fenics import (
    TrialFunction, TestFunction, Function,
    dx, assemble, PETScMatrix
)
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


# #@jit
# def covariance_matrix_csr(x: list[tuple[float]], cutoff: float, ρ: Callable) -> (list[int], list[int], list[float]):
#     """
#     Helper function for getting CSR values for covariance matrix

#     Input:
#         x : list of 2-tuples of floats
#             each entry is a coordinate pair (x, y) of a point in the mesh
#         cutoff : float > 0
#             cutoff distance.  Points which are further in distance than this
#             are considered so far that the covariance is zero.
#             This is used to enforce sparsity
#         ρ : Callable, signature ρ(x, y) -> float
#             where x, y are 2-tuples of floats
#             This is the covariance kernel.
#     Output:
#         counts, cols, data : CSR values for matrix
#     """
#     # csr matrix values
#     counts = [0]
#     cols = []
#     data = []

#     # size, and counter for row stops
#     n = len(x)
#     counter = 0

#     for i, xs in enumerate(x):
#         for j, ys in enumerate(x):
#             if sqrt((xs[0] - ys[0])**2 + (xs[1] - ys[1])**2) < cutoff:
#                 # points are close enough to include
#                 counter += 1
#                 cols.append(j)
#                 data.append(ρ(xs, ys))

#         # end of row
#         counts.append(counter)

#     return counts, cols, data

class RandomGaussianField:
    """
    Gaussian random field generator for FEniCs

    μ - mean. FEniCs function.  It is assumued that it is constant or μ ∈ V
    ρ - covariance kernel. Python style funcion
    mesh - FEniCs mesh
    V - Scalar function space for the field
    n_max - Number of modes to use in Karhunen-Loeve expansion
    """

    def __init__(self, 
                 mean=None, 
                 kernel=None,
                 eig_gen=None,
                 space=None, 
                 num_modes=0,
                 max_range=np.inf):
        """
        Create a random field object
        
        Input:
            mean : scalar, FEniCS Expression-like, optional
                the mean value of the field, the default is None
            kernel : scalar callable, signature k = k(x, y), optional
                covariance kernel for random field, the default is None
            eig_gen : generator object, optional
                generates both eigenvalues and eigenfunctions 
                (λ, φ), where λ is a scalar and φ is a callable 
                with signature φ = φ(x), the default is None
            space : FEniCS FunctionSpace object, optional
                function space on which the random field lives,
                the default is None
            num_modes : int >= 0, optional
                the number of modes to use, the default is zero
            max_range : float >= 0, optional
                maximum distance overwhich two points can be considered to interact
                this is used to enforce sparsity in the covariance matrix.
                The default is np.inf, in which case no sparsity is enforced
        """

        self.μ = mean
        self.k = kernel
        self.eig_gen = eig_gen
        self.V = space
        self.n_max = num_modes
        self.max_range = max_range

        self.C = None
        self.M = None
        self.λ = None
        self.φ = None

        self.sample_ready = False

    
    def can_sample(self):
        """
        Helper function which checks the given parameters for the field
        this determines whether enough information has been provided
        so that samples can be generated

        Output:
            sampling_is_possible : boolean
                True if enough information is provided so that samples
                can be generated
        """

        # series of guards
        if self.μ is None:
            return False
        if self.k is None and self.eig_gen is None:
            return False
        if self.V is None:
            return False
        if self.n_max <= 1:
            return False
        
        # if we haven't bailed out yet, we should be okay
        return True
    
    def sample_field(self, zero_tol=None):
        """
        Generate a sample of the random field, and return as a function
        on self.V

        Input:
            zero_tol: float, optional
                when generating the covariance operator
                projected onto self.V x self.V, zero_tol
                indicates how small values can be before
                they are considered zero.
                The default is None, in which case no rounding is 
                done

        Output:
            X_sample : FEniCS Function object
                sample of the random field
        """

        # infer the kind of sampling to do based on what has
        # been provided
        if self.can_sample():
            if self.eig_gen is not None:
                self.sample_ready = True
                return self.sample_from_generator()
            return self.sample_from_covariance(zero_tol=zero_tol)
        
        raise RuntimeError("There is not enough provided information to sample from the field!")

    def sample_from_generator(self):
        
        x = self.V.tabulate_dof_coordinates()
        # default vector values are zeros
        Σ = Function(self.V)
        Σ.assign(self.μ)

        # running sum
        vals = np.zeros(len(Σ.vector()[:]))

        # assume φ and vals are large arrays
        # avoid storing 
        for j, (λ, φ) in enumerate(self.eig_gen(self.n_max)):
            ξ = np.random.randn()
            vals += ξ * np.sqrt(np.real(λ)) * np.array(φ(x))

        # add mean-0 field to mean-μ field
        Σ.vector()[:] += vals

        return Σ

    def sample_from_covariance(self, zero_tol=None):
        if not self.sample_ready:
            self.λ, self.φ = self.numerical_eigendecomp()
            self.sample_ready = True
        
        # random numbers, weightings of modes in KL expansion
        ξ_vec = np.random.randn(self.n_max)
        
        # initialize to the mean
        X = Function(self.V)
        X.assign(self.μ)

        #for n in range(self.N-1, self.N-self.n_max-1,-1):
        for n, (ξ, λ, φ) in enumerate(zip(ξ_vec, self.λ, self.φ)):
            if λ < 0:
                logging.warning("Eigenvalue {:d} is negative".format(n))

            X.vector()[:] += np.sqrt(λ) * ξ * φ
        return X

    def numerical_eigendecomp(self):
        """
        Build a numerical eigendecomposition for the provided covariance kernel

        This functions builds and solves the eigenvalue problem
        M C M φ = λ M φ
        where C is the discretization of the covariance kernel,
        and M is the mass matrix on the function space over which the 
        field is defined.

        Output:
            eigenvalues : (n,) array of real floats
                found eigenvalues, eigenvalues[i] = λ_i
            eigenfunctions : (n, m) array of real floats
                collection of eigenfunctions, 
                where eigenfunctions[i, :] is the discretization of the 
                i'th eigenfunction, and m is the dimension of self.V
        """
        # start by building the covariance matrix
        self.C = self.build_covariance_matrix()

        # create mass matrix form
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        Mform = u * v * dx(self.V.mesh())

        # put this form into a PETSc matrix
        MP = PETScMatrix()
        assemble(Mform, tensor=MP) # compute & assign to MP

        # rip out corresponding matrix
        self.M = MP.mat()
        #self.n = self.MP.size(0) # system size, square so size(0) = size(1)
        
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
        val_list = []
        fun_list = []
        logging.info("Extracting Eigenpairs")
        for i in range(min([n_ev, self.n_max])):
            # storage for eigenvectors
            # vectors must be rebuilt so that they are not all the same
            vr, _ = A.getVecs()
            vi, _ = A.getVecs()
            λ = problem.getEigenpair(i, vr, vi)
            # hermitian problem, disregard imaginary part
            val_list.append(np.real(λ))
            fun_list.append(vr.array)

        eigenvalues = np.array(val_list)
        eigenfunctions = np.real(np.array(fun_list))
        return eigenvalues, eigenfunctions

    def build_covariance_matrix(self):
        """
        Build a matrix representation of the covariance operator
        of the field as a matrix

        The assumption is that self.V is a Lagrange FEM space, so
        the matrix can be built by evaluating the kernel pointwise

        Output: 
            C : PETSc matrix
                C_{ij} = self.k(x_i, x_j)
        """

        # DoFs are point evaluations on coordinates
        x = self.V.tabulate_dof_coordinates()
        n_dof = x.shape[0]
        x_indices = np.arange(n_dof)

        logging.info("Building Covariance Matrix CSR Values")
        counts = [0]
        cols = []
        data = []

        counter = 0
        for i, xs in enumerate(x):
            # determine y points which are close enough for interaction
            useable_indexing = np.linalg.norm(x - xs, axis=1) < self.max_range
            # collect these indices and coordinates
            indices = list(x_indices[useable_indexing])
            ys_lst = list(x[useable_indexing, ...])

            # append to CSR data
            counter += np.sum(useable_indexing)
            cols = cols + list(indices)
            data = data + [self.k(xs, ys) for ys in ys_lst]
            counts.append(counter)


        # convert lists to sparse matrix
        logging.info("Assembling PETSc Covariance Matrix")
        C = PETSc.Mat().createAIJWithArrays(
            (n_dof, n_dof), 
            (counts, cols, data)
        )
        return C


    def save_eigendecomp(self, fname_out : str):
        """
        Save the eigendecomposition to a file for later usage
        This is saved using numpy saving

        Input:
            fname_out : string
                name of output file
        """

        if not self.sample_ready:
            self.λ, self.φ = self.numerical_eigendecomp()
            self.sample_ready = True

        n, m = self.φ.shape
        np.savez_compressed(
            fname_out, eigenvalues=self.λ, 
            eigenfunctions=self.φ, 
            num_modes=n
        )

    def load_eigendecomp(self, fname_in : str):
        """
        Read in eigendecomposition from file 

        Input:
            fname_in : string
                name of input file
        """

        data = np.load(fname_in)
        self.λ = data["eigenvalues"]
        self.φ = data["eigenfunctions"]
        self.sample_ready = True


    def get_spectrum(self):
        if not self.sample_ready:
            self.λ, self.φ = self.numerical_eigendecomp()
            self.sample_ready = True
        return self.λ_vals


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

    
    
