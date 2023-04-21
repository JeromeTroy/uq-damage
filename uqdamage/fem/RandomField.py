"""
RandomField.py - Module for random fields

"""


import numpy as np
from fenics import (
    TrialFunction, TestFunction, Function,
    dx, assemble, PETScMatrix, Constant, 
    project
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
from math import sqrt
from typing import Callable

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
        self.max_range = max_range

        self.set_nmax(num_modes)

        self.C = None
        self.M = None
        self.λ = None
        self.φ = None
        
        # hold this value, it may be useful later
        self.trace = 0

        self.sample_ready = False

    def set_nmax(self, num_modes):
        if self.eig_gen is not None:
            self.n_max = num_modes 
        elif num_modes > 0:
            self.n_max = num_modes
        else:
            self.n_max = len(self.V.dofmap().dofs())

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
    
    def sample_field(self):
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
            return self.sample_from_covariance()
        
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

            # update trace value
            self.trace += np.real(λ)

        # add mean-0 field to mean-μ field
        Σ.vector()[:] += vals

        return Σ

    def sample_from_covariance(self):
        if not self.sample_ready:
            self.λ, self.φ = self.numerical_eigendecomp()
            self.sample_ready = True
            self.trace = np.sum(self.λ)
        
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

    """
    Inherits all methods from the above RandomGaussianField
    Overrides sampling method to deal with exp(X(x)) pointwise
    """

    def __init__(self, 
                 mean=None, 
                 **kwargs):
        # initialize with mean-zero gaussian field
        super().__init__(mean=Constant(0), **kwargs)

        self.ln_mean = mean
        self.mult_mean = None

    def sample_field(self):
        logX = super().sample_field()

        # Gaussian Field keeps track of the trace
        if self.mult_mean is None:
            # compute this using formula
            # log Y ~ N(m, C) => E[Y] = exp(m + tr(C) / 2) =: μ (given)
            # so Z ~ N(0, C) => Y = e^m * exp(Z)
            # need factor of e^m = μ * exp(-tr(C) / 2)
            self.mult_mean = self.ln_mean * np.exp(-self.trace / 2)

        X = Function(self.V)
        X.vector()[:] = self.mult_mean * np.exp(logX.vector()[:])

        return X


class RandomWeibullField(RandomGaussianField):
    def __init__(self, 
                 shape_param=1,
                 mean=None,
                 **kwargs):
        """
        Weibull random Field

        Input:
            shape_param : float > 0
                shape parameter for Weibull Marginal
            **kwargs : other arguments for RandomGaussianField() constructor
        """
        # standard initialization
        # Gaussian samples should have mean zero
        super().__init__(mean=Constant(0), **kwargs)
        
        # store mean for later
        self.weibull_μ = mean
        self.κ = shape_param
        self.scale = gamma(1 + 1/shape_param)

    def sample_field(self):
        # see Modeling and estimation of some non Gaussian random fields
        # by Christian Caamaño Carrillo, Ph.D. thesis

        # generate two i.i.d. Gaussian fields
        X1 = super().sample_field()
        X2 = super().sample_field()

        # build Weibull field as described in thesis
        Y = Function(self.V)
        Y.vector()[:] = np.power(
            0.5 * (X1.vector()[:]**2 + X2.vector()[:]**2), 
            1/self.κ
        )

        # apply mean and scaling
        Z = project(self.weibull_μ * Y / self.scale, self.V)

        return Z

    

    
    
