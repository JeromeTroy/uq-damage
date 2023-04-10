"""
Uniaxial stress problem

Assuming a rectangular domain, Ω, with left, right, top, and bottom 
boundaries, assign the following boundary conditions:
top, bottom : free stress σ_iy = 0 for i = x, y
left : pulled with σ_xx = g_left(t), σ_yx = 0
right : pulled with σ_xx = g_right(t), σ_yx = 0

In addition, define Notched uniaxial problem,
in which the rectangular domain has a triangular
notch cut out of the top boundary.
This is designed to give a point of known failure, 
to concentrate the damage statistics.
"""

import numpy as np
import pandas import DataFrame, concat
from scipy import stats
from scipy.interpolate import CubicSpline
import logging
from datetime import datetime

from fenics import (
    Mesh, SubDomain, near, Measure,
    MeshFunction, dot, solve
)

from DamageBase import DamageProblem
from Domains2D import RectangularDomain, RectangularNotchedDomain
from LinearElastodynamics import update_fields

class UniaxialStress(DamageProblem):
    def __init__(self, length, width, res, ρ, E, ν, Δt, f_left, f_right, 
        η_m=1e-4, η_k=1e-4, α_m=0.2, α_f=0.4, mesh=None, 
        solver_params={'newton_solver': {'linear_solver': 'umfpack'}}):
        """
        Build a UniaxialStress problem

        Input:
            length, width : float > 0
                size of rectangular domain,
                will be centered at origin
            res : int > 0 or 2-tuple of ints
                resolution parameter. Can specify
                different resolutions in both x and y directions
            ρ, E, ν : float > 0
                density, young's modulus, and poisson ratio
            Δt : float > 0
                time step
            f_left, f_right : Fenics Expression (like) object
                forcing terms for the left and right sides 
                of the domain.  These must have a parameter t (as in
                f.t) which corresponds to current evaluation time.
                These are copied, then their copies are passed by 
                reference to the cooresponding forms.
            η_m, η_k : float > 0, optional
                rayleigh damping parameters, these default to 1e-4
            α_m, α_f : float > 0, optional
                time stepping parameters for generalized α method. 
                These default to 0.2 and 0.4 resp.
            mesh : Fenics Mesh object or None, optional
                mesh (if provided) on which the problem is defined.
                The default is None, in which case it is built 
                as a rectangle of length x width, with specified
                resolutions
        """
        
        # maximum values of x and y coordinates (absolute value)
        self.xmax = length / 2
        self.ymax = width / 2

        if mesh is None:
            # generate rectangular domain
            # given length, width, centered at origin

            # type checking resolution parameter, either int, or tuple
            if type(res) == type(1):
                resx, resy = res, res
            else:
                # unpack tuple
                resx, resy = res
            mesh = RectangularDomain(self.xmax, self.ymax, resx, resy)

        # see DamageProblem.__init__
        super().__init__(mesh, ρ, E, ν, Δt, 
            η_m=η_m, η_k=η_k, α_m=α_m, α_f=α_f, irreversible=True, 
            solver_params=solver_params)

        # Boundaries
        # define locations using SubDomains
        class LeftEdge(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], -length/2) and on_boundary
        class RightEdge(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], length/2) and on_boundary
        
        # instatiate these and mark boundaries accordingly
        left_edge = LeftEdge()
        right_edge = RightEdge()
        self.boundaries = MeshFunction(
            "size_t", mesh, mesh.topology().dim() - 1
        )
        self.boundaries.set_all(0)
        left_edge.mark(self.boundaries, 1)
        right_edge.mark(self.boundaries, 2)

        # build surface measure which is aware of these subdomains
        # specific subdomains can be accessed w/ their index number
        # done during marking
        self.ds = Measure("ds", domain=mesh, 
            subdomain_data=self.boundaries)

        # assign α_f in floating point form, for easy use
        # when updating time for forcing functions
        self.α_f_float = α_f

        # assign expressions for forcing
        self.f_left = f_left
        self.f_right = f_right
        
        # RHS for linear solve
        self.rhs = dot(self.u_, self.f_left) * self.ds(1) + \
            dot(self.u_, self.f_right) * self.ds(2)

    def update_forces(self, t):
        """
        Update forcing times
        
        Input:
            t : float > 0
                new time 
        """
        # update t values for forces, 
        # forces are passed by reference, so 
        # the necessary values are updated automatically
        self.f_left.t = t - self.α_f_float * self.Δt
        self.f_right.t = t - self.α_f_float * self.Δt

    def integrate(self, nsteps, nsave, record_times=False, **save_kwargs):
        """
        Solve the problem forward in time

        Input:
            nsteps : int > 0
                number of time steps to take.
                The maximum integration time will be
                self.Δt * nsteps.
            nsave : int > 0
                the number of steps in between savings.
                So if nsave = 2, the problem will save every-other
                time step.
        Optional Key word arguments 
            record_times : boolean
                flag for whether to record and store computation times during
                forward time integration.  The default is False
        Saving key word arguments:
            xdmf_file : main output location (viewed with paraview)
            damage_file : hdf5 file for damage variable  (scalar output)
            stress_file : hdf5 file for stress variable (tensor output)
            strain_file : hdf5 file for strain variable (tensor output)
            velocity_file : hdf5 file for velocity variable (vector output)

            These are fed directly to DamageProblem.save_state()

        """

        t_vals = float(self.Δt) * np.arange(nsteps + 1)

        # ensure stresses & damages are up to date
        self.compute_stresses()
        self.compute_damage()
        
        # initial save
        # reference key word saving arguments for file names
        self.save_state(t_vals[0], **save_kwargs)        

        # don't want to see messages about newton solver converging
        # in 1 step
        # set_log_level(40)

        cpu_times_per_linalg_solve = []

        for i in range(1, nsteps + 1):
            t = t_vals[i]

            # update forcing to current time
            self.update_forces(t)

            # main solve
            logging.debug("Main solve at t = {:g}".format(t))
            # record time
            start_time = datetime.now()
            solve(self.Fu - self.rhs == 0, self.u, bcs=self.bcs,
                J = self.Ju, 
                solver_parameters = self.solver_params,
                form_compiler_parameters={"optimize" : True})

            # get time of the linear algebra solve step
            end_time = datetime.now()
            cpu_times_per_linalg_solve.append((end_time - start_time).total_seconds())
            
            # update fields
            update_fields(self.u, self.u_old, self.v_old, self.a_old, 
                self.Δt, self.β, self.γ)
            # update stresses and damages
            self.compute_stresses()
            self.compute_damage()

            if i % nsave == 0:
                logging.info("Save at t = {:g}, \t {:g} %".format(t, i/nsteps * 100))
                self.save_state(t, **save_kwargs)
        
        # get mean cpu time per timestep
        self._mean_cputime_per_timestep = np.mean(cpu_times_per_linalg_solve)

        if record_times:
            workload_stats = self.workload_summary(show=True)
        else:
            workload_stats = self.workload_summary()
        return workload_stats

class NotchedUniaxialStress(UniaxialStress):

    def __init__(self, length, width, res, ρ, E, ν, Δt, f_left, f_right, 
        notch_length=0.2, notch_width=0.1,
        mesh=None, **kwargs):
        """
        Build a NotchedUniaxialStress problem
        In principle, this could be done with only the UniaxialStress
        problem, and providing a prespecified mesh.
        This class exists to that this can be done in one line,
        rather than having to define the mesh separately.
        However a mesh can be provided.

        Input:
            length, width : float > 0
                size of rectangular domain,
                will be centered at origin
            res : int > 0 or 2-tuple of ints
                resolution parameter. Can specify
                different resolutions in both x and y directions
            ρ, E, ν : float > 0
                density, young's modulus, and poisson ratio
            Δt : float > 0
                time step
            f_left, f_right : Fenics Expression (like) object
                forcing terms for the left and right sides 
                of the domain.  These must have a parameter t (as in
                f.t) which corresponds to current evaluation time.
                These are copied, then their copies are passed by 
                reference to the cooresponding forms.
            notch_length, notch_width : float > 0, optional
                length and width of the notch.  The notch forms
                an isosceles triangle at the top middle of the domain,
                with tip pointed inward.  
                These parameters default  to length = 0.2 and width = 0.1
            mesh : Fenics Mesh object or None, optional
                mesh (if provided) on which the problem is defined.
                The default is None, in which case it is built 
                as a rectangle of length x width, with specified
                resolutions
            **kwargs:
                η_m, η_k : float > 0, optional
                    rayleigh damping parameters, these default to 1e-4
                α_m, α_f : float > 0, optional
                    time stepping parameters for generalized α method. 
                    These default to 0.2 and 0.4 resp.
        """
        if mesh is None:
            # type checking resolution parameter
            if type(res) != type(1):
                # if not integer, is iterable, take its mean
                # generate_mesh() is used to build the mesh,
                # and it only takes a single resolution number
                res = np.mean(res)


            # generate mesh with a notch out of the top
            mesh = RectangularNotchedDomain(length / 2, width / 2, res, 
                notch_length, notch_width)

        # perform standard init, mesh now provided
        super().__init__(length, width, res, ρ, E, ν, Δt, f_left, f_right,
            mesh=mesh, **kwargs)

    @classmethod
    # TODO: override this
    def make_new_problem(cls, instance: DamageProblem, 
        mesh: Mesh, dt: float, X=None):
        """
        Generate a sort of duplicate problem, 
        mimicking a Factory design pattern
        "cls" argument stands for class, this is called as
        instance.make_new_problem(instance, mesh, dt, X)

        Input:
            instance : DamageProblem
                instance of problem to copy
            mesh : Mesh object
                new mesh for problem
            dt : float
                new time step
            X : random field for problem, optional.
                The default is None, in which case nothing
                is done with this.
        Output:
            new_problem : DamageProblem instance
                duplicate problem with new mesh, step size, 
                and random data
        """
        args = (mesh, instance.ρ, instance.E, instance.ν, dt)
        kwargs = {
            "η_m" : instance.η_m, "η_k" : instance.η_k, 
            "α_m" : instance.α_m, "α_f" : instance.α_f, 
            "irreversible" : instance.irreversible
        }
        new_problem = cls(*args, **kwargs)
        if X is not None:
            new_problem.set_softening_fields(X, instance.Δσ)
        return new_problem

# analysis tools

# generating stress-strain curves
def interpolate_stress_on_strain(
    ss : DataFrame, strain_grid : np.ndarray, 
    identifiers=None):
    """
    Use interpolation to put all stress-strain curves onto the same
    strain grid

    Input:
        ss : pandas DataFrame object
            data in question - contains at least indices "time", "strain", and "stress"
            all other column names are considered identifiers
        strain_grid : numpy array, shape (n,)
            the new strain values for the resulting grid.
        identifiers : list or None, optional - do not assign on first function call!
            this is used for recursively examining data along multiple identifiers
            e.g. refinement level and seed number
    Output:
        new_ss_data : pandas DataFrame object
            data which is interpolated to the same strain grid.
    """

    if identifiers is None:
        # guard clause first function call
        # determine required identifiers
        identifiers = list(ss.columns)
        identifiers.remove("time")
        identifiers.remove("strain")
        identifiers.remove("stress")
        return interpolate_stress_on_strain(ss, strain_grid, identifiers=identifiers)

    elif len(identifiers) == 0:
        # second guard clause: base case
        # no identifiers given, assume single dataset
        # interpolate stress-strain curve onto given strain grid
        new_stress = CubicSpline(
            np.array(ss["strain"]), np.array(ss["stress"])
        )(strain_grid)

        # assemble new dataset, no longer using time
        new_dataset = DataFrame.from_dict({
            "time" : np.linspace(np.min(ss["time"]), np.max(ss["time"]), len(strain_grid)),
            "strain" : strain_grid,
            "stress" : new_stress
        })
        return new_dataset
        
    # main proceedure: identifiers given, and more than 1 dataset
    identifier_values = [list(set(ss[id])) for id in identifiers]
    
    # apply to first identifier, iterate over possible values
    replacement_dataframes = []
    for index, val in enumerate(identifier_values[0]):
        subdata = ss.query(f"{identifiers[0]} == {val}")
        # new function call will deal with more identifiers
        new_subdata = interpolate_stress_on_strain(subdata, strain_grid, 
                                    identifiers=identifiers[1:])
        # replace the identifier value
        replacement_data = new_subdata.assign(**{
            identifiers[0] : [val] * len(new_subdata["stress"])
        })
        replacement_dataframes.append(replacement_data)

    # concatenate dataframes, they share the same column names
    return concat(replacement_dataframes, axis=0, ignore_index=True)
