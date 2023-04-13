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
from pandas import DataFrame, concat
from scipy import stats
from scipy.interpolate import CubicSpline
import logging
from datetime import datetime
from collections.abc import Iterable

from fenics import (
    Mesh, SubDomain, near, Measure,
    MeshFunction, dot, solve
)

from uqdamage.fem.DamageBase import DamageProblem
from uqdamage.fem.Domains2D import RectangularDomain, RectangularNotchedDomain
from uqdamage.fem.LinearElastodynamics import update_fields

class UniaxialStress(DamageProblem):
    def __init__(self, mesh_params, ν, Δt, g_left, g_right, 
                 ρ=1, E=1, α_f=0.4, **kwargs):
        """
        Build a UniaxialStress problem
        The uniaxial stress problem starts at rest and is pulled on the left
        and right boundaries with some specified forcing condition.  The top
        and bottom boundaries are stress-free. The default assumptions are 
        that the problem is nondimensionalized so that the length of the bar 
        is 1 (this cannot be changed unless the mesh is transformed), and that 
        ρ = E = 1. 

        For simplicity, the default rectangle is centered at the origin

        Input:
            mesh_params : 2-tuple or Mesh object
                if 2-tuple, is two values (w, nx)
                where w is the width along the free axis (y), 
                and nx is either a 2-tuple or integer value and specifies
                the number of subdivisions.  If two values are provided, they
                specify the number of subdivisions in the x and y directions 
                respectively.
                If Mesh object provided, this mesh is used as the fem
                discretization
            ν : float in (0, 0.5)
                Poisson ratio
            Δt : float > 0
                time step
            g_left, g_right : Fenics Expression (like) object
                Dirichlet boundary condition at right boundary
                these must have a parameter t (as in
                g.t) which corresponds to current evaluation time.
                These are copied, then their copies are passed by 
                reference to the cooresponding forms.
            ρ : float or Fenics object, optional
                density, the default is 1
            E : float or Fenics object, optional
                young's modulus, the default is 1
            **kwargs : other keyword arguments, see
                uqdamage.fem.DamageBase.DamageProblem.__init__()
        """
        
        if isinstance(mesh_params, Mesh):
            mesh = mesh_params
            self.xmax, self.ymax = np.max(mesh.coordinates(), axis=0)
            self.xmin, self.ymin = np.min(mesh.coordinates(), axis=0)

        elif isinstance(mesh_params, Iterable):
            w = mesh_params[0]
            nx = mesh_params[1]

            self.xmax = 0.5
            self.ymax = w/2
            self.xmin = -self.xmax
            self.ymin = -self.ymax

            if isinstance(nx, Iterable):
                res = nx
            else:
                res = (nx, int(w * nx))
            
            mesh = RectangularDomain(self.xmax, self.ymax, res)
        
        else:
            raise RuntimeError("Must provide mesh_params as either Mesh object or 2-tuple (w, Nx)")
        
        # see DamageProblem.__init__
        super().__init__(mesh, ρ, E, ν, Δt, α_f=α_f, **kwargs)

        # Boundaries
        # define locations using SubDomains
        class LeftEdge(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], self.xmin) and on_boundary
        class RightEdge(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], self.xmax) and on_boundary
        
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
        self.g_left = g_left
        self.g_right = g_right
        
        # RHS for linear solve
        self.rhs = dot(self.u_, self.g_left) * self.ds(1) + \
            dot(self.u_, self.g_right) * self.ds(2)

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
        self.g_left.t = t - self.α_f_float * self.Δt
        self.g_right.t = t - self.α_f_float * self.Δt

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

    def __init__(self, mesh_params, *args, **kwargs):
        """
        Build a NotchedUniaxialStress problem

        The uniaxial stress problem starts at rest and is pulled on the left
        and right boundaries with some specified forcing condition.  The top
        and bottom boundaries are stress-free. The default assumptions are 
        that the problem is nondimensionalized so that the length of the bar 
        is 1 (this cannot be changed unless the mesh is transformed), and that 
        ρ = E = 1. 
        This has the addition of a notch in the middle of the top boundary
        in the shape of an isoceles triangle.

        For simplicity, the default rectangle is centered at the origin
        Input:
            mesh_params : 2-tuple or Mesh object
                if 2-tuple, is two values (param_dict, h)
                param_dict is a dictionary with entries
                    "width" - domain width
                    "notch_width" - width of notch
                    "notch_depth" - depth of notch
                h can either be a list of numbers (hx, hy), in which case 
                we take the mean, or a single number.  The value of h serves as 
                the mesh size used for mesh generation
                If Mesh object provided, this mesh is used as the fem
                discretization
            ν : float in (0, 0.5)
                Poisson ratio
            Δt : float > 0
                time step
            g_left, g_right : Fenics Expression (like) object
                Dirichlet boundary condition at right boundary
                these must have a parameter t (as in
                g.t) which corresponds to current evaluation time.
                These are copied, then their copies are passed by 
                reference to the cooresponding forms.
            ρ : float or Fenics object, optional
                density, the default is 1
            E : float or Fenics object, optional
                young's modulus, the default is 1
            **kwargs : other keyword arguments, see
                uqdamage.fem.DamageBase.DamageProblem.__init__()
        """
        if isinstance(mesh_params, Mesh):
            # given a mesh, pass it on
            super().__init__(mesh_params, *args, **kwargs)
        
        elif isinstance(mesh_params, Iterable):
            rect_params = mesh_params[0]
            resolution_params = mesh_params[1]

            # extract dimensions for mesh
            w = rect_params["width"]
            notch_width = rect_params["notch_width"]
            notch_depth = rect_params["notch_depth"]

            # extract resolution parameters
            if isinstance(resolution_params, Iterable):
                res = sum(resolution_params) / len(resolution_params)
            else:
                res = resolution_params

            # build mesh
            xmax = 0.5
            ymax = w/2
            mesh = RectangularNotchedDomain(xmax, ymax, res, 
                                            notch_width, notch_depth)
            
            # standard initialization with custom mesh
            super().__init__(mesh, *args, **kwargs)

    

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
