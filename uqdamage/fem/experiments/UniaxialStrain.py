"""
Uniaxial strain problem

Assuming a rectangular domain, Ω, with left, right, top, and bottom
boundaries, assign the following boundary conditions:
top, bottom : free stress σ_iy = 0 for i = x, y
left : fixed with u_i = 0 for i = x, y
right : Dirichlet with u_x = f(t), u_y = 0
"""

import numpy as np
import logging
from datetime import datetime
from collections.abc import Iterable

from fenics import (
    DirichletBC, solve, Mesh,
    near, set_log_level
)

from uqdamage.fem.DamageBase import DamageProblem
from uqdamage.fem.Domains2D import RectangularDomain, RectangularNotchedDomain
from uqdamage.fem.LinearElastodynamics import update_fields


class UniaxialStrain(DamageProblem):
    def __init__(self, mesh_params, ν, Δt, f,
                 ρ=1, E=1, α_f=0.4, **kwargs):
        """
        Build a UniaxialStrain problem
        The uniaxial strain problem starts at rest and is pulled to the right
        with some specified forcing condition.  The left boundary is fixed at 
        zero displacement, and the top/bottom boundaries are stress-free.
        The default assumptions are that the problem is nondimensionalized so that 
        the length of the bar is 1 (this cannot be changed unless the mesh is 
        transformed), and that ρ = E = 1.

        For simplicity, the default rectangle is centered at the origin

        Input:
            mesh_params : 2-tuple or Mesh object
                if 2-tuple, is two values (w, h)
                where w is the width along the free axis (y), 
                and h is either a 2-tuple or integer value and specifies
                the number of subdivisions.  If two values are provided, they
                specify the number of subdivisions in the x and y directions 
                respectively.
                If Mesh object provided, this mesh is used as the fem
                discretization
            ν : float in (0, 0.5)
                Poisson ratio
            Δt : float > 0
                time step
            f : Fenics Expression (like) object
                Dirichlet boundary condition at right boundary
                these must have a parameter t (as in
                f.t) which corresponds to current evaluation time.
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
            # get bounds on coordinates
            self.xmax, self.ymax = np.max(mesh.coordinates(), axis=0)
        
        elif isinstance(mesh_params, Iterable):
            w = mesh_params[0]
            nx = mesh_params[1]
            
            self.xmax = 0.5
            self.ymax = w/2

            if isinstance(nx, Iterable):
                res = nx
            else:
                res = (nx, int(w * nx))

            mesh = RectangularDomain(self.xmax, self.ymax, res)
        else:
            raise RuntimeError("Must provide mesh_params as either Mesh object or 2-tuple (w, Nx)")

        # see DamageProblem.__init__
        super().__init__(mesh, ρ, E, ν, Δt, α_f=α_f, **kwargs)

        # boundaries
        # left side
        def left_boundary(x, on_boundary):
            return near(x[0], -self.xmax) and on_boundary
        # right side
        def right_boundary(x, on_boundary):
            return near(x[0], self.xmax) and on_boundary
        
        # fix left boundary
        zero_vec = (0, 0)
        self.bcs.append(
            DirichletBC(self.V, zero_vec, left_boundary)
        )

        # known displacment on right boundary
        self.u_right = f
        self.bcs.append(
            DirichletBC(self.V, self.u_right, right_boundary)
        )

        # quick reference value of α_f, used in time update of
        # right boundary
        self.α_f_float = α_f


    def update_boundary(self, t):
        """
        Update right boundary condition

        Input:
            t : float > 0
                new time
        """
        self.u_right.t = t - self.α_f_float * self.Δt

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
        Optional Key word arguments (for saving)
            xdmf_file : main output location (viewed with paraview)
            damage_file : hdf5 file for damage variable  (scalar output)
            stress_file : hdf5 file for stress variable (tensor output)
            strain_file : hdf5 file for strain variable (tensor output)
            velocity_file : hdf5 file for velocity variable (vector output)

            These are fed directly to DamageProblem.save_state()

        """

        t_vals = float(self.Δt) * np.arange(nsteps + 1)

        # update stresses and damage
        self.compute_stresses()
        self.compute_damage()

        # initial save
        self.save_state(t_vals[0], **save_kwargs)

        # avoid newton solver messages
        set_log_level(40)

        for i in range(1, nsteps + 1):
            t = t_vals[i]

            # update BC time
            self.update_boundary(t)

            # main solve step
            logging.debug("Main solve at t = {:g}".format(t))
            # record times?
            if record_times: start_time = datetime.now()
            solve(self.Fu == 0, self.u, bcs=self.bcs, 
                J = self.Ju, 
                form_compiler_parameters={"optimize" : True})
            if record_times:
                end_time = datetime.now()
                total_time = (end_time - start_time).total_seconds()
                if (self._mean_cputime_per_timestep is None) or (i == 1):
                    self._mean_cputime_per_timestep = total_time
                else:
                    previous_total_time = self._mean_cputime_per_timestep * (i - 1)
                    self._mean_cputime_per_timestep = (previous_total_time + total_time) / i


            # update fields
            update_fields(self.u, self.u_old, self.v_old, self.a_old, 
                self.Δt, self.β, self.γ)
            # update stresses & damage
            self.compute_stresses()
            self.compute_damage()

            # saving
            if i % nsave == 0:
                logging.info("Save at t = {:g}, \t {:g} %".format(
                    t, i/nsteps * 100
                ))
                self.save_state(t, **save_kwargs)
                
        return None


class NotchedUniaxialStrain(UniaxialStrain):

    def __init__(self, mesh_params, *args, **kwargs):
        """
        Build a NotchedUniaxialStrain problem
        The uniaxial strain problem starts at rest and is pulled to the right
        with some specified forcing condition.  The left boundary is fixed at 
        zero displacement, and the top/bottom boundaries are stress-free.
        The default assumptions are that the problem is nondimensionalized so that 
        the length of the bar is 1 (this cannot be changed unless the mesh is 
        transformed), and that ρ = E = 1.
        This has the addition of a notch in the middle of the top boundary
        in the shape of an isoceles triangle

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
            f : Fenics Expression (like) object
                Dirichlet boundary condition at right boundary
                these must have a parameter t (as in
                f.t) which corresponds to current evaluation time.
                These are copied, then their copies are passed by 
                reference to the cooresponding forms.
            ρ : float or Fenics object, optional
                density, the default is 1
            E : float or Fenics object, optional
                young's modulus, the default is 1
            **kwargs : other keyword arguments, see
                uqdamage.fem.DamageBase.DamageProblem.__init__()
        """

        # type checking parameter input
        if isinstance(mesh_params, Mesh):
            # given a mesh, pass it on
            super().__init__(mesh_params, *args, **kwargs)
        
        elif isinstance(mesh_params, Iterable):
            # given specifications for mesh
            rect_params = mesh_params[0]
            resolution_params = mesh_params[1]

            # extract dimensions for mesh
            w = rect_params["width"]
            notch_length = rect_params["notch_width"]
            notch_depth = rect_params["notch_depth"]

            # resolution parameters
            if isinstance(resolution_params, Iterable):
                res = sum(resolution_params) / len(resolution_params)
            else:
                res = resolution_params

            # build mesh
            xmax = 0.5
            ymax = w/2
            mesh = RectangularNotchedDomain(xmax, ymax, res, 
                                            notch_length, notch_depth)
            
            # standard initialization with custom mesh
            super().__init__(mesh, *args, **kwargs)
        
        else:
            raise RuntimeError("Must provide mesh_params as either Mesh object or 2-tuple (param_dict, Nx)")