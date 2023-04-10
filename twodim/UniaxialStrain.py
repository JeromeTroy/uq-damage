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

from fenics import (
    DirichletBC, solve,
    near, set_log_level
)

from DamageBase import DamageProblem
from Domains2D import RectangularDomain, RectangularNotchedDomain
from LinearElastodynamics import update_fields


class UniaxialStrain(DamageProblem):
    def __init__(self, length, width, res, ρ, E, ν, Δt, f, 
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
            f : Fenics Expression (like) object
                Dirichlet boundary condition at right boundary
                these must have a parameter t (as in
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

        # maximum values of x and y coordinates (absolute values)
        self.xmax = length / 2
        self.ymax = width / 2

        if mesh is None:
            # generate rectangular domain
            # given length, width, centered @ origin

            # type checking resolution parameter
            # can be either int or tuple
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

    def __init__(self, length, width, res, ρ, E, ν, Δt, f,
        notch_length=0.2, notch_height=0.1,
        mesh=None, **kwargs):
        """
        Build a notched version of the Uniaxial Strain problem

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
            f : Fenics Expression (like) object
                Dirichlet boundary condition at right boundary
                these must have a parameter t (as in
                f.t) which corresponds to current evaluation time.
                These are copied, then their copies are passed by 
                reference to the cooresponding forms.
            notch_length, notch_width : float > 0, optional
                length and width of notch.  
                The notch is an isosceles triangle placed at the 
                top middle of the domain.
                The default is a length of 0.2 and width of 0.1
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
            # type check resolution parameter
            if type(res) != type(1):
                # not integer, will be iterable, take its mean
                # value fed to generate_mesh()
                res = np.mean(res)
            # generate mesh    
            mesh = RectangularNotchedDomain(length / 2, width / 2, res, 
                notch_length, notch_height)

        # standard setup
        super().__init__(length, width, res, ρ, E, ν, Δt, f, 
            mesh=mesh, **kwargs)