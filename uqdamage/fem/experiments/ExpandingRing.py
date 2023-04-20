"""
Expanding Ring problem

Assuming an annular domain, expand the inner boundary.
"""

import numpy as np
import logging
from collections.abc import Iterable

from fenics import (
    Expression, DirichletBC, solve, 
    set_log_level, Mesh, sqrt
)

from uqdamage.fem.DamageBase import DamageProblem
from uqdamage.fem.Domains2D import AnnularDomain
from uqdamage.fem.LinearElastodynamics import update_fields

class RingProblem(DamageProblem):
    """
    Sub-class for the expanding ring problem

    r0 - inner ring radius
    r1 - outer ring radius
    Nr - meshing parameter
    c - Ring expansion speed
    """

    # Sub domain for inner ring
    def inner_boundary(self, x, on_boundary):
        return (x[0]**2 + x[1]**2 < self.rm**2) and on_boundary

    def __init__(self, mesh_params, ν, c, Δt,
                 ρ=1, E=1, α_f=0.4,
                 **kwargs):
        """
        Constructor for expanding ring problem
        The expanding ring starts at rest and is expanded outward along the 
        inner boundary at a constant expansion speed c.  
        The default assumptions are that the ring is nondimensionalized so that
        the outer radius = 1 (cannot be changed unless the mesh is transformed), 
        and that ρ = E = 1.

        Input:
            mesh_params : 2-tuple or Mesh object
                if 2-tuple, is two values (r, h)
                where r in (0, 1) is the inner radius
                and h > 0 is the cell size for the fem discretization.
                If Mesh object provided, this mesh is used as the fem
                discretization
            ν : float in (0, 0.5)
                Poisson ratio
            c : float > 0
                expansion speed 
            Δt : float > 0
                time step size
            ρ : float or Fenics object, optional
                density, the default is 1
            E : float or Fenics object, optional
                young's modulus, the default is 1
            **kwargs : other keyword arguments, see
                uqdamage.fem.DamageBase.DamageProblem.__init__()
        """

        # type checking parameter input for mesh
        if isinstance(mesh_params, Mesh):
            # mesh provided, use it
            mesh = mesh_params

            # get inner and outer radii
            x = mesh.coordinates()
            radii = np.linalg.norm(x, axis=1)
            r0, r1 = min(radii), max(radii)
        
        elif isinstance(mesh_params, Iterable):
            # mesh dimensions specified, use these and build mesh
            r0 = mesh_params[0]
            h = mesh_params[1]
            # assumed by nondimensionalization
            r1 = 1

            mesh = AnnularDomain(r0, h)

        else: 
            raise RuntimeError("Must provide mesh_params as either Mesh object or 2-tuple (r, h)")
        
        super().__init__(mesh, ρ, E, ν, Δt, **kwargs)

        # pulling rate
        self.c = c

        # construct domain and set boundaries
        self.r_mid = 0.5 * (r0 + r1)
        self.r = r0

        # Sub domain for inner ring
        def inner_boundary(x, on_boundary):
            return sqrt(x[0]**2 + x[1]**2) < self.r_mid and on_boundary

        # construct expansion bcs
        self.u_inner = Expression(("c*t*x[0]/r",
                            "c*t*x[1]/r"),
                    element=self.V.ufl_element(), c = self.c, t = 0, r=self.r)
        self.bcs.append(DirichletBC(self.V, self.u_inner, inner_boundary))

        self.α_f_float = α_f 

    # def set_inner_bc(self, BC):
    #     self.bcs = []
    #     self.bcs.append(BC)


    # def set_constant_accel(self, a):
    #     expr = Expression(("0.5*a*t*t * x[0]/sqrt(x[0]*x[0] + x[1]*x[1])",
    #                         "0.5*a*t*t * x[1]/sqrt(x[0]*x[0] + x[1]*x[1])"),
    #                         element=self.V.ufl_element(), a=a, t=0)
    #     self.u_inner = expr

    #     # Sub domain for inner ring
    #     def inner_boundary(x, on_boundary):
    #         return (x[0]**2 + x[1]**2 < self.rm**2) and on_boundary

    #     self.set_inner_bc(DirichletBC(self.V, self.u_inner, inner_boundary))

    # def set_free_expansion(self):
    #     self.bcs = []

    def update_boundary(self, t):
        self.u_inner.t = t - self.α_f_float * self.Δt

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

        # swap_bc_flag = False
        # if off_time is not None:
        #     swap_bc_flag = True
        t_vals = float(self.Δt) * np.arange(nsteps+1)

        self.compute_stresses()
        self.compute_damage()

        self.save_state(t_vals[0], **save_kwargs)

        set_log_level(30)

        for i in range(1, t_vals.size):
            t = t_vals[i]

            # if swap_bc_flag:
            #     if t > off_time:
            #         # set to free boundaray condition
            #         self.set_boundary_conditions([])
            #         swap_bc_flag = False

            # update applied displacement at the G-α intermediate time
            self.update_boundary(t)
            logging.debug(f"Main solve at t = {t}")

            solve(self.Fu == 0, self.u, bcs=self.bcs, J = self.Ju,
                form_compiler_parameters={"optimize": True},
                solver_parameters=self.solver_params)
            
            update_fields(self.u, self.u_old, self.v_old, self.a_old,
                self.Δt, self.β, self.γ)
            self.compute_stresses()
            self.compute_damage()

            if i%nsave==0:
                logging.info(" save at t = {:g}".format(t))
                # compute stress and damage fields
                # self.compute_stresses()
                # self.compute_damage()
                self.save_state(t, **save_kwargs)

        return None
