"""
Expanding Ring problem

Assuming an annular domain, expand the inner boundary.
"""

import numpy as np
import logging

from fenics import (
    Expression, DirichletBC, solve, 
    set_log_level
)

from DamageBase import DamageProblem
from Domains2D import AnnularDomain
from LinearElastodynamics import update_fields

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

    def __init__(self, r0, r1, Nr, ρ, E, ν, c, Δt,
        η_m=1e-4, η_k=1e-4, α_m=0.2, α_f=0.4, mesh=None, irreversible=True):

        # construct domain and set boundaries
        self.rm = 0.5 * (r0+r1)

        if mesh is None:
            mesh = AnnularDomain(r0, r1, Nr)

        super().__init__(mesh, ρ, E, ν, Δt, η_m=η_m, η_k=η_k, α_m=α_m, α_f=α_f,irreversible=irreversible)

        # pulling rate
        self.c = c

        # Sub domain for inner ring
        def inner_boundary(x, on_boundary):
            return (x[0]**2 + x[1]**2 < self.rm**2) and on_boundary

        # construct expansion bcs
        self.u_inner = Expression(("c*t*x[0]/sqrt(x[0]*x[0] + x[1]*x[1])",
                            "c*t*x[1]/sqrt(x[0]*x[0] + x[1]*x[1])"),
                    element=self.V.ufl_element(), c = self.c, t = 0)
        self.bcs.append(DirichletBC(self.V, self.u_inner, inner_boundary))

        return None


    def set_inner_bc(self, BC):
        self.bcs = []
        self.bcs.append(BC)


    def set_constant_accel(self, a):
        expr = Expression(("0.5*a*t*t * x[0]/sqrt(x[0]*x[0] + x[1]*x[1])",
                            "0.5*a*t*t * x[1]/sqrt(x[0]*x[0] + x[1]*x[1])"),
                            element=self.V.ufl_element(), a=a, t=0)
        self.u_inner = expr

        # Sub domain for inner ring
        def inner_boundary(x, on_boundary):
            return (x[0]**2 + x[1]**2 < self.rm**2) and on_boundary

        self.set_inner_bc(DirichletBC(self.V, self.u_inner, inner_boundary))

    def set_free_expansion(self):
        self.bcs = []

    def integrate(self, nsteps, nsave, verbose=True, 
        stress_list=None, off_time=None, **save_kwargs):
        """
        Integrate the expanding ring problem over nsteps.  Save to disk in
        the output_file (XDMF format) evrery nsave steps.
        """

        swap_bc_flag = False
        if off_time is not None:
            swap_bc_flag = True
        t_vals = float(self.Δt) * np.arange(nsteps+1)

        self.compute_stresses()
        self.compute_damage()
        self.save_state(t_vals[0], **save_kwargs)

        set_log_level(30)

        for i in range(1,t_vals.size):
            t = t_vals[i]

            if swap_bc_flag:
                if t > off_time:
                    # set to free boundaray condition
                    self.set_boundary_conditions([])
                    swap_bc_flag = False

            # update applied displacement at the G-α intermediate time
            self.u_inner.t = t - float(self.α_f) * float(self.Δt)

            solve(self.Fu==0, self.u, bcs=self.bcs, J = self.Ju,
                form_compiler_parameters={"optimize": True})
            update_fields(self.u, self.u_old, self.v_old, self.a_old,
                self.Δt, self.β, self.γ)
            #self.compute_stresses(stress_list=stress_list)
            self.compute_damage()

            if i%nsave==0:
                logging.info(" save at t = {:g}".format(t))
                # compute stress and damage fields
                # self.compute_stresses()
                # self.compute_damage()
                self.save_state(t, **save_kwargs)

        return None