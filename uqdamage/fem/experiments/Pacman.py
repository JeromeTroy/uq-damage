"""
Pacman problem

Assuming a circular domain, with an arc cut out from it,
extending to the center of the domain, 
expand the outer portion of the domain on the circumference of 
the circle.
"""

import numpy as np
import logging

from fenics import (
    Expression, DirichletBC, set_log_level, solve
)

from uqdamage.fem.DamageBase import DamageProblem
from uqdamage.fem.Domains2D import PacmanDomain
from uqdamage.fem.LinearElastodynamics import update_fields

class PacmanProblem(DamageProblem):
    """
    Sub-class for the expanding ring problem

    r - outer ring radius
    θ - opening angle
    Nr - meshing parameter
    c - Ring expansion speed
    """

    def __init__(self, r, θ, Nr, ρ, E, ν, c, Δt,
        η_m=1e-4, η_k=1e-4, α_m=0.2, α_f=0.4, mesh=None, irreversible=True):

        # construct domain and set boundaries
        self.rm = 0.5 * r

        if mesh is None:
            mesh = PacmanDomain(r, θ, Nr)

        super().__init__(mesh, ρ, E, ν, Δt, η_m=η_m, η_k=η_k, α_m=α_m, α_f=α_f,irreversible=irreversible)

        # pulling rate
        self.c = c

        # Sub domain for inner ring
        def outer_boundary(x, on_boundary):
            return (x[0]**2 + x[1]**2 > self.rm**2) and on_boundary

        # construct expansion bcs
        self.u_outer = Expression(("c*t*x[0]/sqrt(x[0]*x[0] + x[1]*x[1])",
                                "c*t*x[1]/sqrt(x[0]*x[0] + x[1]*x[1])"),
                    element=self.V.ufl_element(), c = self.c, t = 0)
        self.bcs.append(DirichletBC(self.V, self.u_outer, outer_boundary))

        return None

    def integrate(self, nsteps, nsave, **save_kwargs):
        """
        Integrate the expanding ring problem over nsteps.  Save to disk in
        the output_file (XDMF format) evrery nsave steps.
        """

        t_vals = float(self.Δt) * np.arange(nsteps+1)

        self.compute_stresses()
        self.compute_damage()
        self.save_state(t_vals[0], **save_kwargs)

        set_log_level(30)

        for i in range(1,t_vals.size):
            t = t_vals[i]
            # update applied displacement at the G-α intermediate time
            self.u_outer.t = t - float(self.α_f) * float(self.Δt)

            solve(self.Fu==0, self.u, bcs=self.bcs, J = self.Ju,
                form_compiler_parameters={"optimize": True})
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