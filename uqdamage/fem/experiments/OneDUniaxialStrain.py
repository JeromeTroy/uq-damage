"""
Uniaxial strain problem in 1D

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
    UnitIntervalMesh, solve, near, 
    set_log_level, DirichletBC, project
)

from uqdamage.fem.LinearElastodynamics import update_fields
from uqdamage.fem.DamageBase import DamageProblem
from uqdamage.fem.DamageDynamics2D import ω1scalar

class OneDUniaxialStrain(DamageProblem):

    def __init__(self, mesh_res, ν, Δt, f, 
                 ρ=1, E=1, α_f=0, α_m=0, **kwargs):
        
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
            mesh_res : int > 1
                number of nodes to use for mesh
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

        mesh = UnitIntervalMesh(mesh_res)

        super().__init__(mesh, ρ, E, ν, Δt, α_f=α_f, α_m=α_m, **kwargs)

        def left_boundary(x, on_boundary):
            return near(x[0], 0) and on_boundary
        def right_boundary(x, on_boundary):
            return near(x[0], 1) and on_boundary
        
        self.u_right = f 
        zero_vec = (0,)
        self.bcs.append(
            DirichletBC(self.V, zero_vec, left_boundary)
        )
        self.bcs.append(
            DirichletBC(self.V, self.u_right, right_boundary)
        )

        self.α_f_float = α_f 

    def update_boundary(self, t):
        """
        Update right boundary condition

        Input:
            t : float > 0
                new time
        """
        self.u_right.t = t - self.α_f_float * self.Δt

    def compute_damage(self):
        
        σ_tmp = project(self.σL[0, 0], self.Vω)
        
        ω_prop_vec = np.array(list(map(
            ω1scalar, 
            σ_tmp.vector()[:], self.σc.vector()[:], self.Δσ.vector()[:]
        )))
        if self.irreversible:
            self.ω_p.vector()[:] = ω_prop_vec
            self.ω.vector()[:] = np.maximum(
                self.ω.vector()[:], self.ω_p.vector()[:]
            )
        else:
            
            self.ω.vector()[:] = ω_prop_vec
        return None

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