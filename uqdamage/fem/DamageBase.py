import numpy as np
import logging
from abc import ABC as AbstractBaseClass

from fenics import (
    Constant,
    FunctionSpace, VectorFunctionSpace, TensorFunctionSpace, 
    TestFunction, TrialFunction, Function, MeshFunction, 
    Mesh, dx, derivative,
    inner, project, 
    HDF5File
)

import uqdamage.fem.DamageDynamics2D as DD
from uqdamage.fem.LinearElastodynamics import avg_α
import uqdamage.fem.LinearElastodynamics as LE



class DamageProblem(AbstractBaseClass):
    """
    This class encapsulates the common framework and generates the neccessary
    function spaces and fields.

    mesh - Meshed computatational domain
    ρ - Density. Assumed to be a constant scalar.
    E - Young's modulus. Assumed to be a constant scalar.
    ν - Poisson ratio.  Assumed to be a constant scalar.
    Δt - Time step.  Required here as part of the weak form of the time stepper.
    η_m, η_k - Rayleigh damping parameters.
    α_m, α_f - Generalized-α parameters.
    irreversible - Sets whether or not the damage is irreversible or not.
    """

    def __init__(self, 
                 mesh : Mesh, ρ, E, ν, Δt,
                 η_m=1e-4, η_k=1e-4, α_m=0.2, α_f=0.4, irreversible=True, 
                 solver_params={'newton_solver': {'linear_solver': 'umfpack'}}):
        self.mesh = mesh
        self.boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)

        # elasticity constants
        self.E = E
        self.ν = ν
        self.ρ = Constant(ρ)
        self.μ = Constant(E / (2.0*(1.0 + ν)))
        self.λ = Constant(E*ν / ((1.0 + ν)*(1.0 - 2.0*ν)))

        self.Δt = Constant(Δt)

        # Generalized α parameters
        self.α_m = Constant(α_m)
        self.α_f = Constant(α_f)
        self.γ = Constant(0.5+α_f-α_m)
        self.β = Constant((self.γ+0.5)**2/4.)

        # damping parameters
        self.η_m = Constant(η_m)
        self.η_k = Constant(η_k)

        # reversible damage
        self.irreversible = irreversible

        # function spaces
        # Define function space for displacement
        self.V = VectorFunctionSpace(mesh, "CG", 1)
        # Define function space for strains/stresses
        self.Vσ = TensorFunctionSpace(mesh, "DG", 0)
        # Define function space for scalar damage variable
        self.Vω = FunctionSpace(mesh, "DG", 0)
        # Define function space for scalar random field (assumed continuous)
        self.Vθ = FunctionSpace(mesh, "DG", 0)

        # Test and trial functions
        self.du = TrialFunction(self.V)
        self.u_ = TestFunction(self.V)
        # Current (unknown) displacement
        self.u = Function(self.V, name="Displacement")
        self.ω = Function(self.Vω, name="Damage")
        self.ω_p = Function(self.Vω)
        self.σmax = Function(self.Vω, name="Maximum Principal Stress")
        # Fields from previous time step (displacement, velocity, acceleration)
        self.u_old = Function(self.V)
        self.v_old = Function(self.V)
        self.a_old = Function(self.V)
        self.ε = Function(self.Vσ, name="Strain")
        self.σL = Function(self.Vσ, name="Linear Stress")
        self.σN = Function(self.Vσ, name="Nonlinear Stress")

        # field variables
        self.σc = Function(self.Vθ, name="Yield Stress")
        self.Δσ = Function(self.Vθ, name="Softening Length")

        # initialize functions for time stepping
        self.a_new = LE.update_acceleration(self.u, self.u_old, self.v_old,
                                            self.a_old, self.Δt, self.β)
        self.v_new = LE.update_velocity(self.a_new, self.u_old, self.v_old,
                                        self.a_old, self.Δt, self.γ)

        # F - the entire nonlinear variational problem written as = 0, including applied loads
        self.Fu = LE.m_mat(avg_α(self.a_old, self.a_new, self.α_m), self.u_) \
            + LE.c_mat(avg_α(self.v_old, self.v_new, self.α_f), self.u_,
                       μ=self.μ, λ=self.λ, η_m=self.η_m, η_k=self.η_k) \
            + inner((1.0-self.ω) * LE.σL(LE.ϵ(avg_α(self.u_old, self.u, self.α_f)),
                                         μ=self.μ, λ=self.λ), LE.ϵ(self.u_))*dx

        self.Ju = derivative(self.Fu, self.u, self.du)

        # empty boundary conditions by default
        self.bcs = []

        # self.sum_stress = []

        # possible fix for MPI communicator - time checking
        self.current_save_time = None

        # track this for later reporting
        self._mean_cputime_per_timestep = None

        # store solver parameters
        self.solver_params = solver_params

        return None

    def set_softening_fields(self, σc, Δσ):
        """
        Assign the fields for the strain softening model

        σc - Critical yield stress
        Δσ - Softening length scale
        """
        self.σc.assign(σc)
        self.Δσ.assign(Δσ)
        return None

    def zero_fields(self):
        """
        Reset all fields to zero.
        """
        # set fields back to zero
        self.u.assign(Constant((0.0, 0.0)))
        self.ω.assign(Constant(0.0))
        self.u_old.assign(Constant((0.0, 0.0)))
        self.v_old.assign(Constant((0.0, 0.0)))
        self.a_old.assign(Constant((0.0, 0.0)))
        self.σc.assign(Constant(0.0))
        self.Δσ.assign(Constant(0.0))
        # re-initialize functions for time stepping
        self.a_new = LE.update_acceleration(self.u, self.u_old, self.v_old,
                                            self.a_old, self.Δt, self.β)
        self.v_new = LE.update_velocity(self.a_new, self.u_old, self.v_old,
                                        self.a_old, self.Δt, self.γ)
        return None

    def set_boundary_conditions(self, bcs):
        self.bcs = bcs

    def set_initial_conditions(self, u0, v0, a0):
        """
        Set the initial conditions

        Input:
            u0 - initial discplacement
            v0 - initial velocity
            a0 - initial acceleration
        """

        # set all to zero
        self.zero_fields()
        self.u_old.assign(u0)
        self.v_old.assign(v0)
        self.a_old.assign(a0)

    def compute_stresses(self, stress_list=None):
        """
        Compute the linear and hte nonlinear stress fields at the current values of the
        displacement
        """
        self.ε.assign(project(LE.ϵ(self.u), self.Vσ, solver_type="superlu_dist"))
        self.σL.assign(
            project(LE.σL(LE.ϵ(self.u), μ=self.μ, λ=self.λ), self.Vσ, solver_type="superlu_dist"))
        self.σN.assign(project((1.0-self.ω) * self.σL, self.Vσ, solver_type="superlu_dist"))
        if stress_list is not None:
            stress_list.append(project((1.0 - self.ω) * self.σL, solver_type="superlu_dist"))
        # update chronologic total stress
        #self.sum_stress.append(norm(self.σN))
        return None

    def compute_damage(self):
        """
        Compute the current value of the damage field, having computed the stress fields
        """
        if self.irreversible:
            # compute predicted damage field
            self.ω_p.assign(project(DD.ω1(self.σL, self.σc, self.Δσ), self.Vω, solver_type="superlu_dist"))
            # take max of predicted and current
            self.ω.vector()[:] = np.maximum(
                self.ω.vector()[:], self.ω_p.vector()[:])
        else:
            self.ω.assign(project(DD.ω1(self.σL, self.σc, self.Δσ), self.Vω, solver_type="superlu_dist"))

        return None

    def save_state(self, t, xdmf_file=None, damage_file=None, stress_file=None,
                   strain_file=None, velocity_file=None, data_file=None):
        """
        Dump fields to disk at a given time.

        t - time
        xdmf_file - XDMF file that is open and ready for writing
        data_file - HDF5File, that is open and ready for writing
            this stores all the data separately from XDMF file

        Note/Warning:
            XDMF files store data as a VisualizationVector,
            which makes it difficult to extract the specific fields
            for post processing.
            Storing the data in an explicity HDF5File separately
            separates the data by its name, and so is easier to access.
        
        The data_file keyword superseeds the 
            damage_file, stress_file, strain_file, and velocity_file
            arguments, as it saves all this data & more.
            It also works with the load_data function (elow)
        """
        if xdmf_file is not None:
            xdmf_file.write(self.u, t)
            xdmf_file.write(self.ε, t)
            xdmf_file.write(self.σL, t)
            xdmf_file.write(self.σN, t)
            xdmf_file.write(self.ω, t)
            xdmf_file.write(self.σc, t)

        if data_file is not None:
            data_file.write(self.u, "Displacement", t)
            data_file.write(self.ε, "Strain", t)
            data_file.write(self.σL, "Linear Stress", t)
            data_file.write(self.σN, "Stress", t)
            data_file.write(self.ω, "Damage", t)
            data_file.write(self.σc, "Yield Stress", t)

        if damage_file is not None:
            logging.warning("Deprecated parameter, use general data file")
            damage_file.write(self.ω, "Damage", t)
        if stress_file is not None:
            logging.warning("Deprecated parameter, use general data file")
            stress_file.write(self.σN, "Stress", t)
        if strain_file is not None:
            logging.warning("Deprecated parameter, use general data file")
            strain_file.write(self.ε, "Strain", t)
        if velocity_file is not None:
            logging.warning("Deprecated parameter, use general data file")
            velocity_file.write(project(self.v_new, self.V), "Velocity", t)
            
    def workload_summary(self, show=False):
        """
        Print and return a dictionary of a summary of specific workload parameters 
        indicating performance for solvers

        Output:
            workload_summary : dictionary object, with the following keys
                "dofs" -> # degrees of freedom in V (displacement function space)
                "elements" -> # mesh elements / cells
                "nodes" -> # mesh vertices / coordinates
                "mesh_hmin" -> minimum mesh size
                "mesh_hmax" -> maximum mesh size

        """

        workload_summary = {
            "dofs" : len(self.V.dofmap().dofs()),
            "elements" : self.mesh.num_cells(),
            "nodes" : self.mesh.num_vertices(),
            "mesh_hmin" : self.mesh.hmin(),
            "mesh_hmax" : self.mesh.hmax(), 
        }

        if self._mean_cputime_per_timestep is not None:
            workload_summary["cpu_time/timestep"] = self._mean_cputime_per_timestep
        else:
            workload_summary["cpu_time/timestep"] = "N/A"

        if show:
            for key, val in workload_summary.items():
                logging.info("{:s} : {:s}".format(key, str(val)))


        return workload_summary
    
    def integrate(self, nsteps, nsave, record_times=False, **save_kwargs):
        raise NotImplementedError


def load_data(mesh: Mesh, data_fname: str, 
    V: FunctionSpace, dataname: str):
    """
    Utility file for loading in data from a given file
    This is for use in conjunction with DamageBase.save_state(data_file)

    Input:
        mesh : Mesh object
            mesh on which the problem is defined.
            Used for its communicator field: mesh.mpi_comm()
        data_fname : string
            filename of .h5 containing relevant data
        V : FunctionSpace object
            space in which the relavant data live
            Note that VectorFunctionSpace and TensorFunctionSpace
            both inherit from FunctionSpace, and so are valid here too.
        dataname : string
            name of the data field to collect

    Output:
        time : list of floats
            time nodes on which data is collected
        data : list of Function objects
            data collected
    """

    # mpi communicator needed to acces .h5 files
    comm = mesh.mpi_comm()

    time = []
    data = []
    # with statement delete HDF5File after ending
    # closes file as a result
    with HDF5File(comm, data_fname, "r") as fin:
        # determine No. time steps
        data_attr = fin.attributes(dataname)
        nsteps = data_attr["count"]

        # the name of each data entry has step number
        data_str_template = dataname + "/vector_{:d}"

        for i in range(nsteps):
            # collect this timestep
            # given from attribute of data
            data_str = data_str_template.format(i)
            curr_attr = fin.attributes(data_str)
            time.append(curr_attr["timestamp"])

            # collect actual value as a Function object
            value = Function(V)
            fin.read(value, data_str)

            data.append(value)

    return time, data
