"""
Building stress strain curves from data

This is a module which is long overdue
"""
import numpy as np
import pandas as pd
from enum import Enum   # in python standard library
from fenics import (
    Mesh, FunctionSpace, HDF5File, Function, dx, assemble, 
    Measure, SubDomain, near, MeshFunction, VectorFunctionSpace
)
from DamageBase import load_data

# rather than remember 0 -> no boundary, 1 -> left, 2 -> right
# use an enum, which will allow these to be accessed by keywords
UNIAXIAL_BDRY_ENUM = Enum("Boundary", 
    ["INTERIOR", "LEFT", "RIGHT"], start=0)

def get_components_number(comp: str) -> int:
    """
    Convert a component string to a number

    Input:
        comp : string
            component in question: either "xx", "xy", or "yy"
    Output:
        comp : int
            component in question: "xx" -> 0, "xy" -> 1, "yy" -> 3
    """

    if comp == "xx":
        return 0
    elif comp == "yy":
        return 3
    else:
        return 1

def collect_stress_strain_data(mesh : Mesh, 
    fname : str, 
    Vε : FunctionSpace, Vσ : FunctionSpace, 
    components=("xx", "xx")):
    """
    Collect the array data to build a stress-strain curve

    Input:
        mesh : Mesh object
            mesh on which all data are defined
        fname : string
            filename containing all relevant data
        Vε, Vσ : FunctionSpace objects
            strain and stress spaces resp.
        components : 2-tuple of strings
            components of each tensor to use.
            Each entry must be a string of length 2,
            options for each entry are "xx", "xy", and "yy"
    Output:
        time : numpy array of shape (nt,)
            time nodes
        ε : numpy array of shape (nt,)
            strain values
        σ : numpy array of shape (nt,)
            stress values
    """

    # collect relevant data
    time, ε_func = load_data(mesh, fname, Vε, "Strain")
    _, σ_func = load_data(mesh, fname, Vσ, "Stress")

    # map list of functions to relevant scalar data
    integrator = lambda f, comp: assemble(f.split(True)[comp] * dx(mesh))

    components = list(map(get_components_number, components))
    ε = list(map(
        lambda f: integrator(f, components[0]), ε_func
    ))
    σ = list(map(
        lambda f: integrator(f, components[1]), σ_func
    ))

    time, ε, σ = np.array(time), np.array(ε), np.array(σ)
    return time, ε, σ


def uniaxial_forcing_boundary_measure(mesh: Mesh):
    """
    Create ds which has been segmented according to whether a boundary
    is forced

    Input:
        mesh : Mesh object
            mesh corresponding to domain
    Output:
        UNIAXIAL_BDRY_ENUM : Enum object
            enumerates boundaries
            enum.INTERIOR,LEFT,RIGHT
            give access codes for general boundary,
            left and right boundaries resp.
            Note that INTERIOR, LEFT, RIGHT -> 0, 1, 2
            resp.
            This exists purely so that we don't have to remember the order
        ds_forced : Measure("ds") object
            boundary measure.  It has subdomain data
            corresponding to the boundaries defined in the 
            UNIAXIAL_BDRY_ENUM item.
            These can be accessed via for example
            ds_forced(enum.RIGHT.value) -> 
            ds_forced(2) -> right boundary ds
    """

    # compute ends of domain
    x_max = np.max(mesh.coordinates()[:, 0])
    x_min = np.min(mesh.coordinates()[:, 0])
    # subdomain data for edges where forcing would be
    class LeftEdge(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], x_min) and on_boundary
    class RightEdge(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], x_max) and on_boundary

    # create instances for usage
    left_edge = LeftEdge()
    right_edge = RightEdge()

    # build a function which encapsulates this data
    boundaries = MeshFunction(
        "size_t", mesh, mesh.topology().dim() - 1
    )

    
    # set appropriate values
    boundaries.set_all(UNIAXIAL_BDRY_ENUM.INTERIOR.value)
    left_edge.mark(boundaries, UNIAXIAL_BDRY_ENUM.LEFT.value)
    right_edge.mark(boundaries, UNIAXIAL_BDRY_ENUM.RIGHT.value)

    # build boundary measure, with appropriately labeled data
    ds_forced = Measure("ds", domain=mesh, subdomain_data=boundaries)

    return UNIAXIAL_BDRY_ENUM, ds_forced

def uniaxial_displacement_strain(mesh: Mesh, ds: Measure, 
    V: VectorFunctionSpace, data_fname: str, use_max: bool = False):
    """
    Use the displacement values at the ends of the domain to 
    compute strain

    Input:
        mesh : Mesh object
            mesh on which displacement functions are defined
        ds : Measure("ds") object
            surface measure that has been marked with left & right 
            subdomains, see UNIAXIAL_BDRY_ENUM
        V : VectorFunctionSpace object
            space on which displacement values are defined
        data_fname : string
            file name where displacement data is located
        use_max : boolean, optional
            whether to use the maximum value for displacement 
            to compute the strain. 
            The default is False, in which case the 
            mean displacement on the edges is used

    Output:
        time : list object
            list of time nodes
        ε : list object
            list of computed strain values
    """

    if use_max:
        return uniaxial_max_displacement_strain(mesh, ds, V, data_fname)
    else:
        return uniaxial_mean_displacement_strain(mesh, ds, V, data_fname)

    
def uniaxial_mean_displacement_strain(mesh: Mesh, ds: Measure, 
    V: VectorFunctionSpace, data_fname: str):
    """
    Use the mean displacement values at the ends of the domain to 
    compute strain

    Input:
        mesh : Mesh object
            mesh on which displacement functions are defined
        ds : Measure("ds") object
            surface measure that has been marked with left & right 
            subdomains, see UNIAXIAL_BDRY_ENUM
        V : VectorFunctionSpace object
            space on which displacement values are defined
        data_fname : string
            file name where displacement data is located

    Output:
        time : list object
            list of time nodes
        ε : list object
            list of computed strain values
    """
    # get width of domain
    y_min, y_max = np.min(mesh.coordinates()[:, 1]), np.max(mesh.coordinates()[:, 1])
    width = y_max - y_min

    time, displacement = load_data(mesh, data_fname, V, "Displacement")

    # for each time step, integrate on the left/right sides of the 
    # domain the x-component of displacement
    ds_left = ds(UNIAXIAL_BDRY_ENUM.LEFT.value)
    ds_right = ds(UNIAXIAL_BDRY_ENUM.RIGHT.value)

    # get x-comp
    # note: map() creates a generator-like object
    # it is lazy in evaluation and won't actually do anything until needed
    ux = map(lambda u: u.split(True)[0], displacement)
    # integrate on boundaries, each value in ux has left & right boundary value
    integrals = map(
        lambda u: (assemble(u * ds_left), assemble(u * ds_right)), ux
    )

    # differences for strain
    # (ux_right - ux_left) / domain_width = avg_ε
    # list() converts iterable object to list, forces evaluation
    ε = list(map(
        lambda val: (val[1] - val[0]) / width, integrals
    ))

    return time, ε

def uniaxial_max_displacement_strain(mesh: Mesh, ds: Measure, 
    V: VectorFunctionSpace, data_fname: str):
    """
    Use the maximum displacement values at the ends of the domain to 
    compute strain

    Input:
        mesh : Mesh object
            mesh on which displacement functions are defined
        ds : Measure("ds") object
            surface measure that has been marked with left & right 
            subdomains, see UNIAXIAL_BDRY_ENUM
        V : VectorFunctionSpace object
            space on which displacement values are defined
        data_fname : string
            file name where displacement data is located

    Output:
        time : list object
            list of time nodes
        ε : list object
            list of computed strain values
    """

    # determine bounds of mesh
    x_min = np.min(mesh.coordinates()[:, 0])
    x_max = np.max(mesh.coordinates()[:, 0])

    # determine which dofs correspond to the left & right boundaries
    Vx, Vy = V.split()
    # gather dofs & coords corresponding
    x_dofs = Vx.dofmap().dofs()
    all_coords = V.tabulate_dof_coordinates()

    # filter dofs by whether corresponding coordinate is on boundary
    on_left_bound = lambda dof: near(all_coords[dof, 0], x_min)
    on_right_bound = lambda dof: near(all_coords[dof, 0], x_max)
    left_bound_dofs = np.array(list(filter(on_left_bound, x_dofs)))
    right_bound_dofs = np.array(list(filter(on_right_bound, x_dofs)))

    # load displacement data
    time, displacement = load_data(mesh, data_fname, V, "Displacement")

    # get displacement values on left & right boundaries
    # taking the x-component is taken care of 
    # because left/right_bound_dofs ⊂ x_dofs
    u_left_and_right = map(
        lambda u: (u.vector()[left_bound_dofs], u.vector()[right_bound_dofs]),
        displacement
    )

    # evaluate strains
    ε = list(map(
        lambda ulr: np.max(ulr[1]) - np.min(ulr[0]), u_left_and_right
    ))

    return time, ε
    



