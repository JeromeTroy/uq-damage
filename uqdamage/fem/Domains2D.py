"""
Domains2D.py - Module for generating domains to be studied

"""

from fenics import (
    RectangleMesh, Point, 
    Mesh, XDMFFile
)
import numpy as np
from collections.abc import Iterable
import pygmsh
import meshio
import gmsh

def RectangularDomain(Lx, Ly, res):
    """
    RectangularDomain - Construct a rectangular domain of the form
    [-Lx, Lx]×[-Ly, Ly] with a total of Nx nodes in the x coordinate and Ny
    nodes in the y coordinate
    """
    
    if isinstance(res, Iterable):
        Nx, Ny = res
    else:
        Nx = res
        Ny = int(res * Ly / Lx)
    
    domain = RectangleMesh(
        Point(-Lx, -Ly), Point(Lx, Ly), 
        Nx, Ny, "crossed"
    )

    return domain

# def PacmanDomain(r, θ, Nr):
#     """
#     PacmanDomain - Construct a Pacman shaped domain

#     r - radius of the Pacman
#     θ - opening angle is of size 2 * θ
#     """

#     disc = Circle(Point(0,0),r)
#     triangle_vertices = [Point(0,0), Point(-2*r*np.cos(θ),2*r*np.sin(θ)),Point(-2*r*np.cos(θ),-2*r*np.sin(θ))]
#     triangle = Polygon(triangle_vertices)
#     pacman = disc - triangle
#     domain = generate_mesh(disc - triangle, Nr)

#     return domain


def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:, :2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={
                           "name_to_read": [cell_data]})
    return out_mesh

def RectangularNotchedDomain(Lx, Ly, res, nl, nh):
    """
    RectangularNotchedDomain - construct a rectangular domain using gmsh of
    the form [-Lx, Lx] × [-Ly, Ly] with a notch at the upper middle, an
    Isosceles triangle with vertices (-nl/2 * Lx, Ly), (0, Ly - nh * Ly), (nl/2
    * Lx, Ly)

    res specifices the mesh resolution

    """
    # Initialize empty geometry using the build in kernel in GMSH
    geometry = pygmsh.geo.Geometry()
    # Fetch model we would like to add data to
    model = geometry.__enter__()

    points = [model.add_point((-Lx, -Ly, 0), mesh_size=res),
            model.add_point((Lx, -Ly, 0), mesh_size=res),
            model.add_point((Lx, Ly, 0), mesh_size=res),
            model.add_point((nl/2*Lx, Ly, 0), mesh_size= res),
            model.add_point((0, Ly-nh*Ly, 0), mesh_size=res),
            model.add_point((-nl/2 *Lx, Ly, 0), mesh_size=res),
            model.add_point((-Lx, Ly, 0), mesh_size=res)]

    # Add lines between all points creating the notch
    edges = [model.add_line(points[i], points[i+1])
            for i in range(-1, len(points)-1)]
    notch_loop = model.add_curve_loop(edges)
    plane_surface = model.add_plane_surface(notch_loop)
    model.synchronize()

    model.add_physical([plane_surface], "Interior")
    model.add_physical(edges, "Boundary")    

    geometry.generate_mesh(dim=2)

    gmsh.write("_mesh.msh")
    gmsh.clear()
    geometry.__exit__()

    # convert to XDMF
    mesh_from_file = meshio.read("_mesh.msh")
    triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=True)
    meshio.write("_mesh.xdmf", triangle_mesh)

    # read in XDMF mesh
    mesh = Mesh()
    mfile = XDMFFile(mesh.mpi_comm(), "_mesh.xdmf")
    mfile.read(mesh)
    mfile.close()

    return mesh

def AnnularDomain(r, res):
    """
    AnnularDomain - construct a ring domain using gmsh of the form
    ||x|| in (r, 1) with 0 < r < 1.

    Input:
        r : float in (0, 1)
            inner radius for ring
        res : float > 0
            resolution parameter, specifies mesh size
            and used to determine spacing on boundary
    Output:
        mesh : fenics compatible mesh object
    """

    # initialize pygmsh geometry
    geometry = pygmsh.geo.Geometry()
    # fetch model
    model = geometry.__enter__()

    # construct polygonal approximations to the inner and outer rings
    nθ = int(2 * np.pi / res)
    θ = np.linspace(0, 2*np.pi, nθ)
    points_outer = [
        model.add_point((np.cos(t), np.sin(t)), mesh_size=res)
        for t in θ
    ]
    edges_outer = [
        model.add_line(points_outer[i], points_outer[(i+1)%len(points_outer)])
        for i in range(len(points_outer))
    ]
    outer_loop = model.add_curve_loop(edges_outer)

    nθin = int(r * nθ)
    θ = np.linspace(0, 2*np.pi, nθin)
    points_inner = [
        model.add_point((r * np.cos(t), r * np.sin(t)), mesh_size=res)
        for t in θ
    ]
    edges_inner = [
        model.add_line(points_inner[i], points_inner[(i+1)%len(points_inner)])
        for i in range(len(points_inner))
    ]
    inner_loop = model.add_curve_loop(edges_inner)

    # build ring from outer loop with hole at inner loop
    plane_surface = model.add_plane_surface(outer_loop, holes=[inner_loop])
    model.synchronize()

    # create physical representation
    model.add_physical([plane_surface], "Interior")
    model.add_physical(edges_outer + edges_inner, "Boundary")

    # generate mesh, write to file, and cleanup
    geometry.generate_mesh(dim=2)
    gmsh.write("_mesh.msh")
    gmsh.clear()
    geometry.__exit__()

    # convert mesh to fenics compatible version
    mesh_from_file = meshio.read("_mesh.msh")
    triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=True)
    # storing in XDMF allows fenics to read in the mesh
    meshio.write("_mesh.xdmf", triangle_mesh)

    # load mesh into fenics
    mesh = Mesh()
    mfile = XDMFFile(mesh.mpi_comm(), "_mesh.xdmf")
    mfile.read(mesh)
    mfile.close()

    return mesh
