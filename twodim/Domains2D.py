"""
Domains2D.py - Module for generating domains to be studied

"""
import numpy as np

from fenics import (
    Point, RectangleMesh, Mesh, XDMFFile
)

import pygmsh
import meshio
import gmsh


def RectangularDomain(Lx, Ly, Nx, Ny):
    """
    RectangularDomain - Construct a rectangular domain of the form
    [-Lx, Lx]×[-Ly, Ly] with a total of Nx nodes in the x coordinate and Ny
    nodes in the y coordinate
    """

    domain=RectangleMesh(Point(-Lx, -Ly), Point(Lx, Ly), Nx, Ny, "crossed")

    return domain

def AnnularDomain(r0, r1, Nr):
    """
    AnnularDomain - Construct an annular domain

    r0, r1 - inner and outer radii
    Nr - mesh parameter, larger valeus are more refined
    """

    domain=generate_mesh(Circle(Point(0,0),r1)-Circle(Point(0,0),r0), Nr)

    return domain

def PacmanDomain(r, θ, Nr):
    """
    PacmanDomain - Construct a Pacman shaped domain

    r - radius of the Pacman
    θ - opening angle is of size 2 * θ
    """

    disc = Circle(Point(0,0),r)
    triangle_vertices = [Point(0,0), Point(-2*r*np.cos(θ),2*r*np.sin(θ)),Point(-2*r*np.cos(θ),-2*r*np.sin(θ))]
    triangle = Polygon(triangle_vertices)
    pacman = disc - triangle
    domain = generate_mesh(disc - triangle, Nr)

    return domain


def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:, :2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={
                           "name_to_read": [cell_data]})
    return out_mesh

def RectangularNotchedDomainGMSH(Lx, Ly, res, nl, nh):
    """
    RectangularNotchedDomainGMSH - construct a rectangular domain using gmsh of
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
