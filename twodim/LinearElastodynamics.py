"""
LinearElastodyanmics.py - Module for linear elastodynamics based on

https://comet-fenics.readthedocs.io/en/latest/demo/elastodynamics/demo_elastodynamics.py.html
"""

from fenics import *
import numpy as np


def ϵ(u):
    """
    ϵ - compute the symmetrized strain gradient of the displacement,
    ϵ = 0.5 (∇u + ∇uᵀ)
    """

    return sym(grad(u))


def σL(ϵ, μ = 1.0, λ = 1.0):
    """
    σL - compute the linear stress tensor for a given strain
    ϵ - strain (or a similar symmetric 2 tensor)
    μ, λ - Lamé constants

    In the linear regime case

    σ = 2 μ ϵ + λ tr(ϵ) * I
    """

    d = ϵ.geometric_dimension()

    return 2.0*μ*ϵ + λ*tr(ϵ)*Identity(d)

def m_mat(u, u_, ρ=1.0):
    """
    m_mat - compute the mass matrix

    u - trial displacement
    u_ - test displacement
    ρ - density
    """

    return ρ*dot(u, u_)*dx

def kL_mat(u, u_, μ=1.0, λ=1.0):
    """
    kL_mat - compute the stiffness matrix for the linear stress

    u - trial displacement
    u_ - test displacement
    μ, λ - Lamé constants
    """
    return inner(σL(ϵ(u),μ=μ,λ=λ), ϵ(u_))*dx

def c_mat(u, u_,μ=1.0, λ=1.0, η_m = 0.1, η_k=0.1):
    """
    c_mat - Rayleigh style damping matrix for with linear stress

    u - trial displacement
    u_ - test displacement
    μ, λ - Lamé constants
    η_m, η_k - Rayleigh constants
    """
    return η_m*m_mat(u, u_) + η_k*kL_mat(u, u_, μ=μ, λ=λ)

def update_acceleration(u1, u0, v0, a0, Δt, β):
    """
    update_acceleration - Update the acceleration field in accordance with
    generalized-α method

    u1 - updated displacement
    u0 - old displacement
    v0 - old velocity
    a0 - old acceleration
    Δt - time step
    β - β parameter
    """

    return (u1-u0-Δt*v0)/β/Δt**2 - (1-2*β)/2/β*a0

def update_velocity(a1, u0, v0, a0, Δt, γ):
    """
    update_velocity - Update the velocity field in accordance with
    generalized-α method

    a1 - updated update_acceleration
    u0 - old displacement
    v0 - old velocity
    a0 - old acceleration
    Δt - time step
    γ - γ parameter
    """

    return v0 + Δt*((1-γ)*a0 + γ*a1)

def update_fields(u1, u0, v0, a0, Δt, β, γ):
    """
    update_fields - Update the vector fields after having found the new displacement


    u1 - updated displacement
    u0 - old displacement
    v0 - old velocity
    a0 - old acceleration
    Δt - time step
    β - β parameter
    γ - γ parameter
    """

    # Get underlying vectors
    u1_vec = u1.vector()
    u0_vec = u0.vector()
    v0_vec = v0.vector()
    a0_vec = a0.vector()

    # Update the vector fields
    a1_vec = update_acceleration(u1_vec, u0_vec, v0_vec, a0_vec, Δt, β)
    v1_vec = update_velocity(a1_vec, u0_vec, v0_vec, a0_vec, Δt, γ)

    # Transfer the new fields, (u1,v1,a1) → (u0,v0,a0) for the next time step
    v0.vector()[:] = v1_vec
    a0.vector()[:] = a1_vec
    u0.assign(u1)
    #u0.vector()[:] = u1.vector()
    return None

def avg_α(x0, x1, α):
    """
    avg_α - compute the α average of two vector fields

    x0, x1 - the two vector fields
    α - averaging paramter, 0 ≦ α ≦ 1
    """
    return α*x0 + (1-α)*x1
