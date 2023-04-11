"""
DamageDynamics2D.py - Module for a nonlinear damage model, to be used together
                    with a linear elastodynamics model.

Based on work done by A. Kirschtein.
"""

from fenics import (
    sqrt, conditional, lt
)
import numpy as np


def σmax(σ):
    """
    σmax - Compute the maximum principal stress from a stress like tensor in R²

    σ - the stress field which is assumed to already be an element of the
        tensor function space
    """
    return abs(0.5*(σ[0,0]+σ[1,1]+ sqrt( (σ[0,0]-σ[1,1])**2 + 4.0 * σ[0,1]**2)))

def ω1scalar(σ1, σc, Δσ):
    """
    ω1scalar - Model for damage field variable.  This is 1 - value of Arkadz
               NOTE: this is not currently used in code

    σ1 - maximum principal stress
    Δσ - softening length
    σc - yield stress
    """

    if σ1 < σc:
        return 0.0
    elif (σ1 >= σc) and (σc + Δσ > σ1):
        return 1 - (σc/Δσ) * (σc + Δσ - σ1)/(σ1)
    else:
        return 1.0

def ω1(σ, σc, Δσ):
    """
    ω1 - Model for damage field variable.
         NOTE: This can be used in weak forms

    σ - stress like tensor
    Δσ - softening length
    σc - yield stress

    This produces
    ω = 0                           for σ1 <  σc
        1 - σc/Δσ * (σc+Δσ - σ1)/σ1 for σc ≤ σ1 ≦ σc + Δσ
        1                           for σ1 > σc + Δσ
    where σ1 is hte maximum principal stress
    """
    σ1 = σmax(σ) # compute maximum principal stress
    return conditional(lt(σ1,σc), 0,
            conditional(lt(σ1, σc+Δσ), 1 - (σc/Δσ)*(σc+Δσ - σ1)/σ1, 1))

def σN1(σL, σc, Δσ):
    """
    σN - Nonlinear stress with the above damage field variable.  This is intended
         for use in the weak forms.

    σL - Linear stress like variable
    Δσ - softening length
    σc - yield stress
    """

    return (1-ω1(σL, σc, Δσ)) * σL
