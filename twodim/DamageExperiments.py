from DamageBase import *
from UniaxialStrain import *
from UniaxialStress import *
from ExpandingRing import *
from Pacman import *

import pandas as pd
from numpy import isnan
import logging

def get_poisson_ratio(material_name: str, prefix: str = ""):
    """
    With new nondimensionalization, only the poisson ratio 
    changes from material to material.  
    This method collects that information, given the name of the material

    Input:
        prefix : string
            location where the material_parameters.csv file is located 
            relative to current directory
        material_name : string
            name of material. See material_parameters.csv.
    Output:
        ν : poisson ratio for material

    Raises:
        ValueError, is ν is nan
        This is a fail safe to avoid starting a simulation with invalid parameters.
        If this is nan, then the true value needs to be looked up,
        or a different material needs to be chosen.
    """

    logging.info("Loading Poisson Ratio for {:s}".format(material_name))
    df = pd.read_csv(prefix + "material_parameters.csv")

    ν = float(df[df["Material/Group"] == material_name]["Poisson Ratio"])

    if isnan(ν):
        raise ValueError("Poisson Ratio not implemented for material")

    return ν
    
