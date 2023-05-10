# uq-damage
Uncertainty Quantification in Damage Mechanics Models

## Directory Structure

The main source files are located in uqdamage/. Some examples and tests can be seen in the tests/ directory.  These also include demonstrations for creating random fields, and solving the various problems. 

The hpc/ directory includes scripts which can be used to generate data.  These can be run on high-performance computing platforms.  Most are structured as to generate a random field based on a specified random seed for reproducibility.  They can be run using 
$python script.py \<seed-number\>

## Installation Instructions

### Installing Dependencies
It is recommended to use the Anaconda installer for Python to install FEniCS and its dependencies.  It is also recommended to use git for cloning this package and tracking any changes.

To install from command line:

$conda install -c conda-forge fenics

$conda install -c conda-forge numpy scipy matplotlib pandas seaborn pathos meshio

$pip install gmsh pygmsh

It is also recommended to install the ipython and jupyter libraries for interactive environments

$conda install -c conda-forge ipython jupyter

### Cloning and Installing uq-damage

$git clone <https://github.com/JeromeTroy/uq-damage.git> && cd uq-damage/

$pip install -e ./

### Requirements and Dependenacies
- numpy
- scipy
- matplotlib
- pandas
- seaborn
- fenics
- meshio
- pathos

### Additional Software

It is recommended to install the Paraview program for viewing resulting data. Paraview can visualize the output XDMF files from the simulations. 
