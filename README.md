# uq-damage
Uncertainty Quantification in Damage Mechanics Models

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
$git clone https://github.com/JeromeTroy/uq-damage.git && cd uq-damage
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
