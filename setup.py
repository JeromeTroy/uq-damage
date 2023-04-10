import setuptools
from distutils.core import setup
from setuptools import find_packages

# use README as description
from pathlib import Path 
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name = "uqdamage",
    version = "1.0.0",
    packages = find_packages(),
    author = ["Jerome Troy", "Petr Plechac", "Gideon Simpson"],
    author_email = "jrtroy@udel.edu",
    url = "https://github.com/JeromeTroy/uq-damage/tree/package",
    description = "Uncertainty Quantification in Damage Mechanics Models",
    license = "modified BSD",
    long_description = long_description,
    long_description_content_type = "text/markdown"
)