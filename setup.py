from setuptools import setup
from os import path
import re

root_dir = path.abspath(path.dirname(__file__))
package_name = "dktshell"

with open(path.join(root_dir, package_name, '__init__.py')) as f:
    init_text = f.read()
    version = re.search(r'__version__\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)
    license = re.search(r'__license__\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)
    author = re.search(r'__author__\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)
    author_email = re.search(r'__author_email__\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)
    url = re.search(r'__url__\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)

with open("README.md") as f:
    longtext = f.read()

setup(    
     name=package_name,
     keywords='Discrete Kirchhoff triangle (DKT), Shell, Structural analysis, Finite element analysis (FEM), Finite element method (FEM), Mesh',  
     version=version,
     install_requires=[
         "numpy>=1.26",
         "scipy>=1.15",
         "numba>=0.61",
     ],
     python_requires=">=3.12", 
     author=author,
     description="Matrix-based structural analysis module for triangulated thin shells",
     long_description=longtext,
     long_description_content_type="text/markdown",
     packages=[package_name],
     license=license,
     url=url,
)
