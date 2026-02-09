# Matrix-based shape sensitivity analysis for linear strain energy of triangular thin shell elements
Source code of K. Hayashi and R. Mesnil, Matrix-based shape sensitivity analysis for linear strain energy of triangular thin shell elements, International Journal for Numerical Methods in Engineering, Vol. 127, No. 3, 2026. [https://doi.org/10.1002/nme.70276](https://doi.org/10.1002/nme.70276)

![Scordelis-Lo_roof](https://github.com/user-attachments/assets/b7fa99fa-64d8-4f07-8e60-03429ec137a9)

![error](https://github.com/user-attachments/assets/9bfa42b6-43c3-4cdf-b2f9-4d3ee3f419e3)

![time](https://github.com/user-attachments/assets/c17c3d08-e644-4951-8b5a-ea3c251cab54)


# Install
The module can be installed through PyPI using the following command on the command prompt window:
```
pip install dktshell
```

# How to use
```
algo = dktshell.DKTAnalysis()
displacement,internal_force,reaction = algo.RunStructuralAnalysis(vert,face_tri,dirichlet,load,thickness=0.25,elastic_modulus=4.32e8,poisson_ratio=0.0) # Use this input for Scordelis-Lo roof example
strain_energy,strain_energy_gradient = algo.StrainEnergy_with_Gradient(vert,face_tri,dirichlet,load,thickness=1,elastic_modulus=1,poisson_ratio=0.25)
```

For more detailed example, please find **example.py** in the GitHub repository.
The input structural models required to run **example.py** are stored in the **structural_models** folder.


# How to cite this project
BibTeX
```
@article{Hayashi2026,
author = {Hayashi, Kazuki and Mesnil, Romain},
title = {Matrix-Based Shape Sensitivity Analysis for Linear Strain Energy of Triangular Thin Shell Elements},
journal = {International Journal for Numerical Methods in Engineering},
volume = {127},
number = {3},
pages = {e70276},
doi = {https://doi.org/10.1002/nme.70276},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.70276},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/nme.70276},
abstract = {This study presents an approach to analytical shape sensitivity analysis of linear strain energy in shell structures modeled using thin shell elements. By using the Kirchhoff-Love plate theory and extending conventional finite element methods, the strain energy variations in shell structures due to shape changes are rigorously analyzed in a discrete manner. Sensitivity formulations are sequentially derived by the chain rule, and the procedure to obtain the derivatives of mechanical and geometric properties related to strain energy is explained step by step, ensuring reproducibility and straightforward implementation. Numerical examples include verification of structural analysis results using a benchmark structure and measurement of efficiency and accuracy of our sensitivity analysis implementation compared to the finite difference method. The results with these examples demonstrate the superiority of explicitly computing gradients using the proposed approach, underscoring its potential to advance the optimal design and structural analysis of shell elements.},
year = {2026}
}
```

# Note
If you run Python program in a directory and import this module for the first time, it will take several minutes for just-in-time (JIT) compilation using numba.
This JIT compilation requires only once, and you can smoothly import this module from the second time.
