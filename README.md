# Matrix-based shape sensitivity analysis for linear strain energy of triangular thin shell elements
Source code of K. Hayashi and R. Mesnil, "Matrix-based shape sensitivity analysis for linear strain energy of triangular thin shell elements"

![Scordelis-Lo_roof](https://github.com/user-attachments/assets/b7fa99fa-64d8-4f07-8e60-03429ec137a9)

![error](https://github.com/user-attachments/assets/235b22e0-78a4-47cc-af58-0a26ca94412c)

![time](https://github.com/user-attachments/assets/fc7cdea5-4d8e-4c52-942e-8600a498c85e)


# Install
The module can be installed through PyPI using the following command on the command prompt window:
```
pip install dktshell
```

# How to use
```
algo = dktshell.DKTAnalysis(sparse=True)
displacement,internal_force,reaction = algo.RunStructuralAnalysis(vert,face_tri,dirichlet,load,thickness=0.25,elastic_modulus=4.32e8,poisson_ratio=0.0) # Use this input for Scordelis-Lo roof example
strain_energy,strain_energy_gradient = algo.StrainEnergy_with_Gradient(vert,face_tri,dirichlet,load,thickness=1,elastic_modulus=1,poisson_ratio=0.25)
```

For more detailed example, please find **example.py** in the github repository.
