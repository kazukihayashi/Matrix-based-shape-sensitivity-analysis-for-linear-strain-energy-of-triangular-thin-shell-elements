'''
Source code for
K. Hayashi and R. Mesnil, "Matrix-based shape sensitivity analysis for linear strain energy of triangular thin shell elements".

NOTE:
The input consists of the following files.

(1) vertex.dat
(2) face.dat
(3) fix.dat
(4) load.dat

we can directly use the text copied from "Label" component in Grasshopper.
Note that fix.dat must be represented by an 1D array taking either 0 (free) or 1 (fix).
'''

from io import StringIO
import numpy as np
import dktshell
from time import perf_counter

EXAMPLE = "4-3" # "4-1", "4-2" or "4-3"

with open("vertex.dat", 'r') as file:
    data = file.read()
    cleaned_data = data.replace('{', '').replace('}', '')
    vert = np.genfromtxt(StringIO(cleaned_data), delimiter=',',dtype=float)

with open("face.dat", 'r') as file:
    data = file.read()
    cleaned_data = data.replace('Q{', '').replace('T{', '').replace('}', '')
    cleaned_data_linesplit = cleaned_data.splitlines()
    face = []
    for line in cleaned_data_linesplit:
        face.append([int(e) for e in line.split(';')])
    face = np.array(face,dtype=int)
    face_tri = []
    for f in face:
        if len(f)==3 or (len(f)==4 and (f[2]==f[3] or f[3] is None)):
            face_tri.append(f[0:3])
        elif len(f)==4:
            face_tri.append([f[i] for i in [0,1,2]])
            face_tri.append([f[i] for i in [0,2,3]])
    face_tri = np.array(face_tri,dtype=int)

pin = np.where(np.loadtxt("fix.dat").flatten())[0]
algo = dktshell.DKTAnalysis(sparse=True)

if EXAMPLE == "4-1":
    '''
    (Example 4-1)
    Scordelis-Lo roof
    '''
    dirichlet = [[i,j,0.0] for i in pin for j in [0,2]] # Use this option for Scordelis-Lo roof example

    with open("load.dat", 'r') as file:
        data = file.read()
        cleaned_data = data.replace('{', '').replace('}', '')
        load_xyz = np.genfromtxt(StringIO(cleaned_data), delimiter=',')
    load = np.zeros((vert.shape[0],6))
    load[:,0:3] = load_xyz

    displacement,internal_force,reaction = algo.RunStructuralAnalysis(vert,face_tri,dirichlet,load,thickness=0.25,elastic_modulus=4.32e8,poisson_ratio=0.0) # Use this input for Scordelis-Lo roof example
    print(f"Maximum vertical displacement: {np.min(displacement[:,2])}")

elif EXAMPLE == "4-2":
    '''
    (Example 4-2)
    Check if analytical and numerical gradients of the strain energy are almost the same.
    '''
    dirichlet = [[i,j,0.0] for i in pin for j in range(6)]

    load = np.zeros((vert.shape[0],6))
    load[:,2] = -1.0

    strain_energy,strain_energy_analytical_gradient = algo.StrainEnergy_with_Gradient(vert,face_tri,dirichlet,load,thickness=1,elastic_modulus=1,poisson_ratio=0.25)

    delta = 1e-5 # Tiny interval for computing gradients numerically
    strain_energy_numerical_gradient = np.empty(3*len(vert))
    e = np.empty(3*len(vert))
    for iii in range(3*len(vert)):
        vert_minus = np.copy(vert)
        vert_minus[iii//3,iii%3] -= delta
        d_minus, _, _ = algo.RunStructuralAnalysis(vert_minus,face_tri,dirichlet,load,thickness=1,elastic_modulus=1,poisson_ratio=0.25)
        F_minus = np.sum(d_minus*load)/2
        vert_plus = np.copy(vert)
        vert_plus[iii//3,iii%3] += delta
        d_plus, _, _ = algo.RunStructuralAnalysis(vert_plus,face_tri,dirichlet,load,thickness=1,elastic_modulus=1,poisson_ratio=0.25)
        F_plus = np.sum(d_plus*load)/2
        strain_energy_numerical_gradient[iii] = (F_plus-F_minus)/(2*delta)
    e = np.abs(strain_energy_analytical_gradient-strain_energy_numerical_gradient)/(np.abs(strain_energy_analytical_gradient)+1e-10)
    print(f"Max relative diff: {e.max()}")
    print(f"Mean relative diff: {e.mean()}")
    print(f"Std. Dev. relative diff: {e.std()}")

elif EXAMPLE == "4-3":
    '''
    (Example 4-3)
    Comparison of computational performance.
    '''
    REPETITION = 5

    dirichlet = [[i,j,0.0] for i in pin for j in range(6)]

    load = np.zeros((vert.shape[0],6))
    load[:,2] = -1.0

    time_analytical_gradient = np.empty(REPETITION)
    for i in range(REPETITION):
        t0 = perf_counter()
        strain_energy,strain_energy_analytical_gradient = algo.StrainEnergy_with_Gradient(vert,face_tri,dirichlet,load,thickness=1,elastic_modulus=1,poisson_ratio=0.25)
        t1 = perf_counter()
        time_analytical_gradient[i] = t1-t0
    print(f"<Time for Analytical Gradient Computation [s], Repetition: {REPETITION}>")
    print(f"Max      : {np.max(time_analytical_gradient)}")
    print(f"Mean     : {np.mean(time_analytical_gradient)}")
    print(f"Medean   : {np.median(time_analytical_gradient)}")
    print(f"Min      : {np.min(time_analytical_gradient)}")
    print(f"Std. Dev.: {np.std(time_analytical_gradient)}")

    time_numerical_gradient = np.empty(REPETITION)
    for i in range(REPETITION):
        t0 = perf_counter()
        delta = 1e-5 # Tiny interval for computing gradients numerically
        strain_energy_numerical_gradient = np.empty(3*len(vert))
        for iii in range(3*len(vert)):
            vert_minus = np.copy(vert)
            vert_minus[iii//3,iii%3] -= delta
            d_minus, _, _ = algo.RunStructuralAnalysis(vert_minus,face_tri,dirichlet,load,thickness=1,elastic_modulus=1,poisson_ratio=0.25)
            F_minus = np.sum(d_minus*load)/2
            vert_plus = np.copy(vert)
            vert_plus[iii//3,iii%3] += delta
            d_plus, _, _ = algo.RunStructuralAnalysis(vert_plus,face_tri,dirichlet,load,thickness=1,elastic_modulus=1,poisson_ratio=0.25)
            F_plus = np.sum(d_plus*load)/2
            strain_energy_numerical_gradient[iii] = (F_plus-F_minus)/(2*delta)
        t1 = perf_counter()
        time_numerical_gradient[i] = t1-t0

    print(f"<Time for Numerical Gradient Computation [s], Repetition: {REPETITION}>")
    print(f"Max      : {np.max(time_numerical_gradient)}")
    print(f"Mean     : {np.mean(time_numerical_gradient)}")
    print(f"Medean   : {np.median(time_numerical_gradient)}")
    print(f"Min      : {np.min(time_numerical_gradient)}")
    print(f"Std. Dev.: {np.std(time_numerical_gradient)}")
