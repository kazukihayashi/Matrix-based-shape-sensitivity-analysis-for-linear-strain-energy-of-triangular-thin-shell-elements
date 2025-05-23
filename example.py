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

with open("load.dat", 'r') as file:
    data = file.read()
    cleaned_data = data.replace('{', '').replace('}', '')
    load_xyz = np.genfromtxt(StringIO(cleaned_data), delimiter=',')

algo = dktshell.DKTAnalysis(sparse=True)
# dirichlet = [[i,j,0.0] for i in pin for j in range(6)]
dirichlet = [[i,j,0.0] for i in pin for j in [0,2]] # Use this option for Scordelis-Lo roof example
load = np.zeros((vert.shape[0],6))
load[:,0:3] = load_xyz

displacement,internal_force,reaction = algo.RunStructuralAnalysis(vert,face_tri,dirichlet,load,thickness=0.25,elastic_modulus=4.32e8,poisson_ratio=0.0) # Use this input for Scordelis-Lo roof example
strain_energy,strain_energy_gradient = algo.StrainEnergy_with_Gradient(vert,face_tri,dirichlet,load,thickness=1,elastic_modulus=1,poisson_ratio=0.25)
