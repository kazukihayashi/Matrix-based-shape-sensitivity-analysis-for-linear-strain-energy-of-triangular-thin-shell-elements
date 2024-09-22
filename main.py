'''
Source code of
K. Hayashi and R. Mesnil, "Matrix-based shape sensitivity analysis for strain energy minimization of shells modeled by finite triangular elements".
'''

from io import StringIO
import numpy as np
import MindlinShell
import time

'''
For vertex.dat, face.dat and fix.dat,
we can directly use the text copied from "Label" component in Grasshopper.
Note that fix.dat must be represented by an 1D array taking either 0 (free) or 1 (fix).
'''
with open("vertex.dat", 'r') as file:
    data = file.read()
    cleaned_data = data.replace('{', '').replace('}', '')
    vert = np.genfromtxt(StringIO(cleaned_data), delimiter=',')

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

algo = MindlinShell.MindlinShell(sparse=True)
dirichlet = [[i,j,0.0] for i in pin for j in range(3)]
load = np.zeros((vert.shape[0],6))
load[:,2] = -1

n_repeat = 10
t_a = [time.time()]
for i in range(n_repeat+1):
    s, gs_analytical = algo.StrainEnergy_with_Gradient(vert,face,dirichlet,load)
    t_a.append(time.time())
t_a2 = [t_a[i+1]-t_a[i] for i in range(1,n_repeat)] # The first iteration is discarded to disregard function compilation and bytecode generation
print(f"<Time for computing Analytical gradient ({n_repeat:d} iterations)>")
print(f"Maximum time :{np.max(t_a2)}")
print(f"Average time :{np.mean(t_a2)}")
print(f"Minimum time :{np.min(t_a2)}")
print(f"St.dev. time :{np.std(t_a2)}")

t_b = [time.time()]
eps = 1e-5
algo.RunStructuralAnalysis(vert,face_tri,dirichlet,load) # Disregard function compilation and bytecode generation
for i in range(n_repeat):
    gs_differential = np.empty_like(gs_analytical)
    for i in range(len(vert)):
        for j in range(3):
            v1 = np.copy(vert)
            v1[i,j] += eps
            d1, r = algo.RunStructuralAnalysis(v1,face,dirichlet,load)
            s1 = np.sum(d1*load)/2
            v2 = np.copy(vert)
            v2[i,j] -= eps
            d2, r = algo.RunStructuralAnalysis(v2,face,dirichlet,load)
            s2 = np.sum(d2*load)/2
            gs_differential[3*i+j] = (s1-s2)/(2*eps)
    t_b.append(time.time())
t_b2 = [t_b[i+1]-t_b[i] for i in range(n_repeat)]
print(f"<Time for computing Numerical gradient ({n_repeat:d} iterations)>")
print(f"Maximum time :{np.max(t_b2)}")
print(f"Average time :{np.mean(t_b2)}")
print(f"Minimum time :{np.min(t_b2)}")
print(f"St.dev. time :{np.std(t_b2)}")

print(f"<Difference of analytical and differential gradients>")
relative_err = np.abs((gs_analytical-gs_differential)/gs_differential)
print(f"Max. Error: {relative_err.max()}")
print(f"Ave. Error: {relative_err.mean()}")
print(f"Min. Error: {relative_err.min()}")
print(f"Std. Error: {relative_err.std()}")
