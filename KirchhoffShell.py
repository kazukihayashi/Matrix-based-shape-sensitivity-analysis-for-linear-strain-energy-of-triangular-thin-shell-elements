'''
Source code of
K. Hayashi and R. Mesnil (2025):
"Matrix-based shape sensitivity analysis for linear strain energy of triangular thin shell elements".
'''

import numpy as np
import scipy as sp
from numba import njit, f8, i4, b1
from numba.types import Tuple

CACHE = True # Reduce overhead by saving the compiled code. NOTE: It is recommended to set to True to avoid repetitive compilation.
PARALLEL = False # Parallel might be effective only for +1M variables. NOTE: In the original paper this value is fixed to False.
FASTMATH = False # Relax some numerical rigour to gain additional performance. NOTE: it does not improve computational efficiency much, so it is recommended to set this option to False.

@njit(f8[:,:,:](f8[:,:],i4[:,:]),cache=CACHE,parallel=PARALLEL,fastmath=FASTMATH)
def _Coord3D(vert,connectivity):
    '''
    (Input)

    vert[nv,3]<float> : Nodal coordinates.

    connectivity[nm,n_corners_per_member]<int> : Connectivity.

    (Output)

    coord[nm,n_corners_per_member,3]: Nodal coordinate matrices (3D) per face.
    '''
    nm = connectivity.shape[0]
    n_corners_per_member = connectivity.shape[1]
    coord = np.zeros((nm,n_corners_per_member,3))
    for i in range(nm):
        for j in range(n_corners_per_member):
            coord[i,j,:] = vert[connectivity[i,j],:]
    return coord

@njit(f8[:](f8[:,:],i4[:,:]),cache=CACHE,parallel=PARALLEL,fastmath=FASTMATH)
def _Length(vert,edge):
    '''
    (Input)

    vert[nv,3]<float> : Nodal coordinates.

    edge[nm,2]<int> : Connectivity.

    (Output)

    L[nm]: Edge lengths with respect to nodal coordinates.
    '''
    L = np.array([np.sum((vert[edge[i,1]]-vert[edge[i,0]])**2)**0.5 for i in range(edge.shape[0])],dtype=np.float64)

    return L

@njit(f8[:,:](f8[:,:],i4[:,:]),cache=CACHE,parallel=PARALLEL,fastmath=FASTMATH)
def _Grad_Length(vert,edge):
    '''
    (Input)

    vert[nv,3]<float> : Nodal coordinates.

    edge[nm,2]<int> : Connectivity.

    (Output)

    L_g[6,nm]: Gradient of edge lengths with respect to the endpoint coordinates.

    (Description)

    Rows 0,1,2 are the gradients with respect to the x,y,z coordinate of the 1st endpoint.

    Rows 3,4,5 are the gradients with respect to the x,y,z coordinate of the 2nd endpoint.
    '''
    ne = edge.shape[0]
    L = _Length(vert,edge)
    L_g = np.zeros((6,ne),dtype=np.float64)
    for i in range(ne):
        L_g[0:3,i] = -(vert[edge[i,1]]-vert[edge[i,0]])/L[i]
        L_g[3:6,i] = (vert[edge[i,1]]-vert[edge[i,0]])/L[i]

    return L_g

@njit(Tuple((f8[:],f8[:,:]))(f8[:,:],i4[:,:]),cache=CACHE,parallel=PARALLEL,fastmath=FASTMATH)
def _TriangleArea(vert,face):
    '''
    (Input)

    vert[nv,3]<float> : Nodal coordinates.

    face[nf,3]<int> : Connectivity.

    (Output)

    area[nf]<float>: Face areas.

    a[nf,3]<float>: Components for computing face areas, where A = (a[0]^2+a[1]^2+a[2]^3)^(1/2)
    '''
    nf = face.shape[0]
    coord = _Coord3D(vert,face)
    a = np.zeros((nf,3))
    for i in range(3): # corner
        for j in range(3): # x,y,z
            a[:,j] += coord[:,i,(j+1)%3]*coord[:,(i+2)%3,(j+2)%3] - coord[:,i,(j+1)%3]*coord[:,(i+1)%3,(j+2)%3]

    area = np.sqrt(np.sum(a**2,axis=1))/2

    return area, a

@njit(f8[:,:](f8[:,:],i4[:,:]),cache=CACHE,parallel=PARALLEL,fastmath=FASTMATH)
def _Grad_TriangleArea(vert,face):
    '''
    (Input)

    vert[nv,3]<float> : Nodal coordinates.

    face[nf,3]<int> : Connectivity.

    (Output)

    area_g[9,nf]<float>: Gradient of face areas with respect to the corner coordinates.

    (Description)

    Rows 0,1,2 are the gradients with respect to the x,y,z coordinate of the 1st corner.

    Rows 3,4,5 are the gradients with respect to the x,y,z coordinate of the 2nd corner.

    Rows 6,7,8 are the gradients with respect to the x,y,z coordinate of the 3rd corner.
    '''
    nf = face.shape[0]
    coord = _Coord3D(vert,face)
    area, a = _TriangleArea(vert,face)
    area_g = np.zeros((9,nf),dtype=np.float64)
    for i in range(nf):
        for j in range(3): # corner
            for k in range(3): # x,y,z
                area_g[j*3+k,i] = a[i,(k+2)%3]*(coord[i,(j+2)%3,(k+1)%3]-coord[i,(j+1)%3,(k+1)%3]) + a[i,(k+1)%3]*(coord[i,(j+1)%3,(k+2)%3]-coord[i,(j+2)%3,(k+2)%3])
        area_g[:,i] /= 4*area[i]

    return area_g

@njit(f8[:,:,:](f8[:,:],i4[:,:]),cache=CACHE,parallel=PARALLEL,fastmath=FASTMATH)
def _RotationMatrix(vert,face):
    '''
    (Input)

    vert[nv,3]<float> : Nodal coordinates.

    face[nf,3]<int> : Connectivity.

    (Output)

    R[nf,3,3]<float>: 3x3 rotation matrices.

    1st row (R[i,0,:]) is the unit directional vector from the 1st to the 2nd corner.
    
    3rd row (R[i,2,:]) is the unit normal vector.

    2nd row (R[i,1,:]) is the unit in-plane vector orthogonal to the 1st and 3rd vectors.
    '''
    nf = face.shape[0]
    R = np.empty((nf,3,3))
    L1 = _Length(vert,face)
    A, _ = _TriangleArea(vert,face)
    coord = _Coord3D(vert,face)

    for i in range(3): # 1st row, repeat for i = 0,1,2 (x,y,z)
        R[:,0,i] = (coord[:,1,i]-coord[:,0,i])/L1
    for i in range(3): # 3rd row, repeat for i = 0,1,2 (x,y,z)
        R[:,2,i] = ((coord[:,1,(i+1)%3]-coord[:,0,(i+1)%3])*(coord[:,2,(i+2)%3]-coord[:,0,(i+2)%3])-(coord[:,2,(i+1)%3]-coord[:,0,(i+1)%3])*(coord[:,1,(i+2)%3]-coord[:,0,(i+2)%3]))/(2*A)
    for i in range(3): # 2nd row, repeat for i = 0,1,2 (x,y,z)
        R[:,1,i] = R[:,0,(i+2)%3]*R[:,2,(i+1)%3]-R[:,0,(i+1)%3]*R[:,2,(i+2)%3]

    return R

@njit(f8[:,:,:,:](f8[:,:],i4[:,:]),cache=CACHE,parallel=PARALLEL,fastmath=FASTMATH)
def _Grad_RotationMatrix(vert,face):
    '''
    (Input)

    vert[nv,3]<float> : Nodal coordinates.

    face[nf,3]<int> : Connectivity.

    (Output)

    R_g[9,nf,3,3]<float>: Gradient of 3x3 rotation matrices.

    (Description)

    Rows 0,1,2 are the gradients with respect to the x,y,z coordinate of the 1st corner.

    Rows 3,4,5 are the gradients with respect to the x,y,z coordinate of the 2nd corner.

    Rows 6,7,8 are the gradients with respect to the x,y,z coordinate of the 3rd corner.
    '''
    nf = face.shape[0]

    R = _RotationMatrix(vert,face)
    L1 = _Length(vert,face[:,0:2])
    L1_g = _Grad_Length(vert,face)
    L1_g = np.vstack((L1_g,np.zeros((3,L1_g.shape[1]))))
    A, _ = _TriangleArea(vert,face)
    A_g = _Grad_TriangleArea(vert,face)

    R_g = np.zeros((9,nf,3,3),dtype=np.float64)

    ## 1st row
    for i in range(nf): # face
        for j in range(3): # x, y, z
            R_g[j,i,0,j] += -1/L1[i]
            R_g[3+j,i,0,j] += 1/L1[i]
            R_g[:,i,0,j] -= L1_g[:,i]*(vert[face[i,1],j]-vert[face[i,0],j])/(L1[i]**2)           

    ## 3rd row
    for i in range(nf): # face
        for j in range(3): # x, y, z
            for k in range(3): # corner 1, 2, 3
                R_g[k*3+(j+1)%3,i,2,j] += (vert[face[i,(k+1)%3],(j+2)%3]-vert[face[i,(k+2)%3],(j+2)%3])/(2*A[i])
                R_g[k*3+(j+2)%3,i,2,j] += (vert[face[i,(k+2)%3],(j+1)%3]-vert[face[i,(k+1)%3],(j+1)%3])/(2*A[i])
            R_g[:,i,2,j] -= A_g[:,i] * R[i,2,j]/A[i]
            
    ## 2nd row
    for i in range(nf): # face
        for j in range(3): # x, y, z
            # 2nd row
            R_g[:,i,1,j] += R_g[:,i,0,(j+2)%3]*R[i,2,(j+1)%3] + R[i,0,(j+2)%3]*R_g[:,i,2,(j+1)%3]
            R_g[:,i,1,j] -= R_g[:,i,0,(j+1)%3]*R[i,2,(j+2)%3] + R[i,0,(j+1)%3]*R_g[:,i,2,(j+2)%3]

    return R_g

@njit(f8[:,:,:](f8[:,:,:],f8[:,:,:]),cache=CACHE,parallel=PARALLEL,fastmath=FASTMATH)
def _Coord2D(coord3D,R):
    '''
    (Input)

    coord3D[nf,3,3]<float> : Nodal coordinate matrices (3D) per face.

    R[nf,3,3]<float>: 3x3 rotation matrices.

    (Output)

    coord2D[nf,3,2]: Nodal coordinate matrices (2D) per face.
    '''

    nf = coord3D.shape[0]
    coord3D = np.ascontiguousarray(coord3D)
    RT2D = np.ascontiguousarray(R.transpose((0,2,1))[:,:,0:2])

    coord2D = np.zeros((nf,3,2))
    for i in range(nf):
        coord2D[i] = coord3D[i] @ RT2D[i]
    for i in [1,2]:
        coord2D[:,i] -= coord2D[:,0]
    coord2D[:,0] = 0

    return coord2D

@njit(f8[:,:,:,:](f8[:,:,:],f8[:,:,:],f8[:,:,:,:]),cache=CACHE,parallel=PARALLEL,fastmath=FASTMATH)
def _Grad_Coord2D(coord3D,R,R_g):
    '''
    (Input)

    coord3D[nf,3,3]<float> : Nodal coordinate matrices (3D) per face.

    R[nf,3,3]<float>: 3x3 rotation matrices.

    R_g[9,nf,3,3]<float>: Gradient of 3x3 rotation matrices.

    (Output)

    coord2D_g[9,nf,3,2]: Nodal coordinate matrices (2D) per face.

    (Description)

    Rows 0,1,2 are the gradients with respect to the x,y,z coordinate of the 1st corner.

    Rows 3,4,5 are the gradients with respect to the x,y,z coordinate of the 2nd corner.

    Rows 6,7,8 are the gradients with respect to the x,y,z coordinate of the 3rd corner.
    '''

    nf = coord3D.shape[0]
    coord3D = np.ascontiguousarray(coord3D)
    RT2D_g = np.ascontiguousarray(R_g.transpose((0,1,3,2))[:,:,:,0:2])

    coord2D_g = np.zeros((9,nf,3,2))
    for i in range(nf): # face
        for j in range(3): # corner
            for k in range(3): #  x,y,z
                coord2D_g[3*j+k,i,j] += R[i,:2,k]
        for j in range(3): # corner
            for k in range(3): #  x,y,z
                coord2D_g[3*j+k,i] += coord3D[i]@RT2D_g[3*j+k,i]
    for i in [1,2]: # The first corner becomes the origin of local coordinate
        coord2D_g[:,:,i] -= coord2D_g[:,:,0]
    coord2D_g[:,:,0] = 0

    return coord2D_g

@njit(f8[:,:,:](f8[:],f8[:,:,:]),cache=CACHE,parallel=PARALLEL,fastmath=FASTMATH)
def _B_matrix_membrane(A,coord2D):
    '''
    Compute the membrane B-matrix relating the strain (e) and displacement (d) as e = Bd,
    where e is 3-sized vector corresponding to in-plane deformations (2 expansions and 1 shear).
    Suppose the shape function is linear as

    N1 = 1-xi-eta

    N2 = xi

    N3 = eta

    (Input)

    A[nf]<float>: Face areas.

    coord2D[nf,3,2]<float> : Nodal coordinate matrices (2D) per face.

    (Output)

    Bm[nf,3,6]<float>: B-matrices with respect to in-plane displacements.
    '''
    nf = len(A) # number of faces
    A_2 = A*2 # double of triangle area

    b = np.zeros((nf,3)) # N_x
    c = np.zeros((nf,3)) # N_y
    for i in range(nf): # face
        for j in range(3): # corner
            b[i,j] = (coord2D[i,(j+1)%3,1] - coord2D[i,(j+2)%3,1])/A_2[i]
            c[i,j] = (coord2D[i,(j+2)%3,0] - coord2D[i,(j+1)%3,0])/A_2[i]

    Bm = np.zeros((nf,3,6))
    for j in range(3): # corner
        Bm[:,0,2*j] = b[:,j]
        Bm[:,1,2*j+1] = c[:,j]
        Bm[:,2,2*j] = c[:,j]
        Bm[:,2,2*j+1] = b[:,j]
    
    return Bm

@njit(f8[:,:,:,:](f8[:],f8[:,:],f8[:,:,:],f8[:,:,:,:]),cache=CACHE,parallel=PARALLEL,fastmath=FASTMATH)
def _Grad_B_matrix_membrane(A,A_g,coord2D,coord2D_g):
    '''
    (Input)

    A[nf]<float>: Face areas.

    A_g[9,nf]<float> : Gradient of face areas.

    coord2D[nf,3,2]<float> : Nodal coordinate matrices (2D) per face.

    coord2D[nf,3,2]<float> : Gradient of coordinate matrices (2D) per face.

    (Output)

    Bm_g[9,nf,3,6]<float>: Gradient of B-matrices with respect to in-plane displacements.

    (Description)

    Rows 0,1,2 are the gradients with respect to the x,y,z coordinate of the 1st corner.

    Rows 3,4,5 are the gradients with respect to the x,y,z coordinate of the 2nd corner.

    Rows 6,7,8 are the gradients with respect to the x,y,z coordinate of the 3rd corner.
    '''
    nf = len(A)
    ndof_per_face = 9

    Nx_g = np.zeros((ndof_per_face,nf,3))
    Ny_g = np.zeros((ndof_per_face,nf,3))

    for i in range(nf): # face
        for j in range(3): # corner
            Nx_g[:,i,j] += (coord2D_g[:,i,(j+1)%3,1] - coord2D_g[:,i,(j+2)%3,1])/(2*A[i])
            Nx_g[:,i,j] -= (coord2D[i,(j+1)%3,1] - coord2D[i,(j+2)%3,1])*A_g[:,i]/(2*A[i]**2)
            Ny_g[:,i,j] += (coord2D_g[:,i,(j+2)%3,0] - coord2D_g[:,i,(j+1)%3,0])/(2*A[i])
            Ny_g[:,i,j] -= (coord2D[i,(j+2)%3,0] - coord2D[i,(j+1)%3,0])*A_g[:,i]/(2*A[i]**2)

    Bm_g = np.zeros((ndof_per_face,nf,3,6))
    for i in range(nf):
        for j in range(3):
            Bm_g[:,i,0,2*j] += Nx_g[:,i,j]
            Bm_g[:,i,1,2*j+1] += Ny_g[:,i,j]
            Bm_g[:,i,2,2*j] += Ny_g[:,i,j]
            Bm_g[:,i,2,2*j+1] += Nx_g[:,i,j]

    return Bm_g

@njit(f8[:,:,:](f8[:],f8[:,:,:],f8,f8),cache=CACHE,parallel=PARALLEL,fastmath=FASTMATH)
def _B_matrix_bending(A,coord2D,xi,eta):
    '''
    Compute the bending B-matrix relating the strain (e) and displacement (d) as e = Bd,
    where e is 3-sized vector corresponding to out-of-plane translation and 2 out-of-plane bending strain components.
    Suppose the shape function is quadratic as

    N1 = 2*(1-xi-eta)(0.5-xi-eta)

    N2 = xi*(2*xi-1)

    N3 = eta*(2*eta-1)

    N4 = 4*xi*eta

    N5 = 4*eta*(1-xi-eta)

    N6 = 4*xi*(1-xi-eta)

    (Input)

    A[nf]<float> : Face areas.

    coord2D[nf,3,2]<float> : Nodal coordinate matrices (2D) per face.

    xi<float> : Area coordinate (L2).

    eta<float> : Area coordinate (L3).

    (Output)

    Bb[nf,3,9]<float>: B-matrices with respect to out-of-plane displacements (one translation and two rotations per corner).

    (reference)

    Jean-Louis Batoz, Klaus-JÜRgen Bathe, Lee-Wing Ho (1980):
    "A study of three-node triangular plate bending elements",
    International Journal for Numerical Methods in Engineering,
    Volume 15, Issue 12, pp. 1735-1876.
    '''
    nf = len(A)

    x = np.zeros((nf,3))
    y = np.zeros((nf,3))
    for i in range(3): # face edge
        x[:,i] = coord2D[:,(i+2)%3,0] - coord2D[:,(i+1)%3,0]
        y[:,i] = coord2D[:,(i+2)%3,1] - coord2D[:,(i+1)%3,1]

    l2 = x**2 + y**2
    A2 = A*2

    P = -6*x/l2
    q = 3*x*y/l2
    r = 3*y*y/l2
    t = -6*y/l2

    H_x_xi = np.column_stack((
        P[:,2]*(1-2*xi)+(P[:,1]-P[:,2])*eta,
        q[:,2]*(1-2*xi)-(q[:,1]+q[:,2])*eta,
        -4+6*(xi+eta)+r[:,2]*(1-2*xi)-eta*(r[:,1]+r[:,2]),
        -P[:,2]*(1-2*xi)+eta*(P[:,0]+P[:,2]),
        q[:,2]*(1-2*xi)-eta*(q[:,2]-q[:,0]),
        -2+6*xi+r[:,2]*(1-2*xi)+eta*(r[:,0]-r[:,2]),
        -eta*(P[:,1]+P[:,0]),
        eta*(q[:,0]-q[:,1]),
        -eta*(r[:,1]-r[:,0])
    ))

    H_y_xi = np.column_stack((
        t[:,2]*(1-2*xi)+(t[:,1]-t[:,2])*eta,
        1+r[:,2]*(1-2*xi)-(r[:,1]+r[:,2])*eta,
        -q[:,2]*(1-2*xi)+eta*(q[:,1]+q[:,2]),
        -t[:,2]*(1-2*xi)+eta*(t[:,0]+t[:,2]),
        -1+r[:,2]*(1-2*xi)+eta*(r[:,0]-r[:,2]),
        q[:,2]*(1-2*xi)-eta*(q[:,0]-q[:,2]),
        -eta*(t[:,0]+t[:,1]),
        eta*(r[:,0]-r[:,1]),
        -eta*(q[:,0]-q[:,1])
    ))

    H_x_eta = np.column_stack((
        -P[:,1]*(1-2*eta)-xi*(P[:,2]-P[:,1]),
        q[:,1]*(1-2*eta)-xi*(q[:,1]+q[:,2]),
        -4+6*(xi+eta)+r[:,1]*(1-2*eta)-xi*(r[:,1]+r[:,2]),
        xi*(P[:,0]+P[:,2]),
        xi*(q[:,0]-q[:,2]),
        -xi*(r[:,2]-r[:,0]),
        P[:,1]*(1-2*eta)-xi*(P[:,0]+P[:,1]),
        q[:,1]*(1-2*eta)+xi*(q[:,0]-q[:,1]),
        -2+6*eta+r[:,1]*(1-2*eta)+xi*(r[:,0]-r[:,1])
    ))

    H_y_eta = np.column_stack((
        -t[:,1]*(1-2*eta)-xi*(t[:,2]-t[:,1]),
        1+r[:,1]*(1-2*eta)-xi*(r[:,1]+r[:,2]),
        -q[:,1]*(1-2*eta)+xi*(q[:,1]+q[:,2]),
        xi*(t[:,0]+t[:,2]),
        xi*(r[:,0]-r[:,2]),
        -xi*(q[:,0]-q[:,2]),
        t[:,1]*(1-2*eta)-xi*(t[:,0]+t[:,1]),
        -1+r[:,1]*(1-2*eta)+xi*(r[:,0]-r[:,1]),
        -q[:,1]*(1-2*eta)-xi*(q[:,0]-q[:,1])
    ))

    Bb0 = np.empty((nf,3,9), dtype=np.float64)

    for i in range(nf):
        Bb0[i,0,:] = y[i,1] * H_x_xi[i,:] + y[i,2] * H_x_eta[i,:]
        Bb0[i,1,:] = -x[i,1] * H_y_xi[i,:] - x[i,2] * H_y_eta[i,:]
        Bb0[i,2,:] = -x[i,1] * H_x_xi[i,:] - x[i,2] * H_x_eta[i,:] + y[i,1] * H_y_xi[i,:] + y[i,2] * H_y_eta[i,:]

    Bb = Bb0 / A2[:,np.newaxis,np.newaxis]

    return Bb

@njit(f8[:,:,:,:](f8[:],f8[:,:],f8[:,:,:],f8[:,:,:,:],f8,f8),cache=CACHE,parallel=PARALLEL,fastmath=FASTMATH)
def _Grad_B_matrix_bending(A,A_g,coord2D,coord2D_g,xi,eta):
    '''
    (Input)

    A[nf]<float>: Face areas.

    A_g[9,nf]<float> : Gradient of face areas.

    coord2D[nf,3,2]<float> : Nodal coordinate matrices (2D) per face.

    coord2D[nf,3,2]<float> : Gradient of coordinate matrices (2D) per face.

    xi<float> : Area coordinate (L2).

    eta<float> : Area coordinate (L3).

    (Output)

    Bb_g[9,nf,3,9]<float>: Gradient of B-matrices with respect to out-of-plane displacements (one translation and two rotations per corner).
    
    (Description)

    Rows 0,1,2 are the gradients with respect to the x,y,z coordinate of the 1st corner.

    Rows 3,4,5 are the gradients with respect to the x,y,z coordinate of the 2nd corner.

    Rows 6,7,8 are the gradients with respect to the x,y,z coordinate of the 3rd corner.
    '''
    nf = len(A)

    x = np.zeros((nf,3))
    y = np.zeros((nf,3))
    for j in range(3): # face edge
        x[:,j] = coord2D[:,(j+2)%3,0]-coord2D[:,(j+1)%3,0]
        y[:,j] = coord2D[:,(j+2)%3,1]-coord2D[:,(j+1)%3,1]

    x_g = np.zeros((9,nf,3))
    y_g = np.zeros((9,nf,3))
    for i in range(nf): # face
        for j in range(3): # face edge
            x_g[:,i,j] = coord2D_g[:,i,(j+2)%3,0]-coord2D_g[:,i,(j+1)%3,0]
            y_g[:,i,j] = coord2D_g[:,i,(j+2)%3,1]-coord2D_g[:,i,(j+1)%3,1]

    l2 = x**2 + y**2
    l2_g = np.zeros((9,nf,3))
    for i in range(nf): # face
        for j in range(3): # face edge
            l2_g[:,i,j] = 2*x[i,j]*x_g[:,i,j] + 2*y[i,j]*y_g[:,i,j]

    P = -6*x/l2
    q = 3*x*y/l2
    r = 3*y*y/l2
    t = -6*y/l2

    H_x_xi = np.column_stack((
        P[:,2]*(1-2*xi)+(P[:,1]-P[:,2])*eta,
        q[:,2]*(1-2*xi)-(q[:,1]+q[:,2])*eta,
        -4+6*(xi+eta)+r[:,2]*(1-2*xi)-eta*(r[:,1]+r[:,2]),
        -P[:,2]*(1-2*xi)+eta*(P[:,0]+P[:,2]),
        q[:,2]*(1-2*xi)-eta*(q[:,2]-q[:,0]),
        -2+6*xi+r[:,2]*(1-2*xi)+eta*(r[:,0]-r[:,2]),
        -eta*(P[:,1]+P[:,0]),
        eta*(q[:,0]-q[:,1]),
        -eta*(r[:,1]-r[:,0])
    ))

    H_y_xi = np.column_stack((
        t[:,2]*(1-2*xi)+(t[:,1]-t[:,2])*eta,
        1+r[:,2]*(1-2*xi)-(r[:,1]+r[:,2])*eta,
        -q[:,2]*(1-2*xi)+eta*(q[:,1]+q[:,2]),
        -t[:,2]*(1-2*xi)+eta*(t[:,0]+t[:,2]),
        -1+r[:,2]*(1-2*xi)+eta*(r[:,0]-r[:,2]),
        q[:,2]*(1-2*xi)-eta*(q[:,0]-q[:,2]),
        -eta*(t[:,0]+t[:,1]),
        eta*(r[:,0]-r[:,1]),
        -eta*(q[:,0]-q[:,1])
    ))

    H_x_eta = np.column_stack((
        -P[:,1]*(1-2*eta)-xi*(P[:,2]-P[:,1]),
        q[:,1]*(1-2*eta)-xi*(q[:,1]+q[:,2]),
        -4+6*(xi+eta)+r[:,1]*(1-2*eta)-xi*(r[:,1]+r[:,2]),
        xi*(P[:,0]+P[:,2]),
        xi*(q[:,0]-q[:,2]),
        -xi*(r[:,2]-r[:,0]),
        P[:,1]*(1-2*eta)-xi*(P[:,0]+P[:,1]),
        q[:,1]*(1-2*eta)+xi*(q[:,0]-q[:,1]),
        -2+6*eta+r[:,1]*(1-2*eta)+xi*(r[:,0]-r[:,1])
    ))

    H_y_eta = np.column_stack((
        -t[:,1]*(1-2*eta)-xi*(t[:,2]-t[:,1]),
        1+r[:,1]*(1-2*eta)-xi*(r[:,1]+r[:,2]),
        -q[:,1]*(1-2*eta)+xi*(q[:,1]+q[:,2]),
        xi*(t[:,0]+t[:,2]),
        xi*(r[:,0]-r[:,2]),
        -xi*(q[:,0]-q[:,2]),
        t[:,1]*(1-2*eta)-xi*(t[:,0]+t[:,1]),
        -1+r[:,1]*(1-2*eta)+xi*(r[:,0]-r[:,1]),
        -q[:,1]*(1-2*eta)-xi*(q[:,0]-q[:,1])
    ))

    Bb0 = np.empty((nf,3,9), dtype=np.float64)

    for i in range(nf): # face
        Bb0[i,0,:] = y[i,1] * H_x_xi[i,:] + y[i,2] * H_x_eta[i,:]
        Bb0[i,1,:] = -x[i,1] * H_y_xi[i,:] - x[i,2] * H_y_eta[i,:]
        Bb0[i,2,:] = -x[i,1] * H_x_xi[i,:] - x[i,2] * H_x_eta[i,:] + y[i,1] * H_y_xi[i,:] + y[i,2] * H_y_eta[i,:]

    P_g = 6*(x*l2_g-x_g*l2)/(l2*l2)
    q_g = 3*(l2*x*y_g+l2*x_g*y-x*y*l2_g)/(l2*l2)
    r_g = 3*(2*l2*y*y_g-y*y*l2_g)/(l2*l2)
    t_g = 6*(y*l2_g-y_g*l2)/(l2*l2)

    H_x_xi_g = np.dstack((
        P_g[:,:,2]*(1-2*xi)+(P_g[:,:,1]-P_g[:,:,2])*eta,
        q_g[:,:,2]*(1-2*xi)-(q_g[:,:,1]+q_g[:,:,2])*eta,
        r_g[:,:,2]*(1-2*xi)-eta*(r_g[:,:,1]+r_g[:,:,2]),
        -P_g[:,:,2]*(1-2*xi)+eta*(P_g[:,:,0]+P_g[:,:,2]),
        q_g[:,:,2]*(1-2*xi)-eta*(q_g[:,:,2]-q_g[:,:,0]),
        r_g[:,:,2]*(1-2*xi)+eta*(r_g[:,:,0]-r_g[:,:,2]),
        -eta*(P_g[:,:,1]+P_g[:,:,0]),
        eta*(q_g[:,:,0]-q_g[:,:,1]),
        -eta*(r_g[:,:,1]-r_g[:,:,0])
    ))

    H_y_xi_g = np.dstack((
        t_g[:,:,2]*(1-2*xi)+(t_g[:,:,1]-t_g[:,:,2])*eta,
        r_g[:,:,2]*(1-2*xi)-(r_g[:,:,1]+r_g[:,:,2])*eta,
        -q_g[:,:,2]*(1-2*xi)+eta*(q_g[:,:,1]+q_g[:,:,2]),
        -t_g[:,:,2]*(1-2*xi)+eta*(t_g[:,:,0]+t_g[:,:,2]),
        r_g[:,:,2]*(1-2*xi)+eta*(r_g[:,:,0]-r_g[:,:,2]),
        q_g[:,:,2]*(1-2*xi)-eta*(q_g[:,:,0]-q_g[:,:,2]),
        -eta*(t_g[:,:,0]+t_g[:,:,1]),
        eta*(r_g[:,:,0]-r_g[:,:,1]),
        -eta*(q_g[:,:,0]-q_g[:,:,1])
    ))

    H_x_eta_g = np.dstack((
        -P_g[:,:,1]*(1-2*eta)-xi*(P_g[:,:,2]-P_g[:,:,1]),
        q_g[:,:,1]*(1-2*eta)-xi*(q_g[:,:,1]+q_g[:,:,2]),
        r_g[:,:,1]*(1-2*eta)-xi*(r_g[:,:,1]+r_g[:,:,2]),
        xi*(P_g[:,:,0]+P_g[:,:,2]),
        xi*(q_g[:,:,0]-q_g[:,:,2]),
        -xi*(r_g[:,:,2]-r_g[:,:,0]),
        P_g[:,:,1]*(1-2*eta)-xi*(P_g[:,:,0]+P_g[:,:,1]),
        q_g[:,:,1]*(1-2*eta)+xi*(q_g[:,:,0]-q_g[:,:,1]),
        r_g[:,:,1]*(1-2*eta)+xi*(r_g[:,:,0]-r_g[:,:,1])
    ))

    H_y_eta_g = np.dstack((
        -t_g[:,:,1]*(1-2*eta)-xi*(t_g[:,:,2]-t_g[:,:,1]),
        r_g[:,:,1]*(1-2*eta)-xi*(r_g[:,:,1]+r_g[:,:,2]),
        -q_g[:,:,1]*(1-2*eta)+xi*(q_g[:,:,1]+q_g[:,:,2]),
        xi*(t_g[:,:,0]+t_g[:,:,2]),
        xi*(r_g[:,:,0]-r_g[:,:,2]),
        -xi*(q_g[:,:,0]-q_g[:,:,2]),
        t_g[:,:,1]*(1-2*eta)-xi*(t_g[:,:,0]+t_g[:,:,1]),
        r_g[:,:,1]*(1-2*eta)+xi*(r_g[:,:,0]-r_g[:,:,1]),
        -q_g[:,:,1]*(1-2*eta)-xi*(q_g[:,:,0]-q_g[:,:,1])
    ))

    Bb0_g = np.zeros((9,nf,3,9))
    for i in range(nf): # face
        for j in range(9): # dof
            Bb0_g[:,i,0,j] = y[i,1] * H_x_xi_g[:,i,j] + y_g[:,i,1] * H_x_xi[i,j] + y[i,2] * H_x_eta_g[:,i,j] + y_g[:,i,2] * H_x_eta[i,j]
            Bb0_g[:,i,1,j] = -x[i,1] * H_y_xi_g[:,i,j] - x_g[:,i,1] * H_y_xi[i,j] - x[i,2] * H_y_eta_g[:,i,j] - x_g[:,i,2] * H_y_eta[i,j]
            Bb0_g[:,i,2,j] = -x[i,1] * H_x_xi_g[:,i,j] - x_g[:,i,1] * H_x_xi[i,j] - x[i,2] * H_x_eta_g[:,i,j] - x_g[:,i,2] * H_x_eta[i,j] + y[i,1] * H_y_xi_g[:,i,j] + y_g[:,i,1] * H_y_xi[i,j] + y[i,2] * H_y_eta_g[:,i,j] + y_g[:,i,2] * H_y_eta[i,j]
       
    Bb_g = np.zeros((9,nf,3,9))
    for i in range(nf):
        for j in range(3):
            for k in range(9):
                Bb_g[:,i,j,k] = (A[i]*Bb0_g[:,i,j,k]-A_g[:,i]*Bb0[i,j,k])/(2*A[i]*A[i])

    return Bb_g
    

@njit(Tuple((f8[:,:,:],f8[:,:,:]))(f8[:,:],i4[:,:],f8[:],f8,f8),cache=CACHE,parallel=PARALLEL,fastmath=FASTMATH)
def _LocalStiffnessMatrix(vert, face, thickness, E, poisson):
    '''
    (Input)

    vert[nv,3]<float> : Nodal coordinates.

    face[nf,3]<int> : Connectivity.

    thickness[nf]<float> : Thickness of the shell elements.

    E<float> : Young's modulus of the material.

    poisson<float> : Poisson's ratio of the material.

    (Output)

    Kls[nf,18,18]<float>: Local stiffness matrices.

    R[nf,3,3]<float>: Rotation matrices.
    '''
    dof = 6 # DOF per node (x,y,z,rx,ry,rz)
    mdof = 2 # DOF corresponding to membrane forces per node (x,y)
    bdof = 3 # DOF corresponding to bending forces per node (z,rx,ry)
    nf = face.shape[0] # number of faces

    ## Nodal coordinate (3D) matrices per face element
    coord3D = np.ascontiguousarray(_Coord3D(vert,face)) # Nodal coordinate 

    ## Rotation matrices to align the face elements onto the x-y plane
    R = np.ascontiguousarray(_RotationMatrix(vert,face))

    ## Nodal coordinate (2D) matrices per face element
    coord2D = np.ascontiguousarray(_Coord2D(coord3D,R))

    ## Constitutive matrix (strain x, strain y, and shear xy)
    Cb = np.zeros((3,3))
    k1 = E/(1-poisson**2)
    Cb[0,0] = Cb[1,1] = k1
    Cb[0,1] = Cb[1,0] = k1*poisson
    Cb[2,2] = k1 * (1-poisson)/2

    ## Triangle area
    A, _ = _TriangleArea(vert,face)

    ## B-matrix (membrane)
    Bm = np.ascontiguousarray(_B_matrix_membrane(A,coord2D))

    ## B-matrix (bending)
    GipB = np.array([[0.5,0.5],[0.0,0.5],[0.5,0.0]]) # Gauss integration points (xi, eta)
    GiwB = [1.0/3.0,1.0/3.0,1.0/3.0] # weights of Gauss integration points
    Bbs = np.zeros((3,nf,3,9)) # Gauss integration points, faces, strains, displacements (out-of-plane translation + two rotations per corner)
    for i in range(len(GiwB)):
        Bbs[i] = _B_matrix_bending(A,coord2D,GipB[i,0],GipB[i,1])

    ## Initialize local element stiffness matrices
    Kls = np.zeros((nf,18,18))

    for id_face in range(nf):
    
        ## Membrane element
        Kml = thickness[id_face] * A[id_face] * Bm[id_face].T @ Cb @ Bm[id_face]

        ## Bending element
        Kbl = np.zeros((9,9))
        for i in range(len(GipB)):
            Kbl += GiwB[i] * A[id_face] * (thickness[id_face]**3)/12 * Bbs[i,id_face].T @ Cb @ Bbs[i,id_face]

        ## Element stiffness matrix
        Kl = np.zeros((18,18))

        ## Assign membrane stiffness
        for k1 in range(3):
            for k2 in range(3):
                Kl[k1*dof:k1*dof+mdof,k2*dof:k2*dof+mdof] = Kml[k1*mdof:(k1+1)*mdof,k2*mdof:(k2+1)*mdof]

        ## Assign bending stiffness
        for k1 in range(3):
            for k2 in range(3):
                Kl[k1*dof+mdof:k1*dof+mdof+bdof,k2*dof+mdof:k2*dof+mdof+bdof] = Kbl[k1*bdof:(k1+1)*bdof,k2*bdof:(k2+1)*bdof]

        ## Assign fictitious (tiny) drilling stiffness
        for i in range(3):
            f15 = [Kl[k,k] for k in range(dof*i,dof*i+mdof+bdof)]
            f66 = 1e-8 * max(f15)
            Kl[i*dof + mdof + bdof, i*dof + mdof + bdof] = f66

        Kls[id_face] = Kl

    return Kls, R

@njit(Tuple((f8[:,:,:,:],f8[:,:,:,:]))(f8[:,:],i4[:,:],f8[:],f8,f8),cache=CACHE,parallel=PARALLEL,fastmath=FASTMATH)
def _Grad_LocalStiffnessMatrix(vert, face, thickness, E, poisson):
    '''
    (Input)

    vert[nv,3]<float> : Nodal coordinates.

    face[nf,3]<int> : Connectivity.

    thickness[nf]<float> : Thickness of the shell elements.

    E<float> : Young's modulus of the material.

    poisson<float> : Poisson's ratio of the material.

    (Output)

    Kls_g[9,nf,18,18]<float>: Gradient of local stiffness matrices.

    R_g[9,nf,3,3]<float>: Gradient of rotation matrices.

    (Description)

    Rows 0,1,2 are the gradients with respect to the x,y,z coordinate of the 1st corner.

    Rows 3,4,5 are the gradients with respect to the x,y,z coordinate of the 2nd corner.

    Rows 6,7,8 are the gradients with respect to the x,y,z coordinate of the 3rd corner.
    '''
    dof = 6 # dof per node (x,y,z,rx,ry,rz)
    mdof = 2 # dof corresponding to membrane forces per node (x,y)
    bdof = 3 # dof corresponding to bending forces per node (z,rx,ry)
    tri_dof_trans = 9 # translational dofs per triangle face (3(x,y,z) x 3(corners) =9)
    nf = face.shape[0] # number of faces

    ## Nodal coordinate (3D) matrices per face element
    coord3D = np.ascontiguousarray(_Coord3D(vert,face))

    ## Rotation matrices to align the face elements onto the x-y plane
    R = np.ascontiguousarray(_RotationMatrix(vert,face))
    R_g = np.ascontiguousarray(_Grad_RotationMatrix(vert,face))

    ## Nodal coordinate (2D) matrices per face element
    coord2D = np.ascontiguousarray(_Coord2D(coord3D,R))
    coord2D_g = np.ascontiguousarray(_Grad_Coord2D(coord3D,R,R_g))

    ## Constitutive matrix (strain x, strain y, and shear xy)
    Cb = np.zeros((3,3),dtype=np.float64)
    k1 = E/(1-poisson**2)
    Cb[0,0] = Cb[1,1] = k1
    Cb[0,1] = Cb[1,0] = k1*poisson
    Cb[2,2] = k1 * (1-poisson)/2

    ## Triangle area
    A, _ = _TriangleArea(vert,face)
    A_g = _Grad_TriangleArea(vert,face)

    ## B-matrix (membrane)
    Bm = np.ascontiguousarray(_B_matrix_membrane(A,coord2D))
    Bm_g = np.ascontiguousarray(_Grad_B_matrix_membrane(A,A_g,coord2D,coord2D_g))

    ## B-matrix (bending)
    GipB = np.array([[0.5,0.5],[0.0,0.5],[0.5,0.0]]) # Gauss integration points (xi, eta)
    GiwB = [1.0/3.0,1.0/3.0,1.0/3.0] # weights of Gauss integration points
    
    Bbs = np.zeros((3,nf,3,9)) # Gauss integration points, faces, strains, displacements (out-of-plane translation + two rotations per corner)
    for i in range(len(GiwB)):
        Bbs[i] = _B_matrix_bending(A,coord2D,GipB[i,0],GipB[i,1])

    Bbs_g = np.zeros((3,9,nf,3,9)) # Gauss integration points, sensitivity coefficients, faces, strains, displacements (out-of-plane translation + two rotations per corner)
    for i in range(len(GiwB)):
        Bbs_g[i] = _Grad_B_matrix_bending(A,A_g,coord2D,coord2D_g,GipB[i,0],GipB[i,1])

    ## Initialize local stiffness matrices
    Kls_g = np.zeros((tri_dof_trans,nf,18,18),dtype=np.float64)

    for id_face in range(nf):

        ## Membrane element
        Kml_g1 = np.empty((tri_dof_trans,6,6))
        Kml_g1_const_part = thickness[id_face] * Bm[id_face].T @ Cb @ Bm[id_face]
        for id_var in range(tri_dof_trans):
            Kml_g1[id_var] = A_g[id_var,id_face] * Kml_g1_const_part

        Kml_g2 = np.empty((tri_dof_trans,6,6))
        Kml_g2_const_part = thickness[id_face] * A[id_face] * Bm[id_face].T @ Cb
        for id_var in range(tri_dof_trans):
            Kml_g2[id_var] = Kml_g2_const_part @ Bm_g[id_var,id_face]

        Kml_g = Kml_g1 + Kml_g2 + Kml_g2.transpose((0,2,1))

        ## Bending element
        Kbl_g = np.zeros((tri_dof_trans,9,9),dtype=np.float64)

        for i in range(len(GipB)):
            
            Kbl_g1 = np.empty((tri_dof_trans,9,9))
            Kbl_g1_const_part = GiwB[i] * (thickness[id_face]**3)/12 * Bbs[i,id_face].T @ Cb @ Bbs[i,id_face]
            for id_var in range(tri_dof_trans):
                Kbl_g1[id_var] = A_g[id_var,id_face] * Kbl_g1_const_part

            Kbl_g2 = np.empty((tri_dof_trans,9,9))
            Kbl_g2_const_part = GiwB[i] * (thickness[id_face]**3)/12 * A[id_face] * Bbs[i,id_face].T @ Cb
            for id_var in range(tri_dof_trans):
                Kbl_g2[id_var] = Kbl_g2_const_part @ Bbs_g[i,id_var,id_face]

            Kbl_g += Kbl_g1 + Kbl_g2 + Kbl_g2.transpose((0,2,1))

        ## Gradient of element stiffness matrix
        Kl_g = np.zeros((tri_dof_trans,18,18),dtype=np.float64)

        ## Assign membrane stiffness
        for k1 in range(3):
            for k2 in range(3):
                Kl_g[:,k1*dof:k1*dof+mdof,k2*dof:k2*dof+mdof] = Kml_g[:,k1*mdof:(k1+1)*mdof,k2*mdof:(k2+1)*mdof]

        ## Assign bending stiffness
        for k1 in range(3):
            for k2 in range(3):
                Kl_g[:,k1*dof+mdof:k1*dof+mdof+bdof,k2*dof+mdof:k2*dof+mdof+bdof] = Kbl_g[:,k1*bdof:(k1+1)*bdof,k2*bdof:(k2+1)*bdof]

        ## Assign fictitious (tiny) drilling stiffness
        for j in range(id_var):
            for i in range(3):
                f66_g = 1e-8 * np.array([Kl_g[j,k,k] for k in range(dof*i,dof*i+mdof+bdof)]).max()
                Kl_g[j,i*dof + mdof + bdof, i*dof + mdof + bdof] = f66_g

        Kls_g[:,id_face,:,:] = Kl_g

    return Kls_g, R_g

@njit(Tuple((i4[:],i4[:],f8[:]))(i4[:,:],f8[:,:,:],f8[:,:,:],b1),cache=CACHE,parallel=PARALLEL,fastmath=FASTMATH)
def _AssembleLocalStiffnessMatrices(face,Kls,R,compute_index):
    '''
    (Input)

    face[nf,3]<int> : Connectivity.

    Kls[nf,18,18]<float>: Local stiffness matrices.

    R[nf,3,3]<float>: Rotation matrices.

    compute_index<bool>: Compute row and col if True.

    (Output)

    row[num_entries] : 1st indices corresponding to rows of the global stiffness matrix.

    col[num_entries] : 2nd indices corresponding to columns of the global stiffness matrix.

    data[num_entries] : actual values of the global stiffness matrix.
    '''
    Kls = np.ascontiguousarray(Kls)
    nf = np.shape(face)[0]
    dof = 6
    num_entries = Kls.shape[0]*Kls.shape[1]*Kls.shape[2]
    row = np.empty(0, dtype=np.int32)
    col = np.empty(0, dtype=np.int32)
    data = np.empty(num_entries, dtype=np.float64)

    index = 0
    for i in range(nf):
        Rot = np.zeros_like(Kls[i])
        for j in range(6):
            Rot[3*j:3*(j+1),3*j:3*(j+1)] = R[i]
        Ke = Rot.T@Kls[i]@Rot

        for j1 in range(3):
            for j2 in range(3):
                for k1 in range(dof):
                    for k2 in range(dof):
                        data[index] = Ke[dof*j1+k1,dof*j2+k2]
                        index += 1

    if compute_index:
        row = np.empty(num_entries, dtype=np.int32)
        col = np.empty(num_entries, dtype=np.int32)
        index = 0
        for i in range(nf):
            for j1 in range(3):
                for j2 in range(3):
                    for k1 in range(dof):
                        for k2 in range(dof):
                            row[index] = dof*face[i,j1]+k1
                            col[index] = dof*face[i,j2]+k2
                            index += 1
    return row, col, data

@njit(Tuple((i4[:],i4[:],i4[:],f8[:]))(i4[:,:],f8[:,:,:],f8[:,:,:],f8[:,:,:,:],f8[:,:,:,:],b1),cache=CACHE,parallel=PARALLEL,fastmath=FASTMATH)
def _Grad_AssembleLocalStiffnessMatrices(face,Kls,R,Kls_g,R_g,compute_index):
    '''
    (Input)

    face[nf,3]<int> : Connectivity.

    Kls[nf,18,18]<float>: Local stiffness matrices.

    R[nf,3,3]<float>: Rotation matrices.

    Kls_g[9,nf,18,18]<float>: Gradient of local stiffness matrices.

    R_g[9,nf,3,3]<float>: Gradient of rotation matrices.

    compute_index<bool>: Compute row and col if True.

    (Output)

    dof_ids[num_entries] : 1st indices corresponding to variables.

    row[num_entries] : 2nd indices corresponding to rows of the global stiffness matrix.

    col[num_entries] : 3rd indices corresponding to columns of the global stiffness matrix.

    data[num_entries] : actual values of the global stiffness matrix.
    '''
    Kls = np.ascontiguousarray(Kls)
    Kls_g = np.ascontiguousarray(Kls_g)
    nf = np.shape(face)[0]
    dof = 6

    num_entries = Kls_g.shape[0]*Kls_g.shape[1]*Kls_g.shape[2]*Kls_g.shape[3]
    dof_ids = np.empty(0, dtype=np.int32)
    row = np.empty(0, dtype=np.int32)
    col = np.empty(0, dtype=np.int32)
    data = np.empty(num_entries, dtype=np.float64)

    index = 0
    for i in range(nf):
        Rot = np.zeros_like(Kls[i])
        for j in range(6):
            Rot[3*j:3*(j+1),3*j:3*(j+1)] = R[i]
        Rot_g = np.zeros_like(Kls_g[:,i])
        for k in range(9):
            for j in range(6):
                Rot_g[k,3*j:3*(j+1),3*j:3*(j+1)] = R_g[k,i]

        Ke_g1 = np.empty((9,18,18))
        Ke_g2 = np.empty((9,18,18))
        for j in range(9):
            Ke_g1[j] = Rot.T@Kls_g[j,i]@Rot
            Ke_g2[j] = Rot.T@Kls[i]@Rot_g[j]
        Ke_g = Ke_g1 + Ke_g2 + Ke_g2.transpose((0,2,1))

        for i1 in range(3): # corner
            for i2 in range(3): # x,y,z
                for j1 in range(3):
                    for j2 in range(3):
                        for k1 in range(dof):
                            for k2 in range(dof):
                                data[index] = Ke_g[3*i1+i2,dof*j1+k1,dof*j2+k2]
                                index += 1

    if compute_index:
        dof_ids = np.empty(num_entries, dtype=np.int32)
        row = np.empty(num_entries, dtype=np.int32)
        col = np.empty(num_entries, dtype=np.int32)
        index = 0
        for i in range(nf):
            for i1 in range(3): # corner
                for i2 in range(3): # x,y,z
                    for j1 in range(3):
                        for j2 in range(3):
                            for k1 in range(dof):
                                for k2 in range(dof):
                                    dof_ids[index] = 3*face[i,i1]+i2
                                    row[index] = dof*face[i,j1]+k1
                                    col[index] = dof*face[i,j2]+k2
                                    index += 1
    return dof_ids, row, col, data

class KirchhoffShellAnalysis():

    def __init__(self, sparse=True):
        self.dof = 6  # number of dofs per node (u,v,w,r_x,r_y,r_z)
        self.Sparse = sparse

        self.Init = True # This is for optimization to omit computing the row and column data for assembling the global stiffness matrix (in csr format, which is for representing sparse matrices) at each iteration.
        self.row = None # Row componenets of non-zero elements in the global stiffness matrix, which is stored when self.Init==True.
        self.column = None # Column componenets of non-zero elements in the global stiffness matrix, which is stored when self.Init==True.

        self.Init_g = True  # This is for optimization to omit computing the dof_ids, row and column data for assembling the gradients of global stiffness matrix (in a list of csr format) at each iteration.
        self.dof_ids = None # dof_ids of non-zero elements in the gradients of global stiffness matrix, which is stored when self.Init_g==True.
        self.indices_g = None # A list of integer arrays organized by dof_ids. The indices of dof_ids==i is stored in i-th array of this data. We need this data instead of dof_ids, as scipy's csr format cannot handle multi-dimensional (more than 2D) arrays. Computed only when self.Init_g==True.
        self.row_g = None # Row componenets of non-zero elements in the global stiffness matrix, which is stored when self.Init_g==True.
        self.column_g = None # Column componenets of non-zero elements in the global stiffness matrix, which is stored when self.Init_g==True.
        return
    
    def __reset__(self):
        '''
        Please run this code if you change the dofs of the structural model.
        '''
        self.Init = True
        self.Init_g = True
    
    def GlobalStiffness(self, vert, face, thickness, elastic_modulus, poisson):
        '''
        (Input)

        vert[nv,3]<float> : Vertex coordinates.

        face[nf,3]<int> : Connectivity.

        thickness<float> or [nf]<float> : Face thickness(es).

        elastic_modulus<float> : Elastic modulus.

        poisson_ratio<float> : Poisson's ratio.

        (Output)

        K[nv*6,nv*6]<float> : Global stiffness matrix (shape is (nv*6) times (nv*6)).
        '''
        nv = np.shape(vert)[0]
        nf = np.shape(face)[0]
        if type(thickness):
            thickness = np.ones(nf)*thickness

        Kls, R = _LocalStiffnessMatrix(vert,face,thickness,elastic_modulus,poisson)

        if self.Init:
            self.row, self.col, data = _AssembleLocalStiffnessMatrices(face,Kls,R,self.Init)
            self.Init = False
        else:
            _, _, data = _AssembleLocalStiffnessMatrices(face,Kls,R,self.Init)
        
        if self.Sparse:
            K = sp.sparse.csr_matrix((data, (self.row, self.col)), shape=(nv*self.dof,nv*self.dof))

        else:
            K = np.zeros((nv*self.dof,nv*self.dof))
            for r, c, d in zip(self.row, self.col, data):
                K[r, c] = d

        return K
    
    def Grad_GlobalStiffness(self, vert, face, thickness, elastic_modulus, poisson):
        '''
        (Input)

        vert[nv,3]<float> : Vertex coordinates.

        face[nf,3]<int> : Connectivity.

        thickness<float> or [nf]<float> : Face thickness(es).

        elastic_modulus<float> : Elastic modulus.

        poisson_ratio<float> : Poisson's ratio.

        (Output)

        K_g[nv*3,nv*6,nv*6]<float> : Gradients of global stiffness matrix (shape is (nv*6) times (nv*6)) with respect to the nodal coordinates (nv*3).
        '''
        nv = np.shape(vert)[0]
        nf = np.shape(face)[0]
        if type(thickness):
            thickness = np.ones(nf)*thickness

        Kls, R = _LocalStiffnessMatrix(vert, face, thickness, elastic_modulus, poisson)
        Kls_g, R_g = _Grad_LocalStiffnessMatrix(vert, face, thickness, elastic_modulus, poisson)

        if self.Init_g:
            self.dof_ids, self.row_g, self.col_g, data = _Grad_AssembleLocalStiffnessMatrices(face,Kls,R,Kls_g,R_g,self.Init_g)
            self.indices_g = [np.where(self.dof_ids==i)[0] for i in range(nv*3)]
            self.Init_g = False
        else:
            _, _, _, data = _Grad_AssembleLocalStiffnessMatrices(face,Kls,R,Kls_g,R_g,self.Init_g)

        if self.Sparse:
            K_g = []
            for i in range(nv*3):
                K_g.append(sp.sparse.csr_array((data[self.indices_g[i]], (self.row_g[self.indices_g[i]], self.col_g[self.indices_g[i]])), shape=(nv*self.dof,nv*self.dof)))

        else:
            K_g = np.zeros((nv*3,nv*self.dof,nv*self.dof))
            for dof_id, r, c, d in zip(self.dof_ids, self.row_g, self.col_g, data):
                K_g[dof_id, r, c] = d

        return K_g
    
    def LoadVector(self,load):
        '''
        (Input)

        load<nv,6> : Nodal loads in all the dofs (0:x, 1:y, 2:z, 3:rx, 4:ry, 5:rz). Specify 0 if no load is assigned to that DOF.

        (Output)

        load_vector<nv*6> : Nodal load vector. If self.Sparse==true, the values are stored in the csr format.
        '''
        if self.Sparse:
            load_mat = sp.sparse.csr_matrix(load.flatten(),shape=(1,load.shape[0]*self.dof))   
            return sp.sparse.csr_matrix.transpose(load_mat)
        else:
            return load.flatten()[:,np.newaxis]
        
    def StrainEnergy_with_Gradient(self,vert,face,dirichlet_condition,load,thickness=1.0,elastic_modulus=1.0,poisson_ratio=0.25):
        '''
        (Input)

        vert[nv,3]<float> : Vertex coordinates.

        face[nf,3]<int> : Connectivity.

        dirichlet_condition[number_of_conditions]<(int1,int2,float)> : Dirichlet conditions of node "int1" in the "int2" (0:x, 1:y, 2:z, 3:rx, 4:ry, 5:rz) direction.

        load[nv,6]<float> : Nodal loads in all the dofs (0:x, 1:y, 2:z, 3:rx, 4:ry, 5:rz). Specify 0 if no load is assigned to that dof.

        thickness<float> or [nf]<float> : Face thickness(es).

        elastic_modulus<float> : Elastic modulus.

        poisson_ratio<float> : Poisson's ratio.

        (Output)

        strain_energy<float> : Linear strain energy.

        strain_energy_g[nv*3]<float> : Gradient of strain energy with respect to the nodal coordinates.
        '''
        nv = vert.shape[0]
        K = self.GlobalStiffness(vert,face,thickness,elastic_modulus,poisson_ratio)
        K_g = self.Grad_GlobalStiffness(vert,face,thickness,elastic_modulus,poisson_ratio)
        P = self.LoadVector(load)
        U = np.zeros((nv*self.dof,1))

        fix_d = [dirichlet_condition[i][0]*self.dof+dirichlet_condition[i][1] for i in range(len(dirichlet_condition))]
        for i in range(len(fix_d)):
            U[fix_d[i]] = dirichlet_condition[i][2]

        P = P - K@U # Apply dirichlet condition
        
        fix_d.sort()
        free_d = np.setdiff1d(np.arange(nv*self.dof),fix_d)

        K_free = K[free_d][:,free_d]
        K_free_g = [K_g[i][free_d][:,free_d] for i in range(len(K_g))]
        P_free = P[free_d]

        if self.Sparse:
            U_free = sp.sparse.linalg.spsolve(K_free,P_free)
        else:
            U_free = sp.linalg.solve(K_free,P_free).squeeze()#,assume_a='pos')

        strain_energy = 0.5*U_free@K_free@U_free
        strain_energy_g = np.array([-0.5*U_free@K_free_g[i]@U_free for i in range(len(K_free_g))])

        # '''
        # Check if analytical and numerical gradients of the strain energy are almost the same.
        # '''
        # delta = 1e-5 # Tiny interval for computing gradients numerically
        # for iii in range(5):
        #     F_g_a = strain_energy_g[iii]
        #     vert_minus = np.copy(vert)
        #     vert_minus[iii//3,iii%3] -= delta
        #     d_minus, _ = self.RunStructuralAnalysis(vert_minus,face,dirichlet_condition,load,thickness,elastic_modulus,poisson_ratio)
        #     F_minus = np.sum(d_minus*load)/2
        #     vert_plus = np.copy(vert)
        #     vert_plus[iii//3,iii%3] += delta
        #     d_plus, _ = self.RunStructuralAnalysis(vert_plus,face,dirichlet_condition,load,thickness,elastic_modulus,poisson_ratio)
        #     F_plus = np.sum(d_plus*load)/2
        #     F_g_n = (F_plus-F_minus)/(2*delta)
        #     print(f"variable {iii}:\n  analytical gradient:{F_g_a}\n  numerical gradient:={F_g_n}")

        return strain_energy, strain_energy_g
    
    def RunStructuralAnalysis(self,vert,face,dirichlet_condition,load,thickness=1.0,elastic_modulus=1.0,poisson_ratio=0.25):
        '''
        (Input)

        vert[nv,3]<float> : Vertex coordinates.

        face[nf,3]<int> : Connectivity.

        dirichlet_condition[number_of_conditions]<(int1,int2,float)> : Dirichlet conditions of node "int1" in the "int2" (0:x, 1:y, 2:z, 3:rx, 4:ry, 5:rz) direction.

        load[nv,6]<float> : Nodal loads in all the dofs (0:x, 1:y, 2:z, 3:rx, 4:ry, 5:rz). Specify 0 if no load is assigned to that dof.

        thickness<float> or [nf]<float> : Face thickness(es).

        elastic_modulus<float> : Elastic modulus.

        poisson_ratio<float> : Poisson's ratio.

        (Output)

        displacement[nv,6]<float> : Nodal displacements.

        reaction[nv,6]<float> : Reaction forces.
        '''
        nv = vert.shape[0]
        K = self.GlobalStiffness(vert,face,thickness,elastic_modulus,poisson_ratio)
        P = self.LoadVector(load)
        U = np.zeros((nv*self.dof,1))

        fix_d = [dirichlet_condition[i][0]*self.dof+dirichlet_condition[i][1] for i in range(len(dirichlet_condition))]
        for i in range(len(fix_d)):
            U[fix_d[i]] = dirichlet_condition[i][2]

        P = P - K@U # Apply dirichlet condition
        
        fix_d.sort()
        free_d = np.setdiff1d(np.arange(nv*self.dof),fix_d)

        K_free = K[free_d][:,free_d]
        P_free= P[free_d]

        if self.Sparse:
            U_free = sp.sparse.linalg.spsolve(K_free,P_free)
        else:
            U_free = sp.linalg.solve(K_free,P_free)
            
        for i in range(len(free_d)):
            U[free_d[i]] = U_free[i]

        K_fix = K[fix_d]
        P_fix = P[fix_d]

        R_fix = K_fix@U-P_fix
        reaction = np.zeros((nv,6))
        for i in range(len(fix_d)):
            reaction[fix_d[i]//6, fix_d[i]%6] = R_fix[i]
        
        displacement = U.reshape((nv,-1))

        return displacement, reaction
