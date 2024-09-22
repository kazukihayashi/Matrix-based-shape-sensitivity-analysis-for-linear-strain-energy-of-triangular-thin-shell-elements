'''
Source code of
K. Hayashi and R. Mesnil, "Matrix-based shape sensitivity analysis for strain energy minimization of shells modeled by finite triangular elements".
'''

import numpy as np
import scipy as sp
from numba import njit, f8, i4, b1
from numba.types import Tuple

CACHE = True # Reduce overhead by saving the compiled code. NOTE: It is recommended to set to True to avoid repetitive compilation.
PARALLEL = False # Parallel might be effective only for +1M variables. NOTE: In the original paper this value is fixed to False.
FASTMATH = False # Relax some numerical rigour to gain additional performance. NOTE: it does not improve computational efficiency much, so it is recommended to set this option to False.

@njit(f8[:,:,:](f8[:,:],i4[:,:]),cache=CACHE,parallel=PARALLEL,fastmath=FASTMATH)
def _Coords(vert,connectivity):
    '''
    (input)

    vert[nv,3]<float> : Nodal coordinates

    connectivity[nm,n_corners_per_member]<int> : Connectivity

    (output)

    coords[nm,n_corners_per_member,3]: nodal coordinate matrices per member
    '''
    nm = connectivity.shape[0]
    n_corners_per_member = connectivity.shape[1]
    coords = np.zeros((nm,n_corners_per_member,3))
    for i in range(nm):
        for j in range(n_corners_per_member):
            coords[i,j,:] = vert[connectivity[i,j],:]
    return coords

@njit(f8[:](f8[:,:],i4[:,:]),cache=CACHE,parallel=PARALLEL,fastmath=FASTMATH)
def _Length(vert,edge):
    '''
    (input)

    vert[nv,3]<float> : Nodal coordinates

    edge[nm,2]<int> : Connectivity

    (output)

    L[nm]: Edge lengths with respect to nodal coordinates.
    '''
    L = np.array([np.sum((vert[edge[i,1]]-vert[edge[i,0]])**2)**0.5 for i in range(edge.shape[0])],dtype=np.float64)

    return L

@njit(f8[:,:](f8[:,:],i4[:,:]),cache=CACHE,parallel=PARALLEL,fastmath=FASTMATH)
def _Grad_Length(vert,edge):
    '''
    (input)

    vert[nv,3]<float> : Nodal coordinates

    edge[nm,2]<int> : Connectivity

    (output)

    L_g[6,nm]: Gradient of edge lengths with respect to the endpoint coordinates.

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
    (input)

    vert[nv,3]<float> : Nodal coordinates

    face[nf,3]<int> : Connectivity

    (output)

    area[nf]<float>: face areas

    a[nf,3]<float>: components for computing face areas, where A = (a[0]^2+a[1]^2+a[2]^3)^(1/2)
    '''
    nf = face.shape[0]
    coords = _Coords(vert,face)
    a = np.zeros((nf,3))
    for i in range(3): # corner
        for j in range(3): # x,y,z
            a[:,j] += coords[:,i,(j+1)%3]*coords[:,(i+2)%3,(j+2)%3] - coords[:,i,(j+1)%3]*coords[:,(i+1)%3,(j+2)%3]

    area = np.sqrt(np.sum(a**2,axis=1))/2

    return area, a

@njit(f8[:,:](f8[:,:],i4[:,:]),cache=CACHE,parallel=PARALLEL,fastmath=FASTMATH)
def _Grad_TriangleArea(vert,face):
    '''
    (input)

    vert[nv,3]<float> : Nodal coordinates

    face[nf,3]<int> : Connectivity

    (output)

    area_g[9,nf]<float>: Gradient of face areas with respect to the corner coordinates.

    Rows 0,1,2 are the gradients with respect to the x,y,z coordinate of the 1st corner.

    Rows 3,4,5 are the gradients with respect to the x,y,z coordinate of the 2nd corner.

    Rows 6,7,8 are the gradients with respect to the x,y,z coordinate of the 3rd corner.
    '''
    nf = face.shape[0]
    coords = _Coords(vert,face)
    area, a = _TriangleArea(vert,face)
    area_g = np.zeros((9,nf),dtype=np.float64)
    for i in range(nf):
        for j in range(3): # corner
            for k in range(3): # x,y,z
                area_g[j*3+k,i] = a[i,(k+2)%3]*(coords[i,(j+2)%3,(k+1)%3]-coords[i,(j+1)%3,(k+1)%3]) + a[i,(k+1)%3]*(coords[i,(j+1)%3,(k+2)%3]-coords[i,(j+2)%3,(k+2)%3])
        area_g[:,i] /= 4*area[i]

    return area_g

@njit(f8[:,:,:](f8[:,:],i4[:,:]),cache=CACHE,parallel=PARALLEL,fastmath=FASTMATH)
def _RotationMatrix(vert,face):
    '''
    (input)

    vert[nv,3]<float> : Nodal coordinates

    face[nf,3]<int> : Connectivity

    (output)

    R[nf,3,3]<float>: 3x3 rotation matrices.

    1st row (R[i,0,:]) is the unit directional vector from the 1st to the 2nd corner.
    
    3rd row (R[i,2,:]) is the unit normal vector.

    2nd row (R[i,1,:]) is the unit in-plane vector orthogonal to the 1st and 3rd vectors.
    '''
    nf = face.shape[0]
    R = np.empty((nf,3,3))
    L1 = _Length(vert,face)
    A, _ = _TriangleArea(vert,face)
    coords = _Coords(vert,face)

    for i in range(3): # 1st row, repeat for i = 0,1,2 (x,y,z)
        R[:,0,i] = (coords[:,1,i]-coords[:,0,i])/L1
    for i in range(3): # 3rd row, repeat for i = 0,1,2 (x,y,z)
        R[:,2,i] = ((coords[:,1,(i+1)%3]-coords[:,0,(i+1)%3])*(coords[:,2,(i+2)%3]-coords[:,0,(i+2)%3])-(coords[:,2,(i+1)%3]-coords[:,0,(i+1)%3])*(coords[:,1,(i+2)%3]-coords[:,0,(i+2)%3]))/(2*A)
    for i in range(3): # 2nd row, repeat for i = 0,1,2 (x,y,z)
        R[:,1,i] = R[:,0,(i+2)%3]*R[:,2,(i+1)%3]-R[:,0,(i+1)%3]*R[:,2,(i+2)%3]

    return R

@njit(f8[:,:,:,:](f8[:,:],i4[:,:]),cache=CACHE,parallel=PARALLEL,fastmath=FASTMATH)
def _Grad_RotationMatrix(vert,face):
    '''
    (input)
    vert[nv,3]<float> : Nodal coordinates
    face[nf,3]<int> : Connectivity

    (output)
    R_g[9,nf,3,3]<float>: Gradient of 3x3 rotation matrices.

    Rows 0,1,2 are the gradients with respect to the x,y,z coordinate of the 1st corner.

    Rows 3,4,5 are the gradients with respect to the x,y,z coordinate of the 2nd corner.

    Rows 6,7,8 are the gradients with respect to the x,y,z coordinate of the 3rd corner.
    '''
    nv = vert.shape[0]
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

@njit(Tuple((f8[:,:,:,:],f8[:,:,:,:],f8[:,:,:,:],f8[:,:,:,:]))(f8[:,:],i4[:,:]),cache=CACHE,parallel=PARALLEL,fastmath=FASTMATH)
def _Grad_B_Matrices(vert,face):
    '''
    (input)

    vert[nv,3]<float> : Nodal coordinates

    face[nf,3]<int> : Connectivity

    (output)

    Bm_g[9,nf,3,6]<float>: Gradient of 3x6 B matrices (membrane components)

    Bs_g[9,nf,2,9]<float>: Gradient of 2x9 B matrices (shear components)

    Bb_g[9,nf,3,9]<float>: Gradient of 3x9 B matrices (bending components)

    R_g[9,nf,3,3]<float>: Gradient of 3x3 rotation matrices

    (Description)

    Rows 0,1,2 are the gradients with respect to the x,y,z coordinate of the 1st corner.

    Rows 3,4,5 are the gradients with respect to the x,y,z coordinate of the 2nd corner.

    Rows 6,7,8 are the gradients with respect to the x,y,z coordinate of the 3rd corner.
    '''
    nf = face.shape[0]
    ndof_per_face = 9

    R = np.ascontiguousarray(_RotationMatrix(vert,face))
    R_g = np.ascontiguousarray(_Grad_RotationMatrix(vert,face))

    A, _ = _TriangleArea(vert,face)
    A_g = _Grad_TriangleArea(vert,face)

    coord2D = np.zeros((nf,3,3))
    for i in range(nf):
        coord2D[i] = vert[face[i]]@R[i].T

    coord2D_g = np.zeros((ndof_per_face,nf,3,3))
    for i in range(nf): # face
        for j in range(3): # corner
            for k in range(3): #  x,y,z
                coord2D_g[3*j+k,i,j,:] += R[i,:,k]
        for j in range(3): # corner
            for k in range(3): #  x,y,z
                coord2D_g[3*j+k,i] += vert[face[i]]@R_g[3*j+k,i].T

    for i in [1,2,0]:
        coord2D_g[:,:,i] -= coord2D_g[:,:,0]

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

    Bs_g = np.zeros((ndof_per_face,nf,2,9))      
    for i in range(nf):
        for j in range(3):
            Bs_g[:,i,0,3*j] = Nx_g[:,i,j]
            Bs_g[:,i,1,3*j] = Ny_g[:,i,j]

    Bb_g = np.zeros((ndof_per_face,nf,3,9))      
    for i in range(nf):
        for j in range(3):
            Bb_g[:,i,0,3*j+2] = -Nx_g[:,i,j]
            Bb_g[:,i,1,3*j+1] = Ny_g[:,i,j]
            Bb_g[:,i,2,3*j+1] = Nx_g[:,i,j]
            Bb_g[:,i,2,3*j+2] = -Ny_g[:,i,j]

    return Bm_g, Bs_g, Bb_g, R_g


@njit(Tuple((f8[:],f8[:]))(f8[:,:]),cache=CACHE,parallel=PARALLEL,fastmath=FASTMATH)
def _Nu(coords):
    '''
    Suppose the shape function is described as N = a + bx + cy.
    This method computes the partial derivative of N with respect to x and y;
    thus, the outputs are dN/dx=b and dN/dy=c.

    (input)

    coords[3,2or3]<float> : Nodal coordinates. Row i corrsponds to the position of node i.

    (output)

    dN/dx[3]<float>: Derivative of shape function with respect to (local) x coordinate

    dN/dy[3]<float>: Derivative of shape function with respect to (local) y coordinate
    '''
    if coords.shape[1] == 2:
        coords2D = coords - coords[0]
    # elif coord.shape[1] == 3:
    #     T = TransformationMatrix(coord)
    #     coord2D = (coord@T)[:,0:2] - coord[0]
    
    A_2 = coords2D[1,0]*coords2D[2,1] - coords2D[2,0]*coords2D[1,1] # double of triangle area
    
    # a = np.array([np.cross(coord2D[(i+1)%3],coord2D[(i+2)%3]) for i in range(3)])/A_2
    b = np.array([(coords2D[(i+1)%3,1] - coords2D[(i+2)%3,1]) for i in range(3)])/A_2
    c = np.array([(coords2D[(i+2)%3,0] - coords2D[(i+1)%3,0]) for i in range(3)])/A_2

    return b, c

@njit(Tuple((f8[:,:,:],f8[:,:,:]))(f8[:,:],i4[:,:],f8[:],f8,f8),cache=CACHE,parallel=PARALLEL,fastmath=FASTMATH)
def _LocalStiffnessMatrix(vert, face, thickness, E, poisson):
    '''
    (input)

    vert[nv,3]<float> : Nodal coordinates

    face[nf,3]<int> : Connectivity

    thickness[nf]<float> : thickness of the shell element.

    E<float> : Young's modulus of the material.

    poisson<float> : Poisson's ratio of the material.

    (output)

    Kls[nf,18,18]<float>: Local stiffness matrices

    R[nf,3,3]<float>: Rotation matrices
    '''
    kappa = 5/6 # Shear reduction factor
    dof = 6 # dof per node (x,y,z,rx,ry,rz)
    mdof = 2 # dof corresponding to membrane forces per node (x,y)
    bdof = 3 # dof corresponding to bending forces per node (z,rx,ry)
    nf = face.shape[0] # number of faces

    coords3D = _Coords(vert,face) # Nodal coordinate matrix per face element
    coords3D = np.ascontiguousarray(coords3D) # Convert to a contiguous array for faster computing using numba.jit

    ## Constitutive matrix (strain x, strain y, and shear xy)
    Cb = np.zeros((3,3))
    k1 = E/(1-poisson**2)
    Cb[0,0] = Cb[1,1] = k1
    Cb[0,1] = Cb[1,0] = k1*poisson
    Cb[2,2] = k1 * (1-poisson)/2

    ## Constitutive matrix (shear zx and shear yz)
    Cs = np.zeros((2,2))
    Cs[0,0] = Cs[1,1] = E/(2*(1+poisson))

    ## Rotation matrix
    R = np.ascontiguousarray(_RotationMatrix(vert,face))

    ## Initialize local stiffness matrices
    Kls = np.zeros((nf,18,18))

    for id_face in range(nf):
    
        coords2D = (coords3D[id_face] @ R[id_face].T)[:,0:2]
        coords2D -= np.vstack((coords2D[0],coords2D[0],coords2D[0]))
        coords2D = np.ascontiguousarray(coords2D)

        Ae = 0.5 * (coords2D[1,0]*coords2D[2,1] - coords2D[2,0]*coords2D[1,1])
        detJ = Ae * 2

        ## Membrane element
        Nx, Ny = _Nu(coords2D)
        Bm = np.zeros((3,6))
        for i in range(3):
            Bm[0,2*i] = Nx[i]
            Bm[1,2*i+1] = Ny[i]
            Bm[2,2*i] = Ny[i]
            Bm[2,2*i+1] = Nx[i]
        Kml = thickness[id_face] * Ae * Bm.T @ Cb @ Bm

        ## Mindling bending element
        Kbl = np.zeros((9,9))

        ## Shear
        GipS = [[1.0/3.0,1.0/3.0,1.0/3.0]] # Gauss integration point
        GiwS = [1.0] # Weight of Gauss integration point
        for i in range(len(GipS)):
            L = GipS[i]
            N = np.array(L)
            Nx, Ny = _Nu(coords2D)
            Bs = np.array([[Nx[0], 0.0, N[0], Nx[1], 0.0, N[1], Nx[2], 0.0, N[2]],[Ny[0], -N[0], 0.0, Ny[1], -N[1], 0.0, Ny[2], -N[2], 0.0]])            
            Kslip = kappa * thickness[id_face] * Bs.T @ Cs @ Bs
            detJWeight = GiwS[i]*detJ/2
            Kbl += detJWeight * Kslip

        ## Bending
        GipB = [[0.5,0.5,0.0],[0.0,0.5,0.5],[0.5,0.0,0.5]] # Gauss integration point, weight
        GipS = [1.0/3.0,1.0/3.0,1.0/3.0] # Gauss integration point, weight
        for i in range(len(GipB)):
            Nx, Ny = _Nu(coords2D)
            Bb = np.array([[0.0,0.0,-Nx[0],0.0,0.0,-Nx[1],0.0,0.0,-Nx[2]],[0.0,Ny[0],0.0,0.0,Ny[1],0.0,0.0,Ny[2],0.0],[0.0,Nx[0],-Ny[0],0.0,Nx[1],-Ny[1],0.0,Nx[2],-Ny[2]]])            
            Kslip = (thickness[id_face]**3)/12 * Bb.T @ Cb @ Bb
            detJWeight = GipS[i]*detJ/2
            Kbl += detJWeight * Kslip

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
    (input)

    vert[nv,3]<float> : Nodal coordinates

    face[nf,3]<int> : Connectivity

    thickness[nf]<float> : thickness of the shell element.

    E<float> : Young's modulus of the material.

    poisson<float> : Poisson's ratio of the material.

    (output)

    Kls_g[9,nf,18,18]<float>: Gradient of local stiffness matrices

    R_g[9,nf,3,3]<float>: Gradient of rotation matrices

    (Description)

    Rows 0,1,2 are the gradients with respect to the x,y,z coordinate of the 1st corner.

    Rows 3,4,5 are the gradients with respect to the x,y,z coordinate of the 2nd corner.

    Rows 6,7,8 are the gradients with respect to the x,y,z coordinate of the 3rd corner.
    '''
    kappa= 5/6 # Shear reduction factor
    dof = 6 # dof per node (x,y,z,rx,ry,rz)
    mdof = 2 # dof corresponding to membrane forces per node (x,y)
    bdof = 3 # dof corresponding to bending forces per node (z,rx,ry)
    tri_dof_trans = 9 # translational dofs per triangle face (3(x,y,z) x 3(corners) =9)
    nf = face.shape[0] # number of faces

    coords3D = _Coords(vert,face) # Nodal coordinate matrix per face element
    coords3D = np.ascontiguousarray(coords3D) # Convert to a contiguous array for faster computing using numba.jit

    ## Constitutive matrix (strain x, strain y, and shear xy)
    Cb = np.zeros((3,3),dtype=np.float64)
    k1 = E/(1-poisson**2)
    Cb[0,0] = Cb[1,1] = k1
    Cb[0,1] = Cb[1,0] = k1*poisson
    Cb[2,2] = k1 * (1-poisson)/2

    ## Constitutive matrix (shear zx and shear yz)
    Cs = np.zeros((2,2),dtype=np.float64)
    Cs[0,0] = Cs[1,1] = E/(2*(1+poisson))

    ## Rotation matrix
    R = np.ascontiguousarray(_RotationMatrix(vert,face))

    ## Gradient of B matrices and rotation matrices
    Bm_g, Bs_g, Bb_g, R_g = _Grad_B_Matrices(vert,face)
    Bm_g = np.ascontiguousarray(Bm_g)
    Bs_g = np.ascontiguousarray(Bs_g)
    Bb_g = np.ascontiguousarray(Bb_g)

    Ae = _TriangleArea(vert,face)[0]
    Ae_g = _Grad_TriangleArea(vert,face)
    Kls_g = np.zeros((tri_dof_trans,nf,18,18),dtype=np.float64)

    for id_face in range(nf):
   
        coords2D = (coords3D[id_face] @ R[id_face].T)[:,0:2]
        coords2D -= np.vstack((coords2D[0],coords2D[0],coords2D[0]))
        coords2D = np.ascontiguousarray(coords2D)

        ## Membrane element
        Nx, Ny = _Nu(coords2D)

        Bm = np.zeros((3,6),dtype=np.float64)
        for i in range(3):
            Bm[0,2*i] = Nx[i]
            Bm[1,2*i+1] = Ny[i]
            Bm[2,2*i] = Ny[i]
            Bm[2,2*i+1] = Nx[i]

        # ## If not using numba jit
        # Kml_g1 = thickness * Bm.T @ Cb @ Bm * Ae_g[:,id_face,np.newaxis,np.newaxis]
        # Kml_g2 = thickness * Ae[id_face] * Bm.T @ Cb @ Bm_g[:,id_face]

        ## If using numba jit
        Kml_g1 = np.empty((tri_dof_trans,6,6))
        Kml_g1_const_part = thickness[id_face] * Bm.T @ Cb @ Bm
        for id_var in range(tri_dof_trans):
            Kml_g1[id_var] = Ae_g[id_var,id_face] * Kml_g1_const_part
        Kml_g2 = np.empty((tri_dof_trans,6,6))
        Kml_g2_const_part = thickness[id_face] * Ae[id_face] * Bm.T @ Cb
        for id_var in range(tri_dof_trans):
            Kml_g2[id_var] = Kml_g2_const_part @ Bm_g[id_var,id_face]

        Kml_g = Kml_g1 + Kml_g2 + Kml_g2.transpose((0,2,1))

        ## Gradient of Mindling bending element
        Kbl_g = np.zeros((tri_dof_trans,9,9),dtype=np.float64)
        
        ## Shear
        GipS = [[1.0/3.0,1.0/3.0,1.0/3.0]] # Gauss integration point
        GiwS = [1.0] # Weight of Gauss integration point
        for i in range(len(GipS)):
            L = GipS[i]
            N = np.array(L)
            Nx, Ny = _Nu(coords2D)
            Bs = np.array([[Nx[0], 0.0, N[0], Nx[1], 0.0, N[1], Nx[2], 0.0, N[2]],[Ny[0], -N[0], 0.0, Ny[1], -N[1], 0.0, Ny[2], -N[2], 0.0]])            

            # ## If not using numba jit
            # Kbl_g1 = GiwS[i] * kappa * thickness * Bs.T @ Cs @ Bs * Ae_g[:,id_face,np.newaxis,np.newaxis]
            # Kbl_g2 = GiwS[i] * kappa * thickness * Ae[id_face] * Bs.T @ Cs @ Bs_g[:,id_face]

            ## If using numba jit
            Kbl_g1 = np.empty((tri_dof_trans,9,9))
            Kbl_g1_const_part = GiwS[i] * kappa * thickness[id_face] * Bs.T @ Cs @ Bs
            for id_var in range(tri_dof_trans):
                Kbl_g1[id_var] = Ae_g[id_var,id_face] * Kbl_g1_const_part
            Kbl_g2 = np.empty((tri_dof_trans,9,9))
            Kbl_g2_const_part = GiwS[i] * kappa * thickness[id_face] * Ae[id_face] * Bs.T @ Cs
            for id_var in range(tri_dof_trans):
                Kbl_g2[id_var] = Kbl_g2_const_part @ Bs_g[id_var,id_face]

            Kbl_g += Kbl_g1 + Kbl_g2 + Kbl_g2.transpose((0,2,1))

        ## Bending
        GipB = [[0.5,0.5,0.0],[0.0,0.5,0.5],[0.5,0.0,0.5]] # Gauss integration point, weight
        GipS = [1.0/3.0,1.0/3.0,1.0/3.0] # Gauss integration point, weight
        for i in range(len(GipB)):
            Nx, Ny = _Nu(coords2D)
            Bb = np.array([[0.0,0.0,-Nx[0],0.0,0.0,-Nx[1],0.0,0.0,-Nx[2]],[0.0,Ny[0],0.0,0.0,Ny[1],0.0,0.0,Ny[2],0.0],[0.0,Nx[0],-Ny[0],0.0,Nx[1],-Ny[1],0.0,Nx[2],-Ny[2]]])            

            # ## If not using numba jit
            # Kbl_g1 = GipS[i] * (thickness**3)/12 * Bb.T @ Cb @ Bb * Ae_g[:,id_face,np.newaxis,np.newaxis]
            # Kbl_g2 = GipS[i] * (thickness**3)/12 * Ae[id_face] * Bb.T @ Cb @ Bb_g[:,id_face]

            ## If using numba jit
            Kbl_g1 = np.empty((tri_dof_trans,9,9))
            Kbl_g1_const_part = GipS[i] * (thickness[id_face]**3)/12 * Bb.T @ Cb @ Bb
            for id_var in range(tri_dof_trans):
                Kbl_g1[id_var] = Ae_g[id_var,id_face] * Kbl_g1_const_part
            Kbl_g2 = np.empty((tri_dof_trans,9,9))
            Kbl_g2_const_part = GipS[i] * (thickness[id_face]**3)/12 * Ae[id_face] * Bb.T @ Cb
            for id_var in range(tri_dof_trans):
                Kbl_g2[id_var] = Kbl_g2_const_part @ Bb_g[id_var,id_face]
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
def _AssembleLocalStiffnessMatrices(face,Kls,Trs,compute_index):
    '''
    (input)

    face[nf,3]<int> : Connectivity

    Kls[nf,18,18]<float>: Local stiffness matrices

    R[nf,3,3]<float>: Rotation matrices

    compute_index<bool>: Compute row and col if True

    (output)

    row[num_entries] : 1st indices corresponding to rows of the global stiffness matrix

    col[num_entries] : 2nd indices corresponding to columns of the global stiffness matrix
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
            Rot[3*j:3*(j+1),3*j:3*(j+1)] = Trs[i]
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
def _Grad_AssembleLocalStiffnessMatrices(face,Kls,Trs,Kls_g,Trs_g,compute_index):
    '''
    (input)

    face[nf,3]<int> : Connectivity

    Kls[nf,18,18]<float>: Local stiffness matrices

    R[nf,3,3]<float>: Rotation matrices

    Kls_g[9,nf,18,18]<float>: Gradient of local stiffness matrices

    R_g[9,nf,3,3]<float>: Gradient of rotation matrices

    compute_index<bool>: Compute row and col if True

    (output)

    dof_ids[num_entries] : 1st indices corresponding to variables

    row[num_entries] : 2nd indices corresponding to rows of the global stiffness matrix

    col[num_entries] : 3rd indices corresponding to columns of the global stiffness matrix
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
            Rot[3*j:3*(j+1),3*j:3*(j+1)] = Trs[i]
        Rot_g = np.zeros_like(Kls_g[:,i])
        for k in range(9):
            for j in range(6):
                Rot_g[k,3*j:3*(j+1),3*j:3*(j+1)] = Trs_g[k,i]

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

class MindlinShell():

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
        (input)

        vert[nv,3]<float> : Vertex coordinates.

        face[nf,3]<int> : Connectivity.

        thickness<float> or [nf]<float> : Face thickness(es).

        elastic_modulus<float> : Elastic modulus.

        poisson_ratio<float> : Poisson's ratio.

        (output)

        K[nv*6,nv*6]<float> : Global stiffness matrix (shape is (nv*6) times (nv*6)).
        '''
        nv = np.shape(vert)[0]
        nf = np.shape(face)[0]
        if type(thickness):
            thickness = np.ones(nf)*thickness

        Kls, Trs = _LocalStiffnessMatrix(vert,face,thickness,elastic_modulus,poisson)

        if self.Init:
            self.row, self.col, data = _AssembleLocalStiffnessMatrices(face,Kls,Trs,self.Init)
            self.Init = False
        else:
            _, _, data = _AssembleLocalStiffnessMatrices(face,Kls,Trs,self.Init)
        
        if self.Sparse:
            K = sp.sparse.csr_matrix((data, (self.row, self.col)), shape=(nv*self.dof,nv*self.dof))

        else:
            K = np.zeros((nv*self.dof,nv*self.dof))
            for r, c, d in zip(self.row, self.col, data):
                K[r, c] = d

        return K
    
    def Grad_GlobalStiffness(self, vert, face, thickness, elastic_modulus, poisson):
        '''
        (input)

        vert[nv,3]<float> : Vertex coordinates.

        face[nf,3]<int> : Connectivity.

        thickness<float> or [nf]<float> : Face thickness(es).

        elastic_modulus<float> : Elastic modulus.

        poisson_ratio<float> : Poisson's ratio.

        (output)

        K_g[nv*3,nv*6,nv*6]<float> : Gradients of global stiffness matrix (shape is (nv*6) times (nv*6)) with respect to the nodal coordinates (nv*3).
        '''
        nv = np.shape(vert)[0]
        nf = np.shape(face)[0]
        if type(thickness):
            thickness = np.ones(nf)*thickness

        Kls, Trs = _LocalStiffnessMatrix(vert, face, thickness, elastic_modulus, poisson)
        Kls_g, Trs_g = _Grad_LocalStiffnessMatrix(vert, face, thickness, elastic_modulus, poisson)

        if self.Init_g:
            self.dof_ids, self.row_g, self.col_g, data = _Grad_AssembleLocalStiffnessMatrices(face,Kls,Trs,Kls_g,Trs_g,self.Init_g)
            self.indices_g = [np.where(self.dof_ids==i)[0] for i in range(nv*3)]
            self.Init_g = False
        else:
            _, _, _, data = _Grad_AssembleLocalStiffnessMatrices(face,Kls,Trs,Kls_g,Trs_g,self.Init_g)

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
        (input)

        load<nv,6> : Nodal loads in all the dofs (0:x, 1:y, 2:z, 3:rx, 4:ry, 5:rz). Specify 0 if no load is assigned to that dof.

        (output)

        load_vector<nv*6> : Nodal load vector. If self.Sparse==true, the values are stored in the csr format.
        '''
        if self.Sparse:
            load_mat = sp.sparse.csr_matrix(load.flatten(),shape=(1,load.shape[0]*self.dof))   
            return sp.sparse.csr_matrix.transpose(load_mat)
        else:
            return load.flatten()[:,np.newaxis]
        
    def StrainEnergy_with_Gradient(self,vert,face,dirichlet_condition,load,thickness=1.0,elastic_modulus=1.0,poisson_ratio=0.25):
        '''
        (input)

        vert[nv,3]<float> : Vertex coordinates.

        face[nf,3]<int> : Connectivity.

        dirichlet_condition[number_of_conditions]<(int1,int2,float)> : Dirichlet conditions of node "int1" in the "int2" (0:x, 1:y, 2:z, 3:rx, 4:ry, 5:rz) direction.

        load[nv,6]<float> : Nodal loads in all the dofs (0:x, 1:y, 2:z, 3:rx, 4:ry, 5:rz). Specify 0 if no load is assigned to that dof.

        thickness<float> or [nf]<float> : Face thickness(es).

        elastic_modulus<float> : Elastic modulus.

        poisson_ratio<float> : Poisson's ratio.

        (output)

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
        # Check that analytical gradient and numerical gradient of the strain energy are almost the same.
        # '''
        # delta = 1e-5 # Tiny interval for computing gradients numerically
        # for iii in range(5):
        #     F_g_a = strain_energy_g[iii]
        #     vert_minus = np.copy(vert)
        #     vert_minus[iii//3,iii%3] -= delta
        #     d_minus, _ = self.RunStructuralAnalysis(vert_minus,face,dirichlet_condition,load,thickness=thickness,elastic_modulus=1.0,poisson_ratio=0.25)
        #     F_minus = np.sum(d_minus*load)/2
        #     vert_plus = np.copy(vert)
        #     vert_plus[iii//3,iii%3] += delta
        #     d_plus, _ = self.RunStructuralAnalysis(vert_plus,face,dirichlet_condition,load,thickness=thickness,elastic_modulus=1.0,poisson_ratio=0.25)
        #     F_plus = np.sum(d_plus*load)/2
        #     F_g_n = (F_plus-F_minus)/(2*delta)
        #     print(f"variable {iii}:\n  analytical gradient:{F_g_a}\n  numerical gradient:={F_g_n}")

        return strain_energy, strain_energy_g
    
    def RunStructuralAnalysis(self,vert,face,dirichlet_condition,load,thickness=1.0,elastic_modulus=1.0,poisson_ratio=0.25):
        '''
        (input)

        vert[nv,3]<float> : Vertex coordinates.

        face[nf,3]<int> : Connectivity.

        dirichlet_condition[number_of_conditions]<(int1,int2,float)> : Dirichlet conditions of node "int1" in the "int2" (0:x, 1:y, 2:z, 3:rx, 4:ry, 5:rz) direction.

        load[nv,6]<float> : Nodal loads in all the dofs (0:x, 1:y, 2:z, 3:rx, 4:ry, 5:rz). Specify 0 if no load is assigned to that dof.

        thickness<float> or [nf]<float> : Face thickness(es).

        elastic_modulus<float> : Elastic modulus.

        poisson_ratio<float> : Poisson's ratio.

        (output)

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
    