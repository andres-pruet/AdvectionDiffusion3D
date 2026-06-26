gpu = False # make sure this matches in the other files
if gpu:
    print('using gpu')
    import cupy as np
    from cupyx.scipy import sparse
    def njit(fastmath=0): # have to write a wrapper that does nothing for GPU compatibility. 
        def decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator
else:
    print('not using gpu')
    import numpy as np
    from scipy import sparse
    from numba import njit
from matplotlib import pyplot as plt
import sys

pi = np.pi
norm = np.linalg.norm

def g_block_diag(mats, format=None, dtype=None):
    """
    Build a block diagonal sparse matrix from provided matrices.
    https://github.com/cupy/cupy/issues/7058
    """
    row = []
    col = []
    data = []
    r_idx = 0
    c_idx = 0
    for a in mats:
        a = sparse.coo_matrix(a)
        nrows, ncols = a.shape

        row.append(a.row + r_idx)
        col.append(a.col + c_idx)
        data.append(a.data)

        r_idx += nrows
        c_idx += ncols
    row = np.asarray(np.concatenate(row))
    col = np.asarray(np.concatenate(col))
    data = np.asarray(np.concatenate(data))
    return sparse.coo_matrix((data, (row, col)),
                      shape=(r_idx, c_idx),
                      dtype=dtype).asformat(format)

def cheb(N):
    x = np.cos(pi*np.arange(N+1)/N)
    c = np.concatenate([np.array([2]), np.ones(N-1), np.array([2])])*np.power(-1,np.arange(N+1))
    X = np.transpose(np.tile(x,[N+1,1]))
    dX = X - np.transpose(X)
    D = np.outer(c, 1/c) / (dX + np.eye(N+1))
    D = D - np.diag(np.sum(np.transpose(D), axis=0))
    return D,x

def SecondIntegralMatrix(N):
    jj = np.arange(3,N)
    colm2 = np.concatenate([np.array([.25]), 1/(2*jj*(2*jj-2)), np.array([0,0]), np.array([0,0])])
    colp2 = np.concatenate([np.array([0,0]), np.array([0, .125, 1/24]), 1/((2*jj)*(2*jj+2))*(jj<N-2)]) # gotta add these extra [0,0]s becaus
    col0 = np.concatenate([np.array([0, -.125, -1/8-1/24]), -1/((2*jj)*(2*jj-2)) - 1/((2*jj)*(2*jj+2))*(jj<N-1), np.array([0,0])] )

    simat = sparse.dia_matrix(([colm2, col0, colp2], np.array([-2,0,2])), (N, N+2), copy=True)
    # simat = simat.tolil(copy=True)
    simat = simat.tocsc()
    simat[0, N+0] = 1
    simat[1,N+1] = 1
    # simat = simat.tocsc()
    return simat

def get_initial(X,Y,Z,Lx,Ly,source_location,initial_sigma,n_copies):
    xs,ys,zs = source_location
    def delta(r):
        # specifically for 3 dimensions
        return (1/(2*pi*initial_sigma**2)**(3/2)) * np.exp(-0.5*(r/initial_sigma)**2)
    # dSrc = 0.*delta(np.sqrt((X-xs)**2 + (Y-ys)**2 + (Z-zs)**2))
    dSrc = 0*X
    for x_idx in np.arange(-n_copies,(n_copies+1)):
        for y_idx in np.arange(-n_copies,(n_copies+1)):
            dSrc = dSrc + delta(np.sqrt((X-xs-Lx*x_idx)**2 + (Y-ys-Ly*y_idx)**2 + (Z-zs)**2))
            dSrc = dSrc + delta(np.sqrt((X-xs-Lx*x_idx)**2 + (Y-ys-Ly*y_idx)**2 + (Z+zs)**2))
    return dSrc

def get_x_deriv(A,ik):
    # remember, x is on the 2st axis
    return np.fft.ifft(ik[np.newaxis,np.newaxis,:]*np.fft.fft(A,axis=2),axis=2)

def get_y_deriv(A,il):
    # remember, y is on the 0st axis
    return np.fft.ifft(il[:,np.newaxis,np.newaxis]*np.fft.fft(A,axis=0),axis=0)

def get_z_deriv(A,Dcheb):
    # A@b acts on the columns (n,:,m) slices of b
    return Dcheb@A

def get_2nd_x_deriv(A,ksq):
    return np.fft.ifft(-ksq[np.newaxis,np.newaxis,:]*np.fft.fft(A,axis=2), axis=2)

def get_2nd_y_deriv(A,lsq):
    return np.fft.ifft(-lsq[:,np.newaxis,np.newaxis]*np.fft.fft(A,axis=0), axis=0)

def get_2nd_z_deriv(A,Dcheb):
    return Dcheb@(Dcheb@A)

def get_rhs_1step(C,dt,D,Ux,Uy,Uz,Dcheb,ik,il):
    rem = -C/dt/D

    grad_dot_UC = get_x_deriv(Ux*C,ik) + get_y_deriv(Uy*C,il) + get_z_deriv(Uz*C,Dcheb)
    adv = grad_dot_UC / D
    
    return rem + adv

def get_rhs_2step(C_lag1,C_lag2,dt,D,Ux_lag1,Uy_lag1,Uz_lag1,Ux_lag2,Uy_lag2,Uz_lag2,gamma,c,Dcheb,S_lag1,S_lag2,ik,il,ksq,lsq):
    a = 1/(D*(gamma+c/2))
    rem = -a*(2*gamma*C_lag1 - (gamma-.5)*C_lag2)/dt
    
    grad_dot_UC_lag1 = Ux_lag1*get_x_deriv(C_lag1,ik) + Uy_lag1*get_y_deriv(C_lag1,il) + Uz_lag1*get_z_deriv(C_lag1,Dcheb)
    grad_dot_UC_lag2 = Ux_lag2*get_x_deriv(C_lag2,ik) + Uy_lag2*get_y_deriv(C_lag2,il) + Uz_lag2*get_z_deriv(C_lag2,Dcheb)

    adv = -a*((gamma+1)*(-grad_dot_UC_lag1) - gamma*(-grad_dot_UC_lag2))

    grad_squared_C_lag1 = get_2nd_x_deriv(C_lag1,ksq) + get_2nd_y_deriv(C_lag1,lsq) + get_2nd_z_deriv(C_lag1,Dcheb)
    grad_squared_C_lag2 = get_2nd_x_deriv(C_lag2,ksq) + get_2nd_y_deriv(C_lag2,lsq) + get_2nd_z_deriv(C_lag2,Dcheb)
    diff = -a*D*((1-gamma-c)*grad_squared_C_lag1 + (c/2)*grad_squared_C_lag2)

    src = -a*((gamma+1)*(S_lag1) - gamma*(S_lag2))

    return rem + adv + diff + src

def get_rhs_2step_2part(C_lag1,C_lag2,Cp_lag1,Cp_lag2,alpha_lag1,alpha_lag2,dGdx,dGdy,dGdz,dGdx_int,dGdy_int,dGdz_int,alpha_int_lag1,alpha_int_lag2,
                        dt,D,Ux_lag1,Uy_lag1,Uz_lag1,Ux_lag2,Uy_lag2,Uz_lag2,gamma,c,Dcheb,S_lag1,S_lag2,ik,il,ksq,lsq):
    dim = np.shape(C_lag1)
    a = 1/(D*(gamma+c/2))
    rem = -a*(2*gamma*C_lag1 - (gamma-.5)*C_lag2)/dt
    
    grad_dot_UC_lag1_p = Ux_lag1*get_x_deriv(Cp_lag1,ik) + Uy_lag1*get_y_deriv(Cp_lag1,il) + Uz_lag1*get_z_deriv(Cp_lag1,Dcheb)
    grad_dot_UC_lag2_p = Ux_lag2*get_x_deriv(Cp_lag2,ik) + Uy_lag2*get_y_deriv(Cp_lag2,il) + Uz_lag2*get_z_deriv(Cp_lag2,Dcheb)
    grad_dot_UC_lag1_h = Ux_lag1*(dGdx@alpha_lag1).reshape(dim) + Uy_lag1*(dGdy@alpha_lag1).reshape(dim) + Uz_lag1*(dGdz@alpha_lag1).reshape(dim)
    grad_dot_UC_lag2_h = Ux_lag2*(dGdx@alpha_lag2).reshape(dim) + Uy_lag2*(dGdy@alpha_lag2).reshape(dim) + Uz_lag2*(dGdz@alpha_lag2).reshape(dim)
    grad_dot_UC_lag1_int = Ux_lag1*(dGdx_int@alpha_int_lag1).reshape(dim) + Uy_lag1*(dGdy_int@alpha_int_lag1).reshape(dim) + Uz_lag1*(dGdz_int@alpha_int_lag1).reshape(dim)
    grad_dot_UC_lag2_int = Ux_lag2*(dGdx_int@alpha_int_lag2).reshape(dim) + Uy_lag2*(dGdy_int@alpha_int_lag2).reshape(dim) + Uz_lag2*(dGdz_int@alpha_int_lag2).reshape(dim)

    grad_dot_UC_lag1 = grad_dot_UC_lag1_p + grad_dot_UC_lag1_h + grad_dot_UC_lag1_int
    grad_dot_UC_lag2 = grad_dot_UC_lag2_p + grad_dot_UC_lag2_h + grad_dot_UC_lag2_int

    adv = -a*((gamma+1)*(-grad_dot_UC_lag1) - gamma*(-grad_dot_UC_lag2))

    # have to redo all this unless gamma=1, c=0
    grad_squared_C_lag1 = get_2nd_x_deriv(C_lag1,ksq) + get_2nd_y_deriv(C_lag1,lsq) + get_2nd_z_deriv(C_lag1,Dcheb)
    grad_squared_C_lag2 = get_2nd_x_deriv(C_lag2,ksq) + get_2nd_y_deriv(C_lag2,lsq) + get_2nd_z_deriv(C_lag2,Dcheb)
    diff = -a*D*((1-gamma-c)*grad_squared_C_lag1 + (c/2)*grad_squared_C_lag2)

    src = -a*((gamma+1)*(S_lag1) - gamma*(S_lag2))

    return rem + adv + diff + src

def BCRows(N):
    '''
    This function gives you the following BCs:
    BCR1 = first derivative (first integral) evaluated at x=1
    BCR2 = function (second integral) evaluated at x=1
    BCL1 = first derivative (first integral) evaluated at x=-1
    BCL2 = NEGATIVE OF function (second integral) evaluated at x=-1
    '''
    BCR1=np.zeros(N+2)
    BCR2=np.zeros(N+2)
    BCL1=np.zeros(N+2)
    BCL2=np.zeros(N+2)
    # Special cases - right 
    BCR1[N+1]=1; BCR2[N+0]=1; BCR1[0]=1; BCR1[2]=-1/2
    BCR2[N+1]=BCR2[N+1]+1; BCR2[1]=-1/8; BCR2[3]=1/8
    BCR1[1]=BCR1[1]+1/4; BCR1[3]=BCR1[3]-1/4; BCR2[0]=BCR2[0]+1/4; BCR2[2]=BCR2[2]-1/8-1/24
    BCR2[4]=BCR2[4]+1/24

    # Special cases - left
    BCL1[N+1]=1; BCL2[N+0]=-1; BCL1[0]=-1; BCL1[2]=1/2
    BCL2[N+1]=BCL2[N+1]+1; BCL2[1]=-1/8; BCL2[3]=1/8
    BCL1[1]=BCL1[1]+1/4; BCL1[3]=BCL1[3]-1/4; BCL2[0]=BCL2[0]-1/4; BCL2[2]=BCL2[2]+1/8+1/24
    BCL2[4]=BCL2[4]-1/24

    # Easy cases
    jj = np.arange(3,N)
    BCR1[jj-1]=BCR1[jj-1]+1/(2*jj) 
    BCL1[jj-1]=BCL1[jj-1]+(-1)**jj/(2*jj)
    BCR1[jj+2-1]=BCR1[jj+2-1]-1/(2*jj)*(jj<N-1) 
    BCL1[jj+2-1]=BCL1[jj+2-1]-(-1)**jj/(2*jj)*(jj<N-1)
    BCR2[jj-1-1]=BCR2[jj-1-1]+1/(2*jj)*1/(2*jj-2) 
    BCL2[jj-1-1]=BCL2[jj-1-1]-1/(2*jj)*1/(2*jj-2)*(-1)**jj
    BCR2[jj+3-1]=BCR2[jj+3-1]+1/(2*jj)*1/(2*jj+2)*(jj<N-2) 
    BCL2[jj+3-1]=BCL2[jj+3-1]-1/(2*jj)*1/(2*jj+2)*(-1)**jj*(jj<N-2)
    BCR2[jj+1-1]=BCR2[jj+1-1]-1/(2*jj)*1/(2*jj-2)-1/(2*jj)*1/(2*jj+2)*(jj<N-1)
    BCL2[jj+1-1]=BCL2[jj+1-1]+(1/(2*jj)*1/(2*jj-2)+1/(2*jj)*1/(2*jj+2)*(jj<N-1))*(-1)**jj
    BCs = np.stack([BCR1, BCR2, BCL1, BCL2])
    return BCs

def solve_block_2s(A,y):
    # solve many 2-by-2 systems.
    # the matrices are stacked in a 2n-by-2 matrix
    x = np.zeros(len(A), complex)
    n = len(A) // 2
    evens = np.array(np.arange(n)*2,int)
    odds = np.array(evens+1,int)
    y1 = y[evens]
    y2 = y[odds]
    a = A[evens,np.zeros(n,int)]
    b = A[evens,np.zeros(n,int)+1]
    c = A[odds,np.zeros(n,int)]
    d = A[odds,np.zeros(n,int)+1]
    x[evens] = (y1 + (-b/(d-c*b/a))*(y2-c*y1/a))/a
    x[odds] = (y2 - c*y1/a)/(d-c*b/a)
    return x

def ftransform3(gridf):
    # transform each column
    sz = np.shape(gridf)
    Nz = sz[1]
    Nx = sz[2]
    Ny = sz[0]
    lenU = 2*Nz-2
    U = np.zeros((Ny,lenU,Nx),dtype = complex)
    U[:,0:Nz,:] = gridf
    U[:,(Nz):(2*Nz-2),:] = gridf[:,(Nz-2):0:-1,:]
    U = np.fft.fft(U,axis=1)
    fhat = np.zeros(np.shape(gridf), dtype = complex)
    fhat[:,0,:] = U[:,0,:]
    fhat[:,1:(Nz-1),:] = U[:,1:(Nz-1),:] + U[:,lenU-1:Nz-1:-1,:]
    fhat[:,Nz-1,:] = U[:,Nz-1,:]
    fhat = fhat/(2*Nz-2)
    return fhat

def btransform3(fhat):
    # transform each column
    sz = np.shape(fhat)
    Nz = sz[1]
    Nx = sz[2]
    Ny = sz[0]
    lenU = 2*Nz-2
    U = np.zeros((Ny,lenU,Nx), dtype = complex)
    U[:,0,:] = fhat[:,0,:]
    U[:,1:Nz-1,:]=fhat[:,1:Nz-1,:]/2
    U[:,Nz-1,:]=fhat[:,Nz-1,:]
    U[:,Nz:2*Nz-2,:] = fhat[:,Nz-2:0:-1,:]/2
    U = U*(2*Nz-2)
    U = np.fft.ifft(U,axis=1)
    fgrid = U[:,0:Nz,:]
    return fgrid

def InvYukawa(rhs,lambda_k,SIMat_arr,Lz,Nz,Nx,Ny,Ainv,block_C,bigM1,y_ids,x_idxs,y_idxs):
    # SIMat_arr = SIMat[0:Nz,:] (as np array)
    rhs_k = np.fft.fft2(rhs,axes=[2,0])
    bigfhat = ftransform3(rhs_k).transpose((0,2,1)).reshape((-1)) # transform and stack the columns of the matrix. Start with slice y=0, then y=1, ...
    # c = ftransform3(rhs_k)
    # c = btransform3(c)
    # c = np.fft.ifft2(c)
    # c = np.real(c).get()
    # plt.pcolor(c[24,:,:])
    # plt.show()
    # sys.exit()
    H = Lz/2
    # print('before ainv 1')
    big_M2_solve = Ainv.solve(bigfhat)
    # print('after ainv 1')

    bigM2 = block_C @ big_M2_solve -  0 # np.repeat(np.array([0,0]),Nx) because we set 0 penetration on floor and ceiling
    bigy = solve_block_2s(bigM1,bigM2)

    bigBy = np.zeros(Nz*Nx*Ny, dtype=complex)
    bigBy[y_ids] = bigy*(-np.repeat(lambda_k.transpose().reshape((-1)),2)*H**2)

    bigx_rhs_rhs = bigfhat - bigBy
    # print('before ainv 2')
    bigx = Ainv.solve(bigx_rhs_rhs)
    # print('after ainv 2')
    
    bigBVPSoln = np.zeros(Ny*Nx*(Nz+2), dtype=complex)
    bigBVPSoln[x_idxs] = bigx
    bigBVPSoln[y_idxs] = bigy
    # print('should be the same:')
    # print(bigBVPSoln[(Nz+2):(2*Nz+4)])
    bigBVPSoln = bigBVPSoln.reshape((Ny,Nx,Nz+2)).transpose((0,2,1))
    # print(bigBVPSoln[0,:,1])
    # sys.exit()
    
    # print(f'bigBVP shape: {np.shape(bigBVPSoln)}')
    # print(f'mat shape: {np.shape(H**2*SIMat_coo)}')
    secD = H**2*SIMat_arr @ bigBVPSoln
    Cnext = btransform3(secD)
        
    Cn = np.real(np.fft.ifft2(Cnext,axes=[2,0]))
    # c = np.real(Cn).get()
    # plt.pcolor(c[24,:,:])
    # plt.show()
    # sys.exit()
    return Cn