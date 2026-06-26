import pickle
from datetime import datetime
import sys
from SpectralAdvectionDiffusion import *
import os
import time

gpu = True
if gpu:
    print('using gpu')
    import cupy as np
    from cupyx.scipy.sparse.linalg import splu
    from cupyx.scipy import sparse
    from cupyx.scipy import linalg as la
else:
    print('not using gpu')
    import numpy as np
    from scipy.sparse.linalg import splu
    from scipy import sparse
    from scipy import linalg as la

'''interior tests'''
r = 2
h = 4
dx = .1
Lx = 5
Ly = 5
Lz = 8
params = np.array([2,0,h,r,1])

Nx = int(np.round(Lx / dx))
if Nx%2:
    Nx = Nx+1
Ny = int(np.round(Ly / dx))
if Ny%2:
    Ny = Ny+1
Lx = dx*Nx
Ly = dx*Ny
Nz = int(np.round(pi/(np.arcsin(2*dx/Lz))+1)) # this should make the ratio dx/dz close to 1 at the center.
xx = dx*np.arange(1,Nx+1)
yy = dx*np.arange(1,Ny+1)
Dcheb0,zz0 = cheb(Nz-1)
zz = Lz*0.5*(zz0+1)
Dcheb = Dcheb0 / (Lz*0.5)
Y,Z,X = np.meshgrid(yy,zz,xx, indexing='ij')

grid_pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
interior_mask = is_interior_vec(grid_pts, params)

C = np.ravel(X*0)
C[interior_mask] = 1
C = C.reshape(np.shape(X))

y_slice = params[1]
y_idx = np.argmin(np.abs(yy-y_slice))

plane = C[y_slice,:,:]
fig, ax = plt.subplots()
ax.pcolor(xx,zz,plane)
ax.set_aspect('equal')