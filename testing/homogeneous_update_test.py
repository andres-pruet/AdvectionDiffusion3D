'''
This code compares the obstacle-free solver with the analytic solution for advection-diffusion of a Gaussian puff in 2d.
As it turns out, the analytic solution for a Gaussian puff is always Gaussian.
'''

gpu = True # make sure this matches other 4 files
if gpu:
    import cupy as np
else:
    import numpy as np
import sys
sys.path.append('../src/')
from SpectralAdvectionDiffusion import *
from matplotlib import pyplot as plt
norm = np.linalg.norm

dx = 1/4
Lx = 8
Ly = 8
Lz = 10
stop_time = 1.
nsteps_per_second = 140
D = 1
gamma = 1
c = 0

# initial conditions for C
source_location = np.array([3,4,4]) # put this nearby the obstacle
source_spread = 0.5
source_type = 'plume'
n_copies = 10

# obstacle parameters
obstacle = True
shape_params = np.array([4,4,4,1,0]) # uncomment for sphere

# wind MFS parameters
uinf = np.array([10,0,0]) # wind in x,y,z direction. Wind in z must be 0.
rs_wind = 0.2
rs_wind_int = 1.2
Ns_wind = 100 # currently also using this as Ns_wind_int. Investigate later.
Nb_wind = 120

# concentration MFS parameters
rs_conc = 0.8
cutoff = 35
rs_conc_int = .5
Nb_conc = 10
Ns_conc = 10
# Nb_conc_int = 60 # currently just using Nb_conc for this number. Investigate later.
Ns_conc_int = 10
sigma = .5

h = simulate(
    gpu,
    dx,
    Lx,
    Ly,
    Lz,
    stop_time,
    nsteps_per_second,
    D,
    gamma,
    c,

    # initial conditions for C
    n_copies,

    # obstacle parameters
    obstacle,
    shape_params,

    # wind MFS parameters
    uinf,
    rs_wind,
    rs_wind_int,
    Ns_wind,
    Nb_wind,

    # concentration MFS parameters
    rs_conc,
    cutoff,
    rs_conc_int,
    Nb_conc,
    Ns_conc,
    Ns_conc_int,
    sigma,
    )

## calculate naive concentration field ##
Cp = get_initial(h.X,h.Y,h.Z,Lx,Ly,source_location,source_spread,n_copies)

## evaluate leakage on eval points on the obstacle ##
# this part takes a while because a lot of eval pts
n_eval = 240
x0 = shape_params[0:3]
eval_pts = get_tdesign_points(n_eval)*shape_params[3]+x0
nnv_eval = get_27_nearest_vec(eval_pts, h.grid_pts, h.interior_mask,h.Nx,h.Nz)
nnv_coord_eval = np.zeros(np.shape(nnv_eval))
nnv_coord_eval[:,:,0] = h.xx[nnv_eval[:,:,2]]
nnv_coord_eval[:,:,1] = h.yy[nnv_eval[:,:,0]]
nnv_coord_eval[:,:,2] = h.zz[nnv_eval[:,:,1]]
eval_normals = get_normal_arr(eval_pts, shape_params)

print('getting AtA_invAt...')
big_A = np.zeros((27*len(eval_pts), 27*len(eval_pts)))
Xs = nnv_coord_eval - eval_pts[:,np.newaxis,:] # Xs[i,j,k] = (nnv[i,j] - surface_points[i])[k]
Y = np.zeros((len(eval_pts), 27, 27)) # Y[i,j,k] = (nnv_coord[i,j]-surface_point[i])[0]**(k%3) * (nnv_coord[i,j]-surface_point[i])[0]**((k//3)%3) * (nnv_coord[i,j]-surface_point[i])[0]**(k//9)
for k in np.arange(27):
    Y[:,:,k] = Xs[:,:,0]**(k%3) * Xs[:,:,1]**((k//3)%3) * Xs[:,:,2]**(k//9)
row_idxs = np.repeat(np.arange(len(eval_pts))*27, 27)
col_idxs = np.arange(27*len(eval_pts))
for j in np.arange(27):
    big_A[row_idxs + j, col_idxs] = Y[:,j,:].reshape((-1))
AtA_inv = np.linalg.pinv(big_A.transpose() @ big_A)
A_t = big_A.transpose()
AtA_invAt = AtA_inv @ A_t
print('got AtA_invAt')

## calculate leakage versus surface points ##
Nb_vec = np.array([144,180,204,240], dtype=int)
Ns_vec = np.array([100,144,180,204], dtype=int)
# Nb_vec = np.array([20,30], dtype=int)
# Ns_vec = np.array([10,20], dtype=int)
leakage_vec_new = np.zeros(len(Nb_vec))
for i in np.arange(len(Nb_vec)):
    Nb_conc = int(Nb_vec[i])
    Ns_conc = int(Ns_vec[i])
    h = simulate(
    gpu,
    dx,
    Lx,
    Ly,
    Lz,
    stop_time,
    nsteps_per_second,
    D,
    gamma,
    c,

    # initial conditions for C
    n_copies,

    # obstacle parameters
    obstacle,
    shape_params,

    # wind MFS parameters
    uinf,
    rs_wind,
    rs_wind_int,
    Ns_wind,
    Nb_wind,

    # concentration MFS parameters
    rs_conc,
    cutoff,
    rs_conc_int,
    Nb_conc,
    Ns_conc,
    Ns_conc_int,
    sigma,
    )

    M_eval,_ = precompute_mfs_matrices(h.source_points, h.source_points_int, eval_pts, sigma, h.alphasq, shape_params)
    Ch, leakages_new, leakages_Cp = get_Ch(
        Cp,
        h.surface_points,
        h.ik,h.il,h.Dcheb,
        h.nnv,h.AtA_invAt,
        shape_params,
        h.MtMinvMt,h.MinttMintinvMintt,
        h.Ch_support_mask, h.interior_mask,
        h.G_mat_surface, h.G_mat, h.G_mat_int,
        evaluate=True, M=h.M, M_eval=M_eval, eval_pts=eval_pts, nnv_eval=nnv_eval, AtA_invAt_eval=AtA_invAt,
           )
    
    leakage_vec_new[i] = norm(leakages_new)/norm(leakages_Cp)

pass1 = all(leakage_vec_new[1:] < leakage_vec_new[0:-1]) # check that it's getting lower
pass2 = all(leakage_vec_new < 1) # check that every step is an improvement
pass_flag = pass1 and pass2
if pass_flag:
    print('Test passed. relative leakages:')
    print(leakage_vec_new)
else:
    print('Test failed. relative leakages:')
    print(leakage_vec_new)