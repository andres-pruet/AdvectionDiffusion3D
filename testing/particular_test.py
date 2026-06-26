'''
This code compares the obstacle-free solver with the analytic solution for advection-diffusion of a Gaussian puff in 2d.
As it turns out, the analytic solution for a Gaussian puff is always Gaussian.
'''

gpu = False # make sure this matches other 4 files
if gpu:
    import cupy as np
else:
    import numpy as np
import sys
sys.path.append('../src/')
from SpectralAdvectionDiffusion import *
from matplotlib import pyplot as plt

## parameters ##

## use these as canonical params for other tests ##
# grid setup
dx = 1/6
Lx = 10
Ly = 8
Lz = 12
stop_time = 1.
nsteps_per_second = 140
D = 1
gamma = 1.
c = 0

# initial conditions for C
source_location = np.array([5,4,6])
source_spread = 0.5
source_type = 'puff'
n_copies = 20

# obstacle parameters
# shape dict: 0 for sphere, 
obstacle = False
shape_params = np.array([8,4,6,1,0]) # should be [centerx,centery,centerz,radius,shape_id]

# wind MFS parameters
uinf = np.array([10,0,0]) # wind in x,y,z direction. Wind in z must be 0.
rs_wind = 0.2
rs_wind_int = 1.2
Ns_wind = 100
Nb_wind = 120

# concentration MFS parameters
rs_conc = 0.8
cutoff = 35
rs_conc_int = .5
Nb_conc = 60
Ns_conc = 50
# Nb_conc_int = 60 # currently just using Nb_conc for this number
Ns_conc_int = 20
sigma = 0.9*dx

nsteps_per_second_array = np.array([80,160,320])

Nx = int(np.round(Lx / dx))
if Nx%2:
    Nx = Nx+1
Ny = int(np.round(Ly / dx))
if Ny%2:
    Ny = Ny+1
Nz = int(np.round(pi/(np.arcsin(2*dx/Lz))+1)) # this should make the ratio dx/dz close to 1 at the center.
xx = dx*np.arange(1,Nx+1)
yy = dx*np.arange(1,Ny+1)
Dcheb0,zz0 = cheb(Nz-1)
zz = Lz*0.5*(zz0+1)
Y,Z,X = np.meshgrid(yy,zz,xx, indexing='ij')

## get analytic solution at stop_time ##
pi = np.pi
norm = np.linalg.norm
Dt0 = (source_spread)**2/2/D
def c_star(x,y,z,t,xs,ys,zs,u,Dt0):
    # these functions are specifically for 2 dimensions
    return (1/(2*pi*2*(t+Dt0))**(3/2))*np.exp( -((x - xs - u*t)**2 + (y - ys)**2 + (z - zs)**2)/(4*(t + Dt0)) )

def puff_analytic(x, y, z, t, source_location, u, Dt0):
    xs,ys,zs = source_location
    return c_star(x,y,z,t,xs,ys,zs,u,Dt0) + c_star(x,y,-z,t,xs,ys,zs,u,Dt0)

C_analytic = np.zeros((Ny, Nz, Nx))
for x_idx in np.arange(-n_copies,n_copies+1):
    for y_idx in np.arange(-n_copies,n_copies+1):
        C_analytic = C_analytic + puff_analytic(X-x_idx*Lx, Y-y_idx*Ly, Z, stop_time, source_location, uinf[0], Dt0)

# sys.exit()

## compare for various timestep sizes, make sure convergence is quadratic. ##
error_vec = np.zeros(len(nsteps_per_second_array))
dt_vec = np.zeros(len(nsteps_per_second_array))
for i,nsteps_per_second in enumerate(nsteps_per_second_array):
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
    h.run(source_location, source_spread, source_type, plotting=0)
    # fig,axs = plt.subplots(1,2)
    # scat1 = axs[0].pcolor(h.xx.get(),h.zz.get(),h.C[16,:,:].get())
    # cbar1 = plt.colorbar(scat1)
    # scat2 = axs[1].pcolor(h.xx.get(),h.zz.get(),(C_analytic-h.C)[16,:,:].get())
    # cbar2 = plt.colorbar(scat2)
    # plt.show()
    err = norm(h.C - C_analytic)/norm(C_analytic)
    print(f'dt: {h.dt}')
    print(f'norm(C): {norm(h.C)}')
    print(f'norm(C_analytic): {norm(C_analytic)}')
    print(f'ERR: {err}')
    error_vec[i] = err
    dt_vec[i] = h.dt

print(f'errors: {error_vec}')

## verify that error has quadratic convergence ##
relative_vec = (error_vec[0]/dt_vec[0]**2) * dt_vec**2
pass_flag = np.allclose(relative_vec, error_vec, rtol = 0.01, atol = 1e-04)
if pass_flag:
    print('Test passed.')
else:
    print('Test failed: ')
    print(f'absolute error {np.max(np.abs(relative_vec-error_vec))}')
    print(f'relative error: {np.max(np.abs(relative_vec-error_vec)) / np.max(relative_vec)}')

## plot the error as a function of dt. Should be order dt^2 **
if gpu:
    error_vec = error_vec.get()
    dt_vec = dt_vec.get()
plt.plot(dt_vec,error_vec,label='relative RMSE')
plt.plot(dt_vec,(error_vec[0]/dt_vec[0]**2) * dt_vec**2,label='c*dt^2')
plt.yscale('log')
plt.legend()
plt.show()