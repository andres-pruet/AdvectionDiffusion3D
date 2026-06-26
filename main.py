'''
This code compares the obstacle-free solver with the analytic solution for advection-diffusion of a Gaussian puff in 2d.
As it turns out, the analytic solution for a Gaussian puff is always Gaussian.
'''

gpu = False # make sure this matches other 3 files
if gpu:
    import cupy as np
else:
    import numpy as np
import sys
import time
sys.path.append('../src/')
from SpectralAdvectionDiffusion import *
from matplotlib import pyplot as plt

## parameters ##

# Plotting parameters
plotting = 1

# General MFS parameters
Ns_ratio = 5/6

# grid setup
dx = 1/4
Lx = 12
Ly = 8
Lz = 8
stop_time = 1.
nsteps_per_second = 150
D = 1
gamma = 1
c = 0

# initial conditions for C
source_location = np.array([2,4,4])
source_spread = 0.5
source_type = 'puff'
n_copies = 10

# obstacle parameters
obstacle = True
# shape_params = np.array([8,4,6,2,0]) # example parameters for sphere
shape_params = np.array([6,4,4,2,1]) # example parameters for silo

# wind MFS parameters
uinf = np.array([10,0,0]) # wind in x,y,z direction. Wind in z must be 0.
rs_wind = shape_params[3]-.8 # only gonna work if r > 1.0
rs_wind_int = shape_params[3]+.2
Nb_wind = 328

# concentration MFS parameters
rs_conc = shape_params[3]-.5
cutoff = 35
rs_conc_int = shape_params[3]-.5
Nb_conc = 240
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
    Nb_wind,

    # concentration MFS parameters
    rs_conc,
    cutoff,
    rs_conc_int,
    Nb_conc,
    Ns_ratio, # affects both concentration and wind source points. Investigate this.
    sigma,
    )

sim_start_time = time.time()

h.run(source_location, source_spread, source_type, plotting)

print(f'sim runtime: {time.time()-sim_start_time}')

print(f'norm C: {np.linalg.norm(h.C)}')