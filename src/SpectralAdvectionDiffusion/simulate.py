## new wind field code ##
import pickle
from datetime import datetime
import sys
from SpectralAdvectionDiffusion import *
import os
import time

gpu = False
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

class simulate:
    def __init__(
    # simulation parameters
    self,
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
    Ns_ratio,
    sigma,
    ):
        pi = np.pi
        norm = np.linalg.norm
        sphere_number_list = np.load('./data/points/sphere_number_list.npy')
        self.Nx = int(np.round(Lx / dx))
        if self.Nx%2:
            self.Nx = self.Nx+1
        self.Ny = int(np.round(Ly / dx))
        if self.Ny%2:
            self.Ny = self.Ny+1
        Lx = dx*self.Nx
        Ly = dx*self.Ny
        self.Nz = int(np.round(pi/(np.arcsin(2*dx/Lz))+1)) # this should make the ratio dx/dz close to 1 at the center.
        self.xx = dx*np.arange(1,self.Nx+1)
        self.yy = dx*np.arange(1,self.Ny+1)
        Dcheb0,zz0 = cheb(self.Nz-1)
        self.zz = Lz*0.5*(zz0+1)
        self.Dcheb = Dcheb0 / (Lz*0.5)
        self.Y,self.Z,self.X = np.meshgrid(self.yy,self.zz,self.xx, indexing='ij')
        print(f'grid shape: {np.shape(self.X)}')

        dt = 1/nsteps_per_second
        self.nsteps = int(np.round(stop_time/dt))
        self.dt = stop_time/self.nsteps # adjust dt to divide stop_time

        kk = np.zeros(self.Nx)
        for i in range(1,self.Nx//2):
            kk[i]=i
            kk[self.Nx-i]=-i
        self.kk = 2*pi*kk / Lx
        self.ik = 1j * self.kk

        ll = np.zeros(self.Ny)
        for i in range(1,self.Ny//2):
            ll[i]=i
            ll[self.Ny-i]=-i
        self.ll = 2*pi*ll / Ly
        self.il = 1j * self.ll

        kk = np.concatenate([np.arange(self.Nx/2+1), np.arange(-self.Nx/2+1,0)])
        kk = 2*pi*kk / Lx
        self.ksq = kk**2
        ll = np.concatenate([np.arange(self.Ny/2+1), np.arange(-self.Ny/2+1,0)])
        ll = 2*pi*ll / Ly
        self.lsq = ll**2

        if obstacle:
            shape = shape_params[-1]
            # get concentration field points
            if shape==0:
                # sphere
                Ns_conc_argmin = np.argmin(np.abs(Nb_conc*Ns_ratio - sphere_number_list))
                Ns_conc = int(sphere_number_list[Ns_conc_argmin])
                Ns_conc_int = Ns_conc
                x0 = np.array([shape_params[0],shape_params[1],shape_params[2]])
                r = shape_params[3]
                self.surface_points = get_tdesign_points(Nb_conc)*r + x0
                self.source_points = get_tdesign_points(Ns_conc)*rs_conc + x0
                self.source_points_int = get_tdesign_points(Ns_conc_int)*rs_conc_int + x0
            elif shape==1:
                # silo
                x0 = np.array([shape_params[0],shape_params[1],shape_params[1]*0])
                h = shape_params[2]
                r = shape_params[3]
                Ns_conc = Ns_ratio*Nb_conc/(.5+h/(2*rs_conc))
                Ns_conc_int = Ns_ratio*Nb_conc/(.5+h/(2*rs_conc_int))
                Ns_conc_argmin = np.argmin(np.abs(Ns_conc - sphere_number_list))
                Ns_conc_int_argmin = np.argmin(np.abs(Ns_conc_int - sphere_number_list))
                Ns_conc = sphere_number_list[Ns_conc_argmin]
                Ns_conc_int = sphere_number_list[Ns_conc_int_argmin]
                
                Nb_dome = Nb_conc
                Nb_cyl = int(np.round(h*Nb_dome/2/r))
                Ns_dome = Ns_conc
                Ns_cyl = int(np.round(h*Ns_dome/2/rs_conc))
                Ns_dome_int = Ns_conc_int
                Ns_cyl_int = int(np.round(h*Ns_dome_int/2/rs_conc_int))
                self.surface_points = get_silo_pts(Nb_cyl,Nb_dome,h,r,x0)
                self.source_points = get_silo_pts(Ns_cyl,Ns_dome,h,rs_conc,x0)
                self.source_points_int = get_silo_pts(Ns_cyl_int,Ns_dome_int,h,rs_conc_int,x0)

                print(f'Nb_dome, Nb_cyl: {Nb_dome/2,Nb_cyl}')
                print(f'Nb_dome/2pir^2, Nb_cyl/2pirh: {Nb_dome/2/(2*pi*r**2),Nb_cyl/(2*pi*r*h)}')
                print(f'Ns_dome, Ns_cyl: {Ns_dome/2,Ns_cyl}')
                print(f'Ns_dome/2pir^2, Ns_cyl/2pirh: {Ns_dome/2/(2*pi*rs_conc**2),Ns_cyl/(2*pi*rs_conc*h)}')

                print(f'total conc surface pts: {len(self.surface_points)}')
                print(f'total conc source pts: {len(self.source_points)}')
                print(f'total conc interior pts: {len(self.source_points_int)}')
            # get wind surface/source points
            if shape==0:
                # sphere
                Ns_wind_argmin = np.argmin(np.abs(Nb_wind*Ns_ratio - sphere_number_list))
                Ns_wind = int(sphere_number_list[Ns_wind_argmin])
                x0 = np.array([shape_params[0],shape_params[1],shape_params[2]])
                r = shape_params[3]
                self.surface_points_wind = get_tdesign_points(Nb_wind)*r + x0
                self.source_points_wind = get_tdesign_points(Ns_wind)*rs_wind + x0
                self.source_points_int_wind = get_tdesign_points(Ns_wind)*rs_wind_int + x0
            elif shape==1:
                # silo
                x0 = np.array([shape_params[0],shape_params[1],shape_params[1]*0])
                h = shape_params[2]
                r = shape_params[3]
                Ns_wind = Ns_ratio*Nb_wind/(.5+h/(2*rs_wind))
                Ns_wind_int = Ns_ratio*Nb_wind/(.5+h/(2*rs_wind_int))
                Ns_wind_argmin = np.argmin(np.abs(Ns_wind - sphere_number_list))
                Ns_wind_int_argmin = np.argmin(np.abs(Ns_wind_int - sphere_number_list))
                Ns_wind = sphere_number_list[Ns_wind_argmin]
                Ns_wind_int = sphere_number_list[Ns_wind_int_argmin]
                
                Nb_dome = Nb_wind
                Nb_cyl = int(np.round(h*Nb_dome/2/r))
                Ns_dome = Ns_wind
                Ns_cyl = int(np.round(h*Ns_dome/2/rs_wind))
                Ns_dome_int = Ns_wind_int
                Ns_cyl_int = int(np.round(h*Ns_wind_int/2/rs_wind_int))
                self.surface_points_wind = get_silo_pts(Nb_cyl,Nb_dome,h,r,x0)
                self.source_points_wind = get_silo_pts(Ns_cyl,Ns_dome,h,rs_wind,x0)
                self.source_points_int_wind = get_silo_pts(Ns_cyl_int,Ns_dome_int,h,rs_wind_int,x0)

                print(f'total wind surface pts: {len(self.surface_points_wind)}')
                print(f'total wind source pts: {len(self.source_points_wind)}')
                print(f'total wind interior pts: {len(self.source_points_int_wind)}')

        self.SIMat = SecondIntegralMatrix(self.Nz)
        self.SIMat_arr = self.SIMat[0:self.Nz,:].toarray()

        self.y_ids = np.array([np.array([nk*self.Nz,nk*self.Nz+1]) for nk in np.arange(self.Nx*self.Ny)]).reshape((-1))
        self.x_idxs = np.ravel(np.array([np.arange(self.Nz) + nk*(self.Nz+2) for nk in np.arange(self.Nx*self.Ny)]))
        self.y_idxs = np.ravel(np.array([np.array([self.Nz*nk + 2*(nk-1), self.Nz*nk + 2*(nk-1) + 1]) for nk in np.arange(1,self.Nx*self.Ny+1)]))

        self.lambdasq = 1/D/self.dt # used for solving at step 1
        self.alphasq = (gamma+.5)/(self.dt*(D*(gamma+c/2))) # used in steps n > 1

        self.lambda_k = np.repeat(np.array([self.lambdasq]), self.Nx*self.Ny).reshape((self.Nx,self.Ny)) + ((self.kk**2)[:,np.newaxis] + (self.ll**2)[np.newaxis, :])
        self.alpha_k = np.repeat(np.array([self.alphasq]), self.Nx*self.Ny).reshape((self.Nx,self.Ny)) + ((self.kk**2)[:,np.newaxis] + (self.ll**2)[np.newaxis, :])

        H = Lz/2
        print('factoring integral matrices...')
        start = time.time()

        SIDiag = g_block_diag([sparse.coo_matrix(self.SIMat[0:self.Nz,0:self.Nz])]*self.Nx*self.Ny, format='csc')
        bigA0 = sparse.diags(np.ones(self.Ny*self.Nx*self.Nz), dtype=complex) - H**2 * sparse.diags(np.repeat(self.lambda_k.transpose().reshape((-1)),self.Nz)) @ SIDiag
        bigAn = sparse.diags(np.ones(self.Ny*self.Nx*self.Nz), dtype=complex) - H**2 * sparse.diags(np.repeat(self.alpha_k.transpose().reshape((-1)),self.Nz)) @ SIDiag
        bigA0 = sparse.csc_matrix(bigA0)
        bigAn = sparse.csc_matrix(bigAn)
        print(f'time: {time.time()-start}')
        # print(f'err in bigAs: {np.sqrt((bigA0_old - bigA0).power(2).sum())}')

        self.bigAinv0 = splu(bigA0)
        self.bigAinvn = splu(bigAn)

        print('factoring boundary condition matrices...')

        BCs = BCRows(self.Nz)
        self.BCs = BCs
        
        start = time.time()

        lambda_k_long = np.transpose(self.lambda_k).ravel()
        alpha_k_long = np.transpose(self.alpha_k).ravel()
        k0_bool = lambda_k_long != 0
        kn_bool = alpha_k_long != 0
        block_SI_2 = sparse.vstack([sparse.coo_matrix(self.SIMat[0:self.Nz, self.Nz:(self.Nz+2)])]*self.Nx*self.Ny)
        bigB0 = -H**2 * sparse.diags(np.repeat(lambda_k_long, self.Nz)) @ block_SI_2
        bigBn = -H**2 * sparse.diags(np.repeat(alpha_k_long, self.Nz)) @ block_SI_2

        bigBC20 = np.vstack([BCs[1,:], H*BCs[2,:]]*self.Nx*self.Ny)
        even_rows = np.arange(self.Nx*self.Ny)*2
        bigBC20_even = k0_bool[:,np.newaxis]*(
            np.sqrt(lambda_k_long[:,np.newaxis]) * H**2 * np.vstack([BCs[1,:]]*self.Nx*self.Ny) + np.vstack([H*BCs[0,:] - BCs[1,:]]*self.Nx*self.Ny)
            )
        bigBC20[even_rows] = bigBC20[even_rows] + bigBC20_even

        bigBC2n = np.vstack([BCs[1,:], H*BCs[2,:]]*self.Nx*self.Ny)
        bigBC2n_even = kn_bool[:,np.newaxis]*(np.sqrt(alpha_k_long[:,np.newaxis])* H**2 * np.vstack([BCs[1,:]]*self.Nx*self.Ny) + np.vstack([H*BCs[0,:] - BCs[1,:]]*self.Nx*self.Ny))
        bigBC2n[even_rows] = bigBC2n[even_rows] + bigBC2n_even

        C0_list = bigBC20[:,0:self.Nz].reshape(self.Nx*self.Ny,2,self.Nz)
        self.block_C0 = g_block_diag(C0_list, format='csc')
        Cn_list = bigBC2n[:,0:self.Nz].reshape(self.Nx*self.Ny,2,self.Nz)
        self.block_Cn = g_block_diag(Cn_list, format='csc')

        self.big_M1_solve0 = self.bigAinv0.solve(bigB0.toarray())
        self.big_M1_solven = self.bigAinvn.solve(bigBn.toarray())
        self.bigM10 = self.block_C0 @ self.big_M1_solve0 - bigBC20[:,self.Nz:(self.Nz+2)]
        self.bigM1n = self.block_Cn @ self.big_M1_solven - bigBC2n[:,self.Nz:(self.Nz+2)]
        print(f'time: {time.time() - start}')

        self.Ux = np.zeros((self.Ny,self.Nz,self.Nx)) + uinf[0]
        self.Uy = np.zeros((self.Ny,self.Nz,self.Nx)) + uinf[1]
        self.Uz = np.zeros((self.Ny,self.Nz,self.Nx))

        if obstacle:
            print(f'factoring MFS matrices...')
            start = time.time()
            
            self.grid_pts = np.stack([self.X.ravel(), self.Y.ravel(), self.Z.ravel()], axis=1)
            self.interior_mask = is_interior_vec(self.grid_pts, shape_params)

            self.Ux,self.Uy,self.Uz = get_wind_field(self.xx,self.yy,self.zz,self.grid_pts,self.interior_mask,uinf,shape_params,
                                                    self.surface_points_wind, self.source_points_wind, self.source_points_int_wind)
            self.normals_conc = get_normal_arr(self.surface_points,shape_params)
            self.Ch_support_mask = in_Ch_support_vec(self.grid_pts, shape_params, self.alphasq, cutoff, self.interior_mask)
            self.nnv = get_27_nearest_vec(self.surface_points,self.grid_pts,self.interior_mask,self.Nx,self.Nz)
            self.nnv_coord = np.zeros(np.shape(self.nnv))
            self.nnv_coord[:,:,0] = self.xx[self.nnv[:,:,2]]
            self.nnv_coord[:,:,1] = self.yy[self.nnv[:,:,0]]
            self.nnv_coord[:,:,2] = self.zz[self.nnv[:,:,1]]

            self.dGdx,self.dGdy,self.dGdz = gradG_conc(self.grid_pts,self.source_points,np.sqrt(self.alphasq))
            self.dGdx = self.dGdx * (~self.interior_mask)[:,np.newaxis]
            self.dGdy = self.dGdy * (~self.interior_mask)[:,np.newaxis]
            self.dGdz = self.dGdz * (~self.interior_mask)[:,np.newaxis]

            self.dGdx_int,self.dGdy_int,self.dGdz_int = gradG_conc_int(self.grid_pts,self.source_points_int,sigma)
            self.dGdx_int = self.dGdx_int * (self.interior_mask)[:,np.newaxis]
            self.dGdy_int = self.dGdy_int * (self.interior_mask)[:,np.newaxis]
            self.dGdz_int = self.dGdz_int * (self.interior_mask)[:,np.newaxis]

            self.G_mat_surface = get_G_conc_mat(self.surface_points, self.source_points, self.alphasq) # use this for computing Ch on the surface
            self.G_mat = get_G_conc_mat(self.grid_pts, self.source_points, self.alphasq) # use this for computing Ch on domain
            self.G_mat_int = get_G_int_mat(self.grid_pts, self.source_points_int, sigma) # use this for computing Ch_int

            self.M,self.M_int = precompute_mfs_matrices(self.source_points, self.source_points_int, self.surface_points, sigma, self.alphasq, shape_params)
            print(f'condn number of M: {np.linalg.cond(self.M)}')
            self.MtMinvMt = np.linalg.pinv(self.M.transpose() @ self.M) @ self.M.transpose()
            self.MinttMintinvMintt = np.linalg.pinv(self.M_int.transpose() @ self.M_int) @ self.M_int.transpose()

            # prefactor mats for tri-quadratic interpolation
            big_A = np.zeros((27*len(self.surface_points), 27*len(self.surface_points)))
            Xs = self.nnv_coord - self.surface_points[:,np.newaxis,:] # Xs[i,j,k] = (nnv[i,j] - surface_points[i])[k]
            Y = np.zeros((len(self.surface_points), 27, 27)) # Y[i,j,k] = (nnv_coord[i,j]-surface_point[i])[0]**(k%3) * (nnv_coord[i,j]-surface_point[i])[0]**((k//3)%3) * (nnv_coord[i,j]-surface_point[i])[0]**(k//9)
            for k in np.arange(27):
                Y[:,:,k] = Xs[:,:,0]**(k%3) * Xs[:,:,1]**((k//3)%3) * Xs[:,:,2]**(k//9)
            row_idxs = np.repeat(np.arange(len(self.surface_points))*27, 27)
            col_idxs = np.arange(27*len(self.surface_points))
            for j in np.arange(27):
                big_A[row_idxs + j, col_idxs] = Y[:,j,:].reshape((-1))
            AtA_inv = np.linalg.pinv(big_A.transpose() @ big_A)
            A_t = big_A.transpose()
            self.AtA_invAt = AtA_inv @ A_t

            print(f'time: {time.time() - start}')

        self.sigma = sigma
        self.cutoff = cutoff
        self.rs_conc = rs_conc
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.shape_params = shape_params
        self.obstacle = obstacle
        self.gamma = gamma
        self.c = c
        self.D = D
        self.dx = dx
        self.stop_time = stop_time
        self.nsteps_per_second = nsteps_per_second
        self.n_copies = n_copies
        self.uinf = uinf
        self.rs_wind = rs_wind
        self.rs_wind_int = rs_wind_int
        self.Nb_wind = Nb_wind
        self.rs_conc_int = rs_conc_int
        self.Nb_conc = Nb_conc
        self.gpu = gpu

        print(f'np.sqrt(alphasq): {np.sqrt(self.alphasq)}')

    def run(self, source_location, source_spread, source_type, plotting=False):
        # set plotting = k, will cause the plot to be saved every k steps. Alsways saves the last step.
        self.sim_date = datetime.now().strftime("%Y-%m-%d--%H_%M_%S")
        print(f'sim timestamp: {self.sim_date}')
        if plotting:
            n_plots = np.sum(np.bitwise_or((np.arange(self.nsteps) % plotting) == 0, np.arange(self.nsteps) == self.nsteps-1))
            n_plots = int(n_plots)
            C_plots = np.zeros((self.Ny, self.Nz, self.Nx, n_plots))
            Cp_plots = np.zeros((self.Ny, self.Nz, self.Nx, n_plots))
            Ch_plots = np.zeros((self.Ny, self.Nz, self.Nx, n_plots))
            rhs_plots = np.zeros((self.Ny, self.Nz, self.Nx, n_plots))
            timestep_vals = np.zeros(n_plots)
        print('starting sim ...')
        self.source_location = source_location
        self.source_type = source_type
        self.source_spread = source_spread
        if source_type == 'puff':
            self.C_initial = get_initial(self.X,self.Y,self.Z,self.Lx,self.Ly,source_location,source_spread,self.n_copies) # initial concentration field
            self.Sn = 0*self.C_initial
            self.S0 = self.Sn
        elif source_type == 'plume':
            self.S0 = get_initial(self.X,self.Y,self.Z,self.Lx,self.Ly,source_location,source_spread,self.n_copies) / self.dt
            self.Sn = self.S0
            self.C_initial = 0*self.Sn

        for n in np.arange(self.nsteps):

            if n == 0:
                self.C = self.C_initial
                Cp = self.C
                Ch = 0*Cp
                self.rhs = 0*Cp

            if plotting:
                if n % plotting == 0:
                    C_plots[:,:,:,n//plotting] = self.C
                    Cp_plots[:,:,:,n//plotting] = Cp
                    Ch_plots[:,:,:,n//plotting] = Ch
                    timestep_vals[n//plotting] = n*self.dt
                    rhs_plots[:,:,:,n//plotting] = self.rhs
                if n == self.nsteps-1:
                    C_plots[:,:,:,-1] = self.C
                    Cp_plots[:,:,:,-1] = Cp
                    Ch_plots[:,:,:,-1] = Ch
                    timestep_vals[-1] = n*self.dt
                    rhs_plots[:,:,:,n//plotting] = self.rhs

            if n == 0:
                Cp = self.get_first_step()
                self.C = np.real(Cp)
                Ch = 0*Cp # for plotting purposes
                
            else:
                # for n > 0
                start = time.time()
                if self.obstacle:
                    # if we need to calculate Ch, take deriv of Cp and Ch separately
                    Cp = self.step_forward_2part(C_lag1,C_lag2,Cp_lag1,Cp_lag2,alpha_lag1,alpha_lag2,alpha_int_lag1,alpha_int_lag2,Ux_lag1,Uy_lag1,Uz_lag1,Ux_lag2,Uy_lag2,Uz_lag2,S_lag1,S_lag2)
                else:
                    Cp = self.step_forward(C_lag1,C_lag2,Ux_lag1,Uy_lag1,Uz_lag1,Ux_lag2,Uy_lag2,Uz_lag2,S_lag1,S_lag2)
                Cp = np.real(Cp)
                # print(f'Cp time: {time.time()-start}')
                
                if self.obstacle:
                    # print(f'norm(Cp): {norm(Cp)}')
                    start = time.time()
                    alpha,alpha_int,Ch = self.get_Ch_method(Cp)
                    ### DELETE THESE LINES ###
                    # alpha = alpha*0
                    # Ch = Ch*0
                    ### ###

                    # print(f'Ch time: {time.time()-start}')
                    # print(f'norm(Ch): {norm(Ch)}')
                    self.C = Cp + Ch

                else:
                    self.C = Cp

            if n == 0:
                C_lag2 = self.C_initial
                S_lag2 = self.S0
                C_lag1 = self.C
                S_lag1 = self.get_Sn(n)
                Ux_lag2 = self.Ux
                Uy_lag2 = self.Uy
                Uz_lag2 = self.Uz
                Ux_lag1,Uy_lag1,Uz_lag1 = self.get_Un(n)
                if self.obstacle:
                    Cp_lag2 = C_lag2
                    Cp_lag1 = C_lag1
                    alpha_lag2 = np.zeros(len(self.source_points))
                    alpha_lag1 = np.zeros(len(self.source_points))
                    alpha_int_lag2 = np.zeros(len(self.source_points_int))
                    alpha_int_lag1 = np.zeros(len(self.source_points_int))

            else:
                C_lag2 = C_lag1
                S_lag2 = S_lag1
                C_lag1 = self.C
                S_lag1 = self.get_Sn(n)
                Ux_lag2 = Ux_lag1
                Uy_lag2 = Uy_lag1
                Uz_lag2 = Uz_lag1
                Ux_lag1,Uy_lag1,Uz_lag1 = self.get_Un(n)
                if self.obstacle:
                    Cp_lag2 = Cp_lag1
                    Cp_lag1 = Cp
                    alpha_lag2 = alpha_lag1
                    alpha_lag1 = alpha
                    alpha_int_lag2 = alpha_int_lag1
                    alpha_int_lag1 = alpha_int

            print(f'--------------------- end of step {n} ---------------------')
        print('end of sim')
        print(f'sim date: {self.sim_date}')
        if plotting:
            if not os.path.isdir('./data/plots/'):
                os.mkdir('./data/plots/')
            if self.gpu:
                os.mkdir(f'./data/plots/{self.sim_date}/')
                np.save(f'./data/plots/{self.sim_date}/C_plots',C_plots.get())
                np.save(f'./data/plots/{self.sim_date}/Cp_plots',Cp_plots.get())
                np.save(f'./data/plots/{self.sim_date}/Ch_plots',Ch_plots.get())
                np.save(f'./data/plots/{self.sim_date}/rhs_plots',rhs_plots.get())
                np.save(f'./data/plots/{self.sim_date}/timestep_vals',timestep_vals.get())
                np.save(f'./data/plots/{self.sim_date}/xx',self.xx.get())
                np.save(f'./data/plots/{self.sim_date}/yy',self.yy.get())
                np.save(f'./data/plots/{self.sim_date}/zz',self.zz.get())
                np.save(f'./data/plots/{self.sim_date}/shape_params',self.shape_params.get())
                np.save(f'./data/plots/{self.sim_date}/obstacle',self.obstacle)
            else:
                os.mkdir(f'./data/plots/{self.sim_date}/')
                np.save(f'./data/plots/{self.sim_date}/C_plots',C_plots)
                np.save(f'./data/plots/{self.sim_date}/Cp_plots',Cp_plots)
                np.save(f'./data/plots/{self.sim_date}/Ch_plots',Ch_plots)
                np.save(f'./data/plots/{self.sim_date}/rhs_plots',rhs_plots)
                np.save(f'./data/plots/{self.sim_date}/timestep_vals',timestep_vals)
                np.save(f'./data/plots/{self.sim_date}/xx',self.xx)
                np.save(f'./data/plots/{self.sim_date}/yy',self.yy)
                np.save(f'./data/plots/{self.sim_date}/zz',self.zz)
                np.save(f'./data/plots/{self.sim_date}/shape_params',self.shape_params)
                np.save(f'./data/plots/{self.sim_date}/obstacle',self.obstacle)

    def get_first_step(self):
        rhs = get_rhs_1step(self.C_initial,self.dt,self.D,self.Ux,self.Uy,self.Uz,self.Dcheb,self.ik,self.il) - self.S0/self.D
        self.rhs = rhs
        return self.solve_modified_helmholtz(rhs,
                                             self.lambda_k,
                                             self.bigAinv0,
                                             self.block_C0,
                                             self.bigM10)
    
    def step_forward(self,C_lag1,C_lag2,Ux_lag1,Uy_lag1,Uz_lag1,Ux_lag2,Uy_lag2,Uz_lag2,S_lag1,S_lag2):
        rhs = get_rhs_2step(C_lag1,
                            C_lag2,
                            self.dt,
                            self.D,
                            Ux_lag1,
                            Uy_lag1,
                            Uz_lag1,
                            Ux_lag2,
                            Uy_lag2,
                            Uz_lag2,
                            self.gamma,
                            self.c,
                            self.Dcheb,
                            S_lag1,
                            S_lag2,
                            self.ik,
                            self.il,
                            self.ksq,
                            self.lsq)
        self.rhs = rhs
        return self.solve_modified_helmholtz(rhs,
                                             self.alpha_k,
                                             self.bigAinvn,
                                             self.block_Cn,
                                             self.bigM1n)
    
    def step_forward_2part(self,C_lag1,C_lag2,Cp_lag1,Cp_lag2,alpha_lag1,alpha_lag2,alpha_int_lag1,alpha_int_lag2,Ux_lag1,Uy_lag1,Uz_lag1,Ux_lag2,Uy_lag2,Uz_lag2,S_lag1,S_lag2):
        rhs = get_rhs_2step_2part(C_lag1,C_lag2,Cp_lag1,Cp_lag2,alpha_lag1,alpha_lag2,self.dGdx,self.dGdy,self.dGdz,self.dGdx_int,self.dGdy_int,self.dGdz_int,alpha_int_lag1,alpha_int_lag2,
                        self.dt,self.D,Ux_lag1,Uy_lag1,Uz_lag1,Ux_lag2,Uy_lag2,Uz_lag2,self.gamma,self.c,self.Dcheb,S_lag1,S_lag2,self.ik,self.il,self.ksq,self.lsq)
        self.rhs = rhs
        return self.solve_modified_helmholtz(rhs,
                                             self.alpha_k,
                                             self.bigAinvn,
                                             self.block_Cn,
                                             self.bigM1n)
    
    def solve_modified_helmholtz(self,rhs,lambda_k,Ainv,block_C,bigM1):
        return InvYukawa(rhs,
                         lambda_k,
                         self.SIMat_arr,
                         self.Lz,
                         self.Nz,
                         self.Nx,
                         self.Ny,
                         Ainv,
                         block_C,
                         bigM1,
                         self.y_ids,
                         self.x_idxs,
                         self.y_idxs)
    
    def get_Sn(self,n):
        # should return source term for the n+1 step
        return self.Sn
    
    def get_Un(self,n):
        # should return wind term for the n+1 step
        return self.Ux, self.Uy, self.Uz
    
    def get_Ch_method(self,C):
        return get_Ch(
        C,
        self.surface_points,
        self.ik,self.il,self.Dcheb,
        self.nnv,self.AtA_invAt,
        self.shape_params,self.normals_conc,
        self.MtMinvMt,self.MinttMintinvMintt,
        self.Ch_support_mask, self.interior_mask,
        self.G_mat_surface, self.G_mat, self.G_mat_int
        )
    
    