gpu = False # make sure this matches in the other files
from SpectralAdvectionDiffusion import get_x_deriv,get_y_deriv,get_z_deriv
if gpu:
    print('using gpu')
    import cupy as np
    from cupy.linalg import norm
    from cupyx.scipy.special import k1
    def njit(fastmath=0):
        def decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator
else:
    print('not using gpu')
    import numpy as np
    from numpy.linalg import norm
    from scipy.special import k1
    from numba import njit

pi = np.pi

def get_tdesign_points(n):
    pts = np.load(f'./data/points/{n}_tsphere.npy')
    return pts

def get_circle_pts(n,r,dth):
    pts = np.zeros((n,3))
    theta = np.arange(n)*2*pi/n
    x = np.cos(theta + dth)
    y = np.sin(theta + dth)
    pts[:,0] = x
    pts[:,1] = y
    pts = pts*r
    return pts

## old helix formula
# def get_cylinder_pts(n,h,r):
#     # a hollow cylinder, might not be stable at the caps
#     # helix method
#     nrev = np.sqrt(h*n/(2*pi))
#     theta = np.linspace(0,2*pi*nrev,n)
#     dh_vec = np.linspace(0,h,n)
#     pts = np.stack([
#         r*np.cos(theta),
#         r*np.sin(theta),
#         dh_vec
#     ], axis=1)
#     return pts

def get_cylinder_pts(n,h,r):
    # a hollow cylinder, might not be stable at the caps
    # adds dh/2 to the height of each ring
    dh = (2*pi*h/n)**(1/2)
    nr = int(np.round(h/dh))
    nc = int(np.round(n/nr))
    n = nr*nc
    pts = np.zeros((n,3))
    for i in np.arange(nr):
        pts[(i*nc):((i+1)*nc)] = get_circle_pts(nc,r,(i%2)*pi/nc) + np.array([0*dh,0*dh,i*dh])
    pts = pts + np.array([0*dh,0*dh,dh/2])
    return pts

def get_silo_pts(n1,n2,h,r,x0):
    '''
    n1: num of pts in cylinder part
    n2: num of pts in hemisphere part
    h: height of cylinder (not including hemisphere)
    r: radius
    x0: location (x,y,z) of center of bottom of silo
    '''
    pts1 = get_cylinder_pts(n1,h,r)
    sph_pts = get_tdesign_points(n2)
    idxs = np.where(sph_pts[:,2]>0)
    pts2 = sph_pts[idxs]
    pts2 = pts2*r + np.array([h*0,h*0,h])
    pts = np.vstack([pts1,pts2])
    pts = pts+x0
    return pts
    
def solve_ols(M,y):
    return np.linalg.lstsq(M,y)[0]
    
def gradG_wind(x,xj):
    # should take a vector of points to evaluate (x) and a vector of source points (xj) and return 3 matrices: dGdx, dGdy, dGdz.
    # G = 1/||x|| -> dGdxi = -xi/||x||^3
    diffs = x[:, np.newaxis, :] - xj[np.newaxis, :, :] # diffs_inm = x_im - xj_nm
    Rsq = (diffs**2).sum(axis=2)
    Rsq[Rsq==0]=1 # to avoid grid_pts same as source_points. These get set to 0 later anyways.
    x_mirr = np.stack([x[:,0], x[:,1], -x[:,2]], axis=1)
    diffs_mirr = x_mirr[:, np.newaxis, :] - xj[np.newaxis, :, :]
    Rsq_mirr = (diffs_mirr**2).sum(axis=2)
    Rsq_mirr[Rsq_mirr==0]=1

    dGdx = -diffs[:,:,0]/Rsq**(3/2) - diffs[:,:,0]/Rsq_mirr**(3/2)
    dGdy = -diffs[:,:,1]/Rsq**(3/2) - diffs[:,:,1]/Rsq_mirr**(3/2)
    dGdz = -diffs[:,:,2]/Rsq**(3/2) + diffs_mirr[:,:,2]/Rsq_mirr**(3/2)
    return dGdx, dGdy, dGdz

def gradG_conc(x,xj,k):
    # should take a vector of points to evaluate (x) and a vector of source points (xj) and return 3 matrices: dGdx, dGdy, dGdz.
    # G = exp(-k||x-xj||)/||x-xj||
    diffs = x[:, np.newaxis, :] - xj[np.newaxis, :, :] # diffs_inm = x_im - xj_nm
    R = np.sqrt((diffs**2).sum(axis=2))
    R[R==0]=1 # to avoid grid_pts same as source_points. These get set to 0 later anyways.
    x_mirr = np.stack([x[:,0], x[:,1], -x[:,2]], axis=1)
    diffs_mirr = x_mirr[:, np.newaxis, :] - xj[np.newaxis, :, :]
    R_mirr = np.sqrt((diffs_mirr**2).sum(axis=2))
    R_mirr[R_mirr==0]=1
    dGdx = -diffs[:,:,0]*np.exp(-k*R)*(k/R**2 + 1 / R**3) - diffs_mirr[:,:,0] * np.exp(-k*R_mirr) * (k/R_mirr**2 + 1/R_mirr**3)
    dGdy = -diffs[:,:,1]*np.exp(-k*R)*(k/R**2 + 1 / R**3) - diffs_mirr[:,:,1] * np.exp(-k*R_mirr) * (k/R_mirr**2 + 1/R_mirr**3)
    dGdz = -diffs[:,:,2]*np.exp(-k*R)*(k/R**2 + 1 / R**3) + diffs_mirr[:,:,2] * np.exp(-k*R_mirr) * (k/R_mirr**2 + 1/R_mirr**3)
    return dGdx, dGdy, dGdz

def gradG_conc_int(x,xj,sigma):
    diffs = x[:, np.newaxis, :] - xj[np.newaxis, :, :] # diffs_inm = x_im - xj_nm
    Rsq = (diffs**2).sum(axis=2)
    Rsq[Rsq==0]=1 # to avoid grid_pts same as source_points. These get set to 0 later anyways.
    G = np.exp(-Rsq / (2*sigma**2))
    dGdx_int = -(diffs[:,:,0]/sigma**2)*G
    dGdy_int = -(diffs[:,:,1]/sigma**2)*G
    dGdz_int = -(diffs[:,:,2]/sigma**2)*G
    return dGdx_int,dGdy_int,dGdz_int

def get_normal_arr(x, shape_params):
    shape = shape_params[-1]
    if shape==0:
        x0 = np.array([shape_params[0], shape_params[1], shape_params[2]])
        normals = (x - x0)/norm(x - x0, axis=1)[:,np.newaxis]
    elif shape==1:
        # relies on knowing something about the order of the points.
        x0 = np.array([shape_params[0],shape_params[1],shape_params[1]*0])
        h = shape_params[2]
        r = shape_params[3]
        n1 = np.sum(x[:,2]<h)
        n2 = len(x)-n1
        normals = np.zeros(np.shape(x))
        normals[0:n1, 0:2] = ((x - x0)/r)[0:n1, 0:2]
        normals[n1:(n1+n2)] = (x[n1:(n1+n2)] - np.array([h*0,h*0,h]) - x0)/r

    return normals

def get_G_conc_mat(x,xj,ksq):
    k = np.sqrt(ksq)
    diffs = x[:, np.newaxis, :] - xj[np.newaxis, :, :] # diffs_inm = x_im - xj_nm
    Rsq = (diffs**2).sum(axis=2)
    R = np.sqrt(Rsq)
    R[R==0]=1

    x_mirr = np.stack([x[:,0], x[:,1], -x[:,2]], axis=1)
    diffs_mirr = x_mirr[:, np.newaxis, :] - xj[np.newaxis, :, :]
    Rsq_mirr = (diffs_mirr**2).sum(axis=2)
    R_mirr = np.sqrt(Rsq_mirr)
    R_mirr[R_mirr==0]=1

    return np.exp(-k*R)/R + np.exp(-k*R_mirr)/R_mirr

def get_G_int_mat(x, xj, sigma):
    diffs = x[:, np.newaxis, :] - xj[np.newaxis, :, :]
    Rsq = (diffs**2).sum(axis=2)
    Rsq[Rsq==0]=1
    return np.exp(-Rsq / (2*sigma**2))

def get_wind_field(xx,yy,zz,grid_pts,interior_mask,uinf,shape_params,surface_points, source_points, source_points_int):
    # by default grid_pts is 2-by-N
    Nx = len(xx)
    Ny = len(yy)
    Nz = len(zz)

    dGdx_mat_surface, dGdy_mat_surface, dGdz_mat_surface = gradG_wind(surface_points, source_points)

    normal_arr = get_normal_arr(surface_points, shape_params)
    normal_arr_x = normal_arr[:,0]
    normal_arr_y = normal_arr[:,1]
    normal_arr_z = normal_arr[:,2]
    normal_mat_x = np.repeat(normal_arr_x,len(source_points)).reshape(np.shape(dGdx_mat_surface))
    normal_mat_y = np.repeat(normal_arr_y,len(source_points)).reshape(np.shape(dGdx_mat_surface))
    normal_mat_z = np.repeat(normal_arr_z,len(source_points)).reshape(np.shape(dGdx_mat_surface))
    M = dGdx_mat_surface * normal_mat_x + dGdy_mat_surface * normal_mat_y + dGdz_mat_surface * normal_mat_z

    y = -uinf[0] * normal_arr_x - uinf[1] * normal_arr_y # remember uinf[2] = 0
    alpha = solve_ols(M,y)

    M_int1,M_int2,M_int3 = gradG_wind(surface_points, source_points_int)

    y_int1 = dGdx_mat_surface @ alpha + uinf[0]
    y_int2 = dGdy_mat_surface @ alpha + uinf[1]
    y_int3 = dGdz_mat_surface @ alpha

    alpha_int_x = solve_ols(M_int1,y_int1)
    alpha_int_y = solve_ols(M_int2,y_int2)
    alpha_int_z = solve_ols(M_int3,y_int3)

    dGdx_mat,dGdy_mat,dGdz_mat = gradG_wind(grid_pts, source_points)
    Ux = np.reshape((dGdx_mat @ alpha + uinf[0])*(~interior_mask), (Ny,Nz,Nx))
    Uy = np.reshape((dGdy_mat @ alpha + uinf[1])*(~interior_mask), (Ny,Nz,Nx))
    Uz = np.reshape((dGdz_mat @ alpha)*(~interior_mask), (Ny,Nz,Nx))

    dGdx_mat_int, dGdy_mat_int, dGdz_mat_int = gradG_wind(grid_pts, source_points_int)

    Ux_int = np.reshape((dGdx_mat_int @ alpha_int_x)*(interior_mask), (Ny,Nz,Nx))
    Uy_int = np.reshape((dGdy_mat_int @ alpha_int_y)*(interior_mask), (Ny,Nz,Nx))
    Uz_int = np.reshape((dGdz_mat_int @ alpha_int_z)*(interior_mask), (Ny,Nz,Nx))

    Ux = Ux + Ux_int
    Uy = Uy + Uy_int
    Uz = Uz + Uz_int
    return Ux, Uy, Uz

def precompute_mfs_matrices(source_points, source_points_int, surface_points, sigma, alphasq, shape_params):
    Ns_conc = len(source_points)

    dGdx_mat_surface, dGdy_mat_surface, dGdz_mat_surface = gradG_conc(surface_points, source_points, np.sqrt(alphasq))
    normal_arr = get_normal_arr(surface_points, shape_params)
    normal_arr_x = normal_arr[:,0]
    normal_arr_y = normal_arr[:,1]
    normal_arr_z = normal_arr[:,2]
    normal_mat_x = np.repeat(normal_arr_x,Ns_conc).reshape(np.shape(dGdx_mat_surface))
    normal_mat_y = np.repeat(normal_arr_y,Ns_conc).reshape(np.shape(dGdx_mat_surface))
    normal_mat_z = np.repeat(normal_arr_z,Ns_conc).reshape(np.shape(dGdx_mat_surface))
    M = dGdx_mat_surface * normal_mat_x + dGdy_mat_surface * normal_mat_y + dGdz_mat_surface * normal_mat_z

    M_int = get_G_int_mat(surface_points,source_points_int,sigma)

    return(M, M_int)

def is_interior_vec(grid_pts, params):
    shape = params[-1]
    if shape==0:
        x0 = np.array([params[0],params[1],params[2]])
        r = params[3]
        dists = np.sqrt(((grid_pts - x0)**2).sum(axis=1))
        return dists < r
    elif shape==1:
        x0 = np.array([params[0],params[1],params[1]*0])
        h = params[2]
        r = params[3]
        return ((grid_pts[:,2]<h) & (norm(np.stack([
            grid_pts[:,0],grid_pts[:,1],grid_pts[:,1]*0
            ],axis=1) - x0, axis=1) < r)) | ((grid_pts[:,2] >= h) & (norm(grid_pts-np.array([params[0],params[1],params[2]]),axis=1)<r) )
    
def in_Ch_support_vec(pts, params, lambdasq, cutoff, interior_mask):
    shape = params[-1]
    if shape==0:
        x0 = np.array([params[0],params[1],params[2]])
        r = params[3]
        dists = np.sqrt(((pts - x0)**2).sum(axis=1))
        mask = interior_mask | (dists > r + cutoff/np.sqrt(lambdasq))
        return ~mask

def get_27_nearest_vec(surface_points,grid_pts,interior_mask,Nx,Nz):
    # surface_pts in (Ns,3), grid_pts in (N,3)
    # actually takes 27 nearest
    # returns Ns-by-27-by-3 grid, the dimensions are y_idx,z_idx,x_idx.
    sq = surface_points[:, np.newaxis, :] - grid_pts[np.newaxis, :, :]  # (Ns, N, 3)
    dists = np.sqrt((sq ** 2).sum(axis=2))                               # (Ns, N)
    large = np.finfo(np.float64).max / 2
    mask = interior_mask[np.newaxis, :] | (dists == 0.0)   # (Ns, N)
    dists = np.where(mask, large, dists) # set masked values to large.
    order = np.argsort(dists, axis=1)[:, :27]               # (Ns, 27)

    # ── Convert flat indices back to (z_idx, x_idx) pairs ─────────────────
    x_indices = order % Nx
    y_indices = order // (Nx*Nz)
    z_indices = (order // Nx) % Nz

    indices = np.stack([y_indices, z_indices, x_indices], axis=2)
    return indices.astype(int)

def triquad_interp(C,nnv,AtA_invAt):
    # nnv should be Ns-by-27-by-3.
    # it has [y-coord,z-coord,x-coord]
    Ns = len(nnv)
    big_y = C[nnv[:,:,0], nnv[:,:,1], nnv[:,:,2]].reshape((-1))

    beta = AtA_invAt @ big_y
    f = beta[np.arange(Ns)*27]

    return f

def get_Ch(
        Cp,
        surface_points,
        ik,il,Dcheb,
        nnv,AtA_invAt,
        shape_params,normals,
        MtMinvMt,MinttMintinvMintt,
        Ch_support_mask, interior_mask,
        G_mat_surface, G_mat, G_mat_int,
        evaluate=False, M=None, M_eval=None, eval_pts=None, nnv_eval=None, AtA_invAt_eval=None,
           ):
    
    dim = (len(Cp),len(Cp[0]),len(Cp[0,0]))
    Ny,Nz,Nx = dim

    dCdx = np.real(get_x_deriv(Cp,ik))
    dCdy = np.real(get_y_deriv(Cp,il))
    dCdz = np.real(get_z_deriv(Cp, Dcheb))

    dCdx_interpolated = triquad_interp(dCdx,nnv,AtA_invAt)
    dCdy_interpolated = triquad_interp(dCdy,nnv,AtA_invAt)
    dCdz_interpolated = triquad_interp(dCdz,nnv,AtA_invAt)

    y = -dCdx_interpolated*normals[:,0] - dCdy_interpolated*normals[:,1] - dCdz_interpolated*normals[:,2] # this is flux against the surface of the obstacle, assuming u dot n = 0

    alpha = MtMinvMt @ y # to make gradG dot n = 0

    y_int = G_mat_surface @ alpha
    
    alpha_int = MinttMintinvMintt @ y_int
    ### DELETE ###
    # alpha_int = np.zeros(len(MinttMintinvMintt))
    
    ## now get Ch to return ##

    Ch = (G_mat @ alpha)*(~interior_mask) + (G_mat_int @ alpha_int)*(interior_mask)
    # Ch = (G_mat @ alpha)*(~interior_mask) # use this if not calculating derivative of Ch_int in rhs function
    Ch = np.reshape(Ch, (Ny,Nz,Nx))
    
    if evaluate:
        print(f'evaluating...')
        print(f'||Ma - y||/||y||: {norm(M @ alpha - y)/norm(y)}')
        
        eval_normals = get_normal_arr(eval_pts, shape_params)
        dCdx_interpolated_eval = triquad_interp(dCdx,nnv_eval,AtA_invAt_eval)
        dCdy_interpolated_eval = triquad_interp(dCdy,nnv_eval,AtA_invAt_eval)
        dCdz_interpolated_eval = triquad_interp(dCdz,nnv_eval,AtA_invAt_eval)
        leakages_Cp = dCdx_interpolated_eval*eval_normals[:,0] + dCdy_interpolated_eval*eval_normals[:,1] + dCdz_interpolated_eval*eval_normals[:,2]

        leakages_Ch = M_eval @ alpha
        leakages_new = leakages_Ch + leakages_Cp
        print(f'relative RMSE: {norm(leakages_new)/norm(leakages_Cp)}')
        return Ch, leakages_new, leakages_Cp
    else: 
        return alpha,alpha_int,Ch