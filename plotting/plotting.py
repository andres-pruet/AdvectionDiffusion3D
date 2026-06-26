#!/usr/bin/env python
# coding: utf-8

# In[15]:


'''
This code will display plots directly, and make .gif files based on the output of simulations.
The input to both these functions must be a timestamp string. The string is displayed when calling the .log() method of the simulate class, after calling the .run() method.
'''

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable


# In[ ]:

## parameters ##
timestamp = '2026-06-26--13_48_22'
solution_types = ['C']
interval = 110
plots = 0
gifs = 1
show_figs = False
show_obs = 1
slice = 'y'

# In[27]:


def make_plots(timestamp, solution_types, show_figs, slice, show_obs):
    timestep_vals = np.load(f'./data/plots/{timestamp}/timestep_vals.npy')
    # print(timestep_vals)
    xx = np.load(f'./data/plots/{timestamp}/xx.npy')
    yy = np.load(f'./data/plots/{timestamp}/yy.npy')
    zz = np.load(f'./data/plots/{timestamp}/zz.npy')
    obs = np.load(f'./data/plots/{timestamp}/obstacle.npy')
    params = np.load(f'./data/plots/{timestamp}/shape_params.npy')
    shape = params[-1]
    if obs:
        y_slice = np.argmin(np.abs(yy - params[1]))
        z_slice = np.argmin(np.abs(zz - params[2]))
    else:
        y_slice = int(len(yy)/2)
        z_slice = int(len(zz)/2)
    if not os.path.isdir(f'./data/figures/{timestamp}/'):
        os.mkdir(f'./data/figures/{timestamp}/')
    for soln_type in solution_types:
        fname = f'./data/plots/{timestamp}/{soln_type}_plots.npy'
        plots = np.load(fname)
        for i in np.arange(len(timestep_vals)):
            if slice=='y':
                plot = plots[y_slice,:,:,i]
            elif slice=='z':
                plot = plots[:,z_slice,:,i]
            t = round(timestep_vals[i],3)
            fig, ax = plt.subplots()
            if slice=='y':
                scat = ax.pcolor(xx,zz,plot)
                ax.set_xlabel('x')
                ax.set_ylabel('z')
                if obs and show_obs:
                    if shape==0:
                        ax.add_patch(plt.Circle((params[0], params[2]), params[3], fill=0, color='k'))
                    elif shape==1:
                        x0 = np.array([params[0],params[2]])
                        h = params[2]
                        r = params[3]
                        ax.add_patch(plt.Circle(x0, r, fill=0, color='k'))
                        ax.add_patch(plt.Rectangle([params[0]-r,0],2*r,h,fill=0, color='k'))
            elif slice=='z':
                scat = ax.pcolor(xx,yy,plot)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                if obs and show_obs:
                    if shape==0:
                        ax.add_patch(plt.Circle((params[0], params[1]), params[3], fill=0, color='k'))
                    elif shape==1:
                        x0 = np.array([params[0],params[1]])
                        h = params[2]
                        r = params[3]
                        ax.add_patch(plt.Circle(x0, r, fill=0, color='k')) # looks like a circle from top down
            cbar = plt.colorbar(scat)
            
            cbar.set_label('concentration')
            ax.set_title(f'{soln_type} t = {t}')
            plt.savefig(f'./data/figures/{timestamp}/{soln_type}_{i}.png', format='png')
            
            if show_figs:
                plt.show()
            plt.close()

def make_gifs(timestamp, solution_types, interval, slice, show_obs):
    timestep_vals = np.load(f'./data/plots/{timestamp}/timestep_vals.npy')
    xx = np.load(f'./data/plots/{timestamp}/xx.npy')
    yy = np.load(f'./data/plots/{timestamp}/yy.npy')
    zz = np.load(f'./data/plots/{timestamp}/zz.npy')
    obs = np.load(f'./data/plots/{timestamp}/obstacle.npy')
    params = np.load(f'./data/plots/{timestamp}/shape_params.npy')
    shape = params[-1]
    if obs:
        y_slice = np.argmin(np.abs(yy - params[1]))
        z_slice = np.argmin(np.abs(zz - params[2]))
    else:
        y_slice = int(len(yy)/2)
        z_slice = int(len(zz)/2)
    for soln_type in solution_types:
        fname = f'./data/plots/{timestamp}/{soln_type}_plots.npy'
        plots = np.load(fname)
        fig, axs = plt.subplots(1,2)
        fig.tight_layout(pad=5)
        xy = axs[0]
        xz = axs[1]
        plotxy = plots[:,z_slice,:,0]
        plotxz = plots[y_slice,:,:,0]
        scatxy = xy.pcolor(xx,yy,plotxy)
        scatxz = xz.pcolor(xx,zz,plotxz)
        xy.set_xlabel('x')
        xy.set_ylabel('y')
        xz.set_xlabel('x')
        xz.set_ylabel('z')
        xy.set_aspect('equal')
        xz.set_aspect('equal')
        xy.set_title(f'z = {zz[z_slice]}')
        xz.set_title(f'y = {yy[y_slice]}')
        if obs and show_obs:
            if shape==0:
                xy.add_patch(plt.Circle((params[0], params[1]), params[3], fill=0, color='k'))
                xz.add_patch(plt.Circle((params[0], params[2]), params[3], fill=0, color='k'))
            elif shape==1:
                x0 = np.array([params[0],params[2]])
                h = params[2]
                r = params[3]
                xz.add_patch(plt.Circle(x0, r, fill=0, color='k'))
                xz.add_patch(plt.Rectangle([params[0]-r,0],2*r,h,fill=0, color='k'))
                xy.add_patch(plt.Circle([params[0],params[1]], r, fill=0, color='k'))
        # if slice=='y':
        #     plot = plots[y_slice,:,:,0]
        #     scat = ax.pcolor(xx,zz,plot)
        #     ax.set_xlabel('x')
        #     ax.set_ylabel('z')
        #     if obs and show_obs:
        #         ax.add_patch(plt.Circle((params[0], params[2]), params[3], fill=0, color='k'))
        # elif slice=='z':
        #     plot = plots[:,z_slice,:,0]
        #     scat = ax.pcolor(xx,yy,plot)
        #     ax.set_xlabel('x')
        #     ax.set_ylabel('y')
        #     if obs and show_obs:
        #         ax.add_patch(plt.Circle((params[0], params[1]), params[3], fill=0, color='k'))
        # cbar = plt.colorbar(scatxz, shrink=.5)
        # cbar.set_label('concentration')
        divider = make_axes_locatable(xy)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(scatxy, cax=cax, orientation='vertical')
        
        divider = make_axes_locatable(xz)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(scatxz, cax=cax, orientation='vertical')
        # ax.add_patch(plt.Circle((params[0], params[1]), params[2], fill=1, color='k'))

        def update(frame):
            scatxy.set_array(plots[:,z_slice,:,frame])
            scatxz.set_array(plots[y_slice,:,:,frame])
            scatxy.set_clim(vmin=np.min(plots[:,z_slice,:,frame]), vmax=np.max(plots[:,z_slice,:,frame]))
            scatxz.set_clim(vmin=np.min(plots[y_slice,:,:,frame]), vmax=np.max(plots[y_slice,:,:,frame]))

            # if slice=='y':
            #     scat.set_array(plots[y_slice,:,:,frame])
            #     scat.set_clim(vmin=np.min(plots[y_slice,:,:,frame]), vmax=np.max(plots[y_slice,:,:,frame]))
            # elif slice == 'z':
            #     scat.set_array(plots[:,z_slice,:,frame])
                # scat.set_clim(vmin=np.min(plots[:,z_slice,:,frame]), vmax=np.max(plots[:,z_slice,:,frame]))
            # ax.set_title(f'{soln_type}: t = {round(timestep_vals[frame],3)}')
            return scatxy,scatxz

        ani = animation.FuncAnimation(fig=fig, func=update, frames=len(timestep_vals), interval=interval)
        ani.save(f'./data/gifs/{timestamp}_{soln_type}.gif')
        plt.close()


# In[28]:


if plots:
    make_plots(timestamp,solution_types, show_figs,slice, show_obs)
if gifs:
    make_gifs(timestamp,solution_types,interval, slice, show_obs)

