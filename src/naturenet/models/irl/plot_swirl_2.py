import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import sys
import argparse
import pickle
import os
import math
from sit_fuse.utils import read_yaml
import sparse
import copy
import cv2
import operator
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import colorcet as cc

import seaborn as sns

import jax.numpy as jnp

from naturenet.environment.grid_utils import Grid
from naturenet.environment.movement_preprocess_pipeline import gen_grid_point_paths

from naturenet.models.irl.swirl_training_top_level import learnt_LL2
 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
from matplotlib.animation import PillowWriter 
from matplotlib.collections import LineCollection
from matplotlib import cm
import matplotlib.patches as patches

import holoviews as hv
from holoviews import opts

hv.extension("bokeh") 


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The Axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)




# adapted from https://github.com/markusmeister/Rosenberg-2021-Repository
def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], loc=None, title=None,
         xlim=None, ylim=None, xscale='linear', yscale='linear',
         xticks=None, yticks=None, xhide=False, yhide=False, yrot=False, yzero=False, yflip=False, 
         fmts=['g-','m--','b-.','r:'], linewidth=2, markersize=5, fillstyle='full',
         markeredgewidth=1,
         grid=False, equal=False, figsize=(5,3), axes=None):
    """
    Plot data points.
    X: an array or list of arrays
    Y: an array or list of arrays
    If Y exists then those values are plotted vs the X values
    If Y doesn't exist the X values are plotted
    xlabel, ylabel: axis labels
    legend: list of labels for each Y series
    loc: location of the legend, like 'upper right'
    title: duh
    xlim, ylim: [low,high] list of limits for the 2 axes 
    xscale, yscale: 'linear' or 'log'
    xticks, yticks: list of locations for tick marks, or None for auto ticks
    yhide: hide the y axis?
    yrot: rotate the yaxis label to horizontal?
    yzero: zero line for the y-axis?
    fmts: a list of format strings to be applied to successive Y-series
    linewidth, markersize, fillstyle, markeredgewidth: see docs
    grid: draw a grid?
    equal: use equal aspect ratio, i.e. same scale per unit on x and y axis?
    figsize: (h,v) in inches
    axes: pre-existing axes where to draw the plot
    Returns: axes for the plot
    """
    
    if not axes: # start a new figure
        fig = plt.figure(figsize=figsize, dpi=400)
        axes = plt.gca()
    
    def has_one_axis(X): # Return True if X (ndarray or list) has 1 axis
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    # axes.cla() # clears these axes
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt, linewidth=linewidth, markersize=markersize,
            	fillstyle=fillstyle,markeredgewidth=markeredgewidth)
        else:
            axes.plot(y, fmt, linewidth=linewidth, markersize=markersize,
            	fillstyle=fillstyle,markeredgewidth=markeredgewidth)
    set_axes(axes, xlabel, ylabel, legend, loc, xlim, ylim, xscale, yscale, 
             xticks, yticks, xhide, yhide, yrot, yzero, yflip, grid, equal)
    if title:
        plt.title(title)
    plt.tight_layout()

    return axes # useful if we started a new figure

def set_axes(axes, xlabel, ylabel, legend, loc, xlim, ylim, xscale, yscale, 
    	xticks, yticks, xhide, yhide, yrot, yzero, yflip, grid, equal):
    """Set the axes for matplotlib."""
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    if xlim:
        axes.set_xlim(xlim)
    else:
        axes.set_xlim(auto=True)
    if ylim:
        axes.set_ylim(ylim)
    else:
        axes.set_ylim(auto=True)
    if grid:
        axes.grid()
    if equal:
        axes.set_aspect(aspect='equal')
    if ylabel:
        if yrot:
            axes.set_ylabel(ylabel, fontsize=12, rotation=0, labelpad=15)
        else:
            axes.set_ylabel(ylabel, fontsize=12)
    if xlabel:
        axes.set_xlabel(xlabel, fontsize=12)
    axes.get_yaxis().set_visible(not yhide)
    axes.get_xaxis().set_visible(not xhide)
    if yzero:
        axes.axhline(color='black', linewidth=0.5)
    if yflip:
        axes.invert_yaxis()
    axes.tick_params(axis = 'both', which = 'major', labelsize = 10)
    axes.tick_params(axis = 'both', which = 'minor', labelsize = 9)
    if xticks:
        axes.set_xticks(xticks,minor=False); # no minor ticks
    if yticks:
        axes.set_yticks(yticks,minor=False); # no minor ticks
    if legend:
        axes.legend(legend, loc=loc)
    plt.draw()


def plot_map_with_bounds(lon_min, lon_max, lat_min, lat_max):
    # 1. Define the Coordinate Reference System (CRS)
    # PlateCarree is the standard for latitude/longitude
    projection = ccrs.PlateCarree()

    # 2. Create a figure and axes with the specified projection
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=projection)

    # 3. Set the map extent (the lat/lon bounds)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=projection)

    # 4. Add geographical features
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE, linewidth=1.5)
    #ax.add_feature(cfeature.BORDERS, linestyle=':')
    #ax.add_feature(cfeature.LAKES, alpha=0.5)
    #ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.OCEAN)
    ax.set_zorder(1)
    # Optional: Add gridlines and labels
    #gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.7)
    #gl.top_labels = False
    #gl.right_labels = False
 
    return fig, ax

#states, grid location, grid, n_hidden, pyplot axes
def plot_trajs(zs, xy_list, grid, n_hidden, axs=None, figs=None):

    def record_segments_dict(jax_path_vmap, xy_list):
        n_trial, trial_length = jax_path_vmap.shape
        segments = {}
    
        for trial_idx in range(n_trial):
            trial_path = jax_path_vmap[trial_idx]
            trial_xys = xy_list[trial_idx].T
            
            start_idx = 0
            
            for i in range(1, trial_length):
                # If the value changes, record the current segment for the previous value
                if trial_path[i] != trial_path[start_idx] or i == trial_length - 1:
                    value = int(trial_path[start_idx])
                    #print(value)
                    if value not in segments:
                        segments[value] = []
                    segments[value].append(trial_xys[start_idx:i])  # Record the segment timestamps
                
                    start_idx = i  # Reset the start index for the new segment
        return segments
    xy_segments = record_segments_dict(zs, xy_list)


    if axs is None:
        figs, axs = plt.subplots(1, 1, figsize=(18,6), dpi=800)

    def plot_single_map(ax, fig, curr_xy_segments, note="", min_length=1):
        segs_list = []
        t_list = []

        #for g1 in range(grid.lat_tiles+1):
        #    for g2 in range(grid.lon_tiles+1):
        #        rect = patches.Rectangle((g1-0.5, g2-0.5), 1, 1, linewidth=1, edgecolor='lightgray', facecolor='lightgray')
        #        ax.add_patch(rect)
        #        #print("HERE", g1, g2)

        # Loop over all trajectories and collect segments and time arrays
        for xy in curr_xy_segments:
            #if xy.shape[0] < min_length:
            #    continue
            xs = []
            ys = []
            for k in range(len(xy)):
                xs.append(xy[k]["x"])
                ys.append(xy[k]["y"])
            #x = -0.5 + 15 * xy[:, 0]
            #y = -0.5 + 15 * xy[:, 1]
            
            xs = np.array(xs)
            ys = np.array(ys)
            t = np.linspace(0, 1, xs.shape[0])  # Time variable from 0 to 1
            

            print(xs.shape, ys.shape, t.shape, "WHAT?")
            # Set up a list of (x, y) points
            points = np.array([xs, ys]).transpose().reshape(-1, 1, 2)
            print(points.shape, "WHAT2")

            # Set up a list of segments
            segs = np.concatenate([points[:-1], points[1:]], axis=1)
            print(segs.shape, "WHAT3")
 
            if segs.shape[0] > 0 and t.shape[0] > 1:
                # Collect segments and corresponding time arrays
                segs_list.append(segs)
                t_list.append(t[:-1])  # t[:-1] since segments are between points

        # Concatenate all segments and time arrays
        all_segs = np.concatenate(segs_list)
        all_t = np.concatenate(t_list)


        print("HERE SEGS LENGTH", all_segs.shape, all_t.shape)
        # Create a single LineCollection with all segments
        lc = LineCollection(all_segs, cmap=plt.get_cmap('viridis'), linewidths=2)
        lc.set_array(all_t)  # Color the segments by the time parameter
        lc.set_zorder(10)

        # Add the LineCollection to the axes
        lines = ax.add_collection(lc)

        # # Add the color bar
        cax = fig.add_axes([1.05, 0.05, 0.05, 0.9])
        cbar = fig.colorbar(lines, cax=cax)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Start', 'End'])
        cbar.ax.tick_params(labelsize=18)
        ax.set_title(note, fontsize=24)
        return lines, ax, fig
    
    lines_list = []
    for i in range(n_hidden):
        note = "latent_state_" + str(i)

        fig = None
        ax = None

        if i in xy_segments:

            fig, ax = plt.subplots(1, 1, figsize=(18,6), dpi=800)
            lines, ax, fig = plot_single_map(ax, fig, xy_segments[i], note=note) #axs[i], figs[i], xy_segments[i], note=note)
            lines_list.append(lines)

        else:

            lines_list.append(None)

        if lines_list[i] is None:

            continue

        #fig = figs[i]
        print(os.path.join("trajs_latent_" + str(i) + ".png"))
        fig.savefig(os.path.join("trajs_latent_" + str(i) + ".png"), dpi=400, bbox_inches='tight')
        plt.clf()
        plt.cla()
        plt.close()

    return figs, axs, lines_list 




"""
color_options = [
    (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0),
    (0.5490196078431373, 0.33725490196078434, 0.29411764705882354, 1.0),
    (0.09019607843137255, 0.7450980392156863, 0.8117647058823529, 1.0)
]
"""

#def PlotMazeWall(m_wa,axes=None,figsize=4):
#    '''
#    Plots the walls of the maze defined in m.
#    axes: provide this to add to an existing plot
#    figsize: in inches (only if axes=None)
#    '''
#    if axes:
#        plot(m_wa[:,0],m_wa[:,1],fmts=['k-'],equal=True,linewidth=2,yflip=True,
#             xhide=True,yhide=True,axes=axes) # this way we can add to an existing graph
#    else:
#        axes = plot(m_wa[:,0],m_wa[:,1],fmts=['k-'],equal=True,linewidth=2,yflip=True,
#                  figsize=(figsize,figsize),xhide=True,yhide=True)
#    return axes

#import matplotlib.colors as mcolors

#PlotMazeFunction(converted_map, title_list[i], m_wa, m_ru, m_xc, m_yc, numcol='blue', figsize=6, selected_color=color_options[i], axes=axe
#s[i])


def plot_density(learnt_zs, env_maps, paths, f, n_hidden, out_dir, color_list):

    env_states = []
    heat_states = []
    dist_states = []

    max_dist = -1
    for i in range(n_hidden):
        env_states.append([])
        heat_states.append([])
        dist_states.append([])

        f[i] = normalize(f[i])

    f = np.round(f, decimals=2)
    for i in range(learnt_zs.shape[0]):
        zs = learnt_zs[i]
        path = paths[i]
        env_map = env_maps[i]
        for k in range(learnt_zs.shape[1]):
            if  path[k]["y"] < 0 or path[k]["x"] < 0:
                continue
            env_val = int(env_map[k,path[k]["y"],path[k]["x"]])
            if k == 0:
                dist = 0
            else:
                dist = math.sqrt((path[k-1]["y"] - path[k]["y"])**2 + (path[k-1]["x"] - path[k]["x"])**2)
            max_dist = max(max_dist, dist)
            dist = np.round(dist * (0.0009*111.11), decimals=2) #degrees to km at the equator
            dist_states[zs[k]].append(dist)  
            print(dist, zs[k]) 
            env_states[zs[k]].append(env_val)
            print(f[zs[k], env_val], zs[k], env_val)
            heat_states[zs[k]].append(f[zs[k], env_val])


    print(dist_states)
    print(heat_states)
    ax = plt.gca()
    latent = 0
    for i in range(n_hidden):
        latent = latent + 1
        if len(env_states[i]) == 0:
            continue
        sns.kdeplot(env_states[i], ax=ax, label="Latent State " + str(latent), color=color_list[i], alpha=0.5, fill=True, linewidth=0, clip=(0,315))
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(out_dir, "env_kdeplots.png"), bbox_inches='tight')
 
    plt.clf()
    #plt.clear()

    latent = 0
    ax = plt.gca()
    for i in range(n_hidden):
        latent = latent + 1
        print(len(env_states[i]), len(heat_states[i]), i)
        if len(env_states[i]) == 0:
            continue
        sns.kdeplot(heat_states[i], ax=ax, label="Latent State " + str(latent), color=color_list[i], alpha=0.5, fill=True, linewidth=0, clip=(0,1))
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(out_dir, "env_heat_kdeplots.png"), bbox_inches='tight')
    plt.clf()

    latent = 0
    ax = plt.gca()
    for i in range(n_hidden):
        latent = latent + 1
        print(len(env_states[i]), len(dist_states[i]), i)
        if len(env_states[i]) == 0:
            continue
        sns.kdeplot(dist_states[i], ax=ax, label="Latent State " + str(latent), color=color_list[i], alpha=0.5, fill=True, linewidth=0, clip=(0, max_dist + 10))
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(out_dir, "dist_kdeplots.png"), bbox_inches='tight')
    plt.clf()


def plot_reward_heatmaps(f, out_dir):

    f = normalize(f) 
    for j in range(f.shape[0]):
        f_tmp = np.squeeze(f[j,:])
        new_arr_size = math.ceil(math.sqrt(f_tmp.shape[0]))
        new_arr = np.zeros((new_arr_size**2))
        new_arr[:f_tmp.shape[0]] = f_tmp

        im = plt.imshow(new_arr.reshape(new_arr_size, new_arr_size), cmap="jet", interpolation='none', vmin=0, vmax=1)
        plt.colorbar(im, location='bottom', pad=0.05, ticks=[np.min(new_arr), np.max(new_arr)],
                 format='${x:.1f}$')
        plt.tight_layout(h_pad=0.1)
        plt.savefig(os.path.join(out_dir, "reward_heatmap_" + str(j) + ".png"), bbox_inches='tight')
        plt.clf()


def gen_cmap():
 
    palette_100 = sns.color_palette(cc.glasbey, n_colors=120)
    cmap_distinct = ListedColormap(palette_100)
    return cmap_distinct 


def map_reward_heat_to_env(env_map, f):

    f = np.squeeze(f)
    heat_map = np.ones(env_map.shape) - 1.0
    for i in range(f.shape[0]):
        inds = np.where(env_map == i)
        if len(inds[0]) < 1:
            continue
        heat_map[inds] = f[i]
    heat_map[np.where(env_map <= 0)] = np.nan

    return heat_map

def get_path_lims(path, env_map):

    xmin = env_map.shape[1]
    xmax = 0
    ymin = env_map.shape[0]
    ymax = 0

    for i in range(len(path)):
        xmin = min(xmin, path[i]["x"])
        ymin = min(ymin, path[i]["y"])
        xmax = max(xmax, path[i]["x"])
        ymax = max(ymax, path[i]["y"])
    if xmin > 10:
        xmin = xmin - 10
    else:
        xmin = 0

    if ymin > 10:
        ymin = ymin - 10
    else:
        ymin = 0

    if ymax < env_map.shape[0]-10:
        ymax = ymax + 10
    else:
        ymax = env_map.shape[0]

    if xmax < env_map.shape[1] - 10:
        xmax = xmax + 10
    else:
        xmax = env_map.shape[1]

    return xmin, xmax, ymin, ymax


def plot_env_maps(env_map, f, uid, out_dir):

    cmap = gen_cmap()
    hmap = hv.HoloMap(
        {
            t: hv.Image(env_map[t]).opts(
                cmap=cmap,
                aspect="equal",
                frame_width=400,
                colorbar=True,
            )
            for t in range(len(env_map))
        },
        kdims=["t"],
    )
    hv.save(hmap, os.path.join(out_dir, "env_map_backgrounds_slider" + "_" + uid + ".html"), backend="bokeh")

    f = normalize(f)
    hmap = hv.HoloMap(
        {   
            t: hv.Image(map_reward_heat_to_env(env_map[t], f)).opts(
                cmap="jet",
                aspect="equal",
                frame_width=400,
                colorbar=True,
            )
            for t in range(len(env_map))
        },
        kdims=["t"], 
    )
    hv.save(hmap, os.path.join(out_dir, "env_heatmap_backgrounds_slider" + "_" + uid + ".html"), backend="bokeh")

def plot_projection_examples(path, env_map, latents, uid, out_dir, color_list, lon_bounds, lat_bounds):

    xs = []
    ys = []


    print(lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1], len(env_map), len(latents), len(path))
    xmin, xmax, ymin, ymax = get_path_lims(path, env_map[0])

    fig, ax = plt.subplots() 

    tmp = copy.deepcopy(env_map[0]).astype(np.float32)
    if not np.isnan(tmp).any():
        tmp[np.where(tmp <= 0)] = np.nan

    cmap = gen_cmap()
    im = ax.matshow(tmp, cmap=cmap, interpolation='none',\
        aspect="equal", zorder=0)
    ##fig, ax = plot_map_with_bounds(lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1])

    num_frames = len(path)

    xs.append(path[0]["x"])
    ys.append(path[0]["y"])
    (line,) = ax.step(xs, ys, where="post", color="white", linewidth=3, zorder=1)

    latest_point = ax.scatter(
        xs, ys, s=150,           # size controls marker size
    color=color_list[latents[0]], edgecolor="white", zorder=2
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    circles = []
    x2s = []
    y2s = []
    #ci = []
    (line2,) = ax.step(xs, ys, where="post", color="palegreen", linewidth=3, zorder=2)
    #fill = ax.fill_between(x2s, y2s, y2s, color='mediumseagreen', alpha=.2)
    #fill = confidence_ellipse(x2s, y2s, ax, n_std=3.0, facecolor='none', )
    #circle = patches.Circle((x, y), r, facecolor='mediumseagreen', edgecolor='green', linewidth=2)
    #ax.add_patch(circle)
 
    def init():
        tmp = copy.deepcopy(env_map[0]).astype(np.float32)
        if not np.isnan(tmp).any():
            tmp[np.where(tmp <= 0)] = np.nan
        im.set_data(tmp) #[ymin:ymax+1, xmin:xmax+1])

        line.set_data(xs, ys)
        latest_point.set_offsets([[xs[-1], ys[-1]]])
        latest_point.set_facecolor(color_list[latents[0]])
        line2.set_data(x2s, y2s)
        #fill = ax.fill_between(x2s, y2s, y2s, color='mediumseagreen', alpha=.2)
        #circle = patches.Circle((x, y), r, facecolor='mediumseagreen', edgecolor='green', linewidth=2)
        #ax.add_patch(circle)

        return im, line, line2 #, latest_point   # return both artists for blitting

    def update(frame):
        # Update background image
        if frame > latents.shape[0]-1:
            latent = latents[-1]
        else:  
            latent = latents[frame]

        tmp = copy.deepcopy(env_map[frame]).astype(np.float32)
        if not np.isnan(tmp).any():
            tmp[np.where(tmp <= 0)] = np.nan
        im.set_data(tmp)


        if frame < 11:
            xs.append(path[frame]["x"])
            ys.append(path[frame]["y"])
            line.set_data(xs, ys)
            latest_point.set_offsets([[xs[-1], ys[-1]]])
            latest_point.set_facecolor(color_list[latent])
            #fill = ax.fill_between(x2s, y2s, y2s, color='mediumseagreen', alpha=.2)
            line2.set_data(x2s, y2s)
        else:  
            x2s.append(path[frame]["x"])
            y2s.append(path[frame]["y"])
            #ci.append(2*(frame-51)/2)
            r = 2*(frame-11)/8 #/ 2
            circle = patches.Circle((path[frame]["x"], path[frame]["y"]), r, facecolor='honeydew', edgecolor='palegreen', linewidth=2)
            ax.add_patch(circle)
            circles.append(circle)
            #fill = ax.fill_between(x2s, list(map(operator.sub, y2s, ci)), list(map(operator.add, y2s, ci)), color='mediumseagreen', alpha=.2)
            line.set_data(xs, ys)
            line2.set_data(x2s, y2s)
            latest_point.set_offsets([[x2s[-1], y2s[-1]]])
            latest_point.set_facecolor(color_list[latents[latent]])
        ret = copy.deepcopy(circles)
        ret.extend([im, line, line2])
        return ret #circles, im, line, line2 #, latest_point

    ani = animation.FuncAnimation(
        fig,   
        update,
        init_func=init,
        frames=num_frames,
        interval=300,
        blit=True
    )

    gif_writer = PillowWriter(fps=3)
    ani.save(os.path.join(out_dir, "false_proj_" + uid + ".gif"), writer=gif_writer, dpi=800)
    print(os.path.join(out_dir, "false_proj_" + uid + ".gif"))

    plt.close(fig)
    plt.clf()



def plot_env_heat_map_traj_animations(path, env_map, f, latents, uid, out_dir, color_list):

    print(np.nanmin(f), np.nanmax(f), np.nanmean(f))
    
    f = normalize(f)
    print(np.nanmin(f), np.nanmax(f), np.nanmean(f))
    cmap = "jet"

    xs = []
    ys = []

    
    xmin, xmax, ymin, ymax = get_path_lims(path, env_map[0])

    print(path.shape, env_map.shape, f.shape, latents.shape, "LATENTS")
    #time step
    fig, ax = plt.subplots()

    heat_map = map_reward_heat_to_env(env_map[0], f[latents[0]])

    #print(heat_map.shape, heat_map.min(), heat_map.max())
    num_frames = len(path)
    #print(xmin, xmax, ymin, ymax, heat_map[ymin:ymax+1, xmin:xmax+1].min(), heat_map[ymin:ymax+1, xmin:xmax+1].max(), heat_map[ymin:ymax+1, xmin:xmax+1].mean())
    #im = ax.matshow(heat_map[ymin:ymax+1, xmin:xmax+1], cmap=cmap, extent=[xmin, xmax, ymin, ymax], interpolation='none',\
    #    aspect="equal", zorder=0, vmin=0, vmax=1)
    im = ax.matshow(heat_map, cmap=cmap, interpolation='none',\
        aspect="equal", zorder=0, vmin=0, vmax=1)
    # Initial step line 
    xs.append(path[0]["x"])
    ys.append(path[0]["y"])
    (line,) = ax.step(xs, ys, where="post", color="black", linewidth=3, zorder=1)

    latest_point = ax.scatter(
        xs, ys, s=150,           # size controls marker size
    color=color_list[latents[0]], edgecolor="black", zorder=2
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
 
    def init():
        line.set_data(xs, ys)
        heat_map = map_reward_heat_to_env(env_map[0], f[latents[0]])
        im.set_data(heat_map) #[ymin:ymax+1, xmin:xmax+1])
        latest_point.set_offsets([[xs[-1], ys[-1]]])
        latest_point.set_facecolor(color_list[latents[0]])
        return im, line   # return both artists for blitting

    def update(frame):
        # Update background image
        if frame > latents.shape[0]-1:
            latent = latents[-1]
        else:
            latent = latents[frame]
 
        heat_map = map_reward_heat_to_env(env_map[frame], f[latent])
        print(frame, heat_map.shape, np.nanmin(heat_map), np.nanmean(heat_map), np.nanmax(heat_map))
        #print(xmin, xmax, ymin, ymax, heat_map[ymin:ymax+1, xmin:xmax+1].min(), heat_map[ymin:ymax+1, xmin:xmax+1].max(), heat_map[ymin:ymax+1, xmin:xmax+1].mean())
        im.set_data(heat_map) #[ymin:ymax+1, xmin:xmax+1])  # update image without re-calling imshow 
 
        # Update step line (grow over time)
        xs.append(path[frame]["x"])
        ys.append(path[frame]["y"])
        line.set_data(xs, ys)
        latest_point.set_offsets([[xs[-1], ys[-1]]])
        latest_point.set_facecolor(color_list[latent])
        return im, line

    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=num_frames,
        interval=300,
        blit=True
    )

    gif_writer = PillowWriter(fps=3)
    ani.save(os.path.join(out_dir, "env_heatmap_" + uid + ".gif"), writer=gif_writer, dpi=800)
    print(os.path.join(out_dir, "env_heatmap_" + uid + ".gif"))

    plt.close(fig)
    plt.clf()


def plot_env_prev_maps_traj_animations(path, env_map, prev_env_map, uid, out_dir):

    cmap = "jet"

    print(path.shape, env_map.shape)

    xmin, xmax, ymin, ymax = get_path_lims(path, env_map[0])
    num_frames = len(path)
    print(len(prev_env_map), len(prev_env_map[0]))
    for i in range(len(prev_env_map[0])):
        prev_env_map_tmp = prev_env_map[0][i]
        print(prev_env_map_tmp.shape)
        prev_env_map_tmp = np.squeeze(np.mean(prev_env_map_tmp, axis=(-1,-2)))
        for j in range(prev_env_map_tmp.shape[2]): 
            xs = []
            ys = []

            fig, ax = plt.subplots()  

            print("RESAMPLING", j, (prev_env_map_tmp.shape), prev_env_map_tmp[:,:,j].min(),\
                prev_env_map_tmp[:,:,j].max(), prev_env_map_tmp[:,:,j].mean())
            #prev_env_map_tmp[:,:,j] = normalize(prev_env_map_tmp[:,:,j])
            prev_env = cv2.resize(np.squeeze(prev_env_map_tmp[:,:,j]), (env_map[0].shape[1], env_map[0].shape[0]), interpolation=cv2.INTER_CUBIC)
            print(prev_env.min(), prev_env.max(), prev_env.mean(), prev_env.shape, env_map[0].shape, xmin, xmax, ymin, ymax)
            print(prev_env[ymin:ymax+1, xmin:xmax+1].min(), prev_env[ymin:ymax+1, xmin:xmax+1].max(), prev_env[ymin:ymax+1, xmin:xmax+1].mean())

            im = ax.matshow(prev_env[ymin:ymax+1, xmin:xmax+1], cmap=cmap, extent=[xmin, xmax, ymin, ymax], interpolation='none',\
                aspect="equal", zorder=0)

            # Initial step line 
            xs.append(path[0]["x"])
            ys.append(path[0]["y"])
            (line,) = ax.step(xs, ys, where="post", color="red", linewidth=2, zorder=1)

            latest_point = ax.scatter(
                xs, ys, s=150,           # size controls marker size
            color=color_list[latents[0]], edgecolor="black", zorder=2
            )

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

            def init():
                line.set_data(xs, ys)
                prev_env = cv2.resize(np.squeeze(prev_env_map_tmp[:,:,j]), (env_map[0].shape[1], env_map[0].shape[0]),\
                    interpolation=cv2.INTER_CUBIC)
                print(prev_env[ymin:ymax+1, xmin:xmax+1].min(), prev_env[ymin:ymax+1, xmin:xmax+1].max(), prev_env[ymin:ymax+1, xmin:xmax+1].mean())
                im.set_data(prev_env[ymin:ymax+1, xmin:xmax+1])
                latest_point.set_offsets([[xs[-1], ys[-1]]])
                return im, line   # return both artists for blitting
 
            def update(frame):
                # Update background image
                prev_env_map_tmp = prev_env_map[frame][i]
                prev_env_map_tmp = np.squeeze(np.mean(prev_env_map_tmp, axis=(-1,-2)))
   
                prev_env = cv2.resize(np.squeeze(prev_env_map_tmp[:,:,j]), (env_map[0].shape[1], env_map[0].shape[0]),\
                    interpolation=cv2.INTER_CUBIC)
                print(frame, i, prev_env[ymin:ymax+1, xmin:xmax+1].min(), prev_env[ymin:ymax+1, xmin:xmax+1].max(), prev_env[ymin:ymax+1, xmin:xmax+1].mean())
                im.set_data(prev_env[ymin:ymax+1, xmin:xmax+1])

                # Update step line (grow over time)
                xs.append(path[frame]["x"])
                ys.append(path[frame]["y"])
                line.set_data(xs, ys)
                latest_point.set_offsets([[xs[-1], ys[-1]]])
                latest_point.set_facecolor(color_list[latents[0]])	
                return im, line

            ani = animation.FuncAnimation(
                fig,
                update,
                init_func=init,
                frames=num_frames,
                interval=300,
                blit=True
            )
        
            gif_writer = PillowWriter(fps=3)
            ani.save(os.path.join(out_dir, "env_prev_map_" + uid + "_feat_map_" + str(i) + "_channel_" + str(j) + ".gif"),\
                 writer=gif_writer, dpi=800)

            plt.clf()
            plt.close(fig)



def plot_env_map_traj_animations(path, env_map, latents, uid, out_dir, color_list):
    
    cmap = gen_cmap()

    xs = []
    ys = []
     
    print(path.shape, env_map.shape)
    fig, ax = plt.subplots()

    xmin, xmax, ymin, ymax = get_path_lims(path, env_map[0])
    print(env_map[0].shape, env_map[0].min(), env_map[0].max(), env_map[0].mean())
    num_frames = len(path)
    #print(env_map[0][ymin:ymax+1, xmin:xmax+1].min(), env_map[0][ymin:ymax+1, xmin:xmax+1].max(), env_map[0][ymin:ymax+1, xmin:xmax+1].mean())
    #im = ax.matshow(env_map[0][ymin:ymax+1, xmin:xmax+1], cmap=cmap, extent=[xmin, xmax, ymin, ymax], interpolation='none',\
    #    aspect="equal", zorder=0)
    tmp = copy.deepcopy(env_map[0]).astype(np.float32)
    if not np.isnan(tmp).any():
        tmp[np.where(tmp <= 0)] = np.nan
    im = ax.matshow(tmp, cmap=cmap, interpolation='none',\
        aspect="equal", zorder=0)
    # Initial step line 
    xs.append(path[0]["x"])
    ys.append(path[0]["y"])
    (line,) = ax.step(xs, ys, where="post", color="black", linewidth=2, zorder=1)

    latest_point = ax.scatter(
        xs, ys, s=150,           # size controls marker size
    color=color_list[latents[0]], edgecolor="black", zorder=2
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
 
    def init():
        line.set_data(xs, ys)
        tmp = copy.deepcopy(env_map[0]).astype(np.float32)
        if not np.isnan(tmp).any():
            tmp[np.where(tmp <= 0)] = np.nan
        im.set_data(tmp) #[ymin:ymax+1, xmin:xmax+1])
        latest_point.set_offsets([[xs[-1], ys[-1]]])
        latest_point.set_facecolor(color_list[latents[0]])
        return im, line   # return both artists for blitting
 
    def update(frame):
        if frame > latents.shape[0]-1:
            latent = latents[-1]
        else:
            latent = latents[frame]


        # Update background image
        print(frame, env_map[frame].shape, np.nanmin(env_map[frame]), np.nanmean(env_map[frame]), np.nanmax(env_map[frame]))
        #print(env_map[frame][ymin:ymax+1, xmin:xmax+1].min(), env_map[frame][ymin:ymax+1, xmin:xmax+1].max(), env_map[frame][ymin:ymax+1, xmin:xmax+1].mean())
        #im.set_data(env_map[frame][ymin:ymax+1, xmin:xmax+1])  # update image without re-calling imshow 
        tmp = copy.deepcopy(env_map[frame]).astype(np.float32)
        if not np.isnan(tmp).any():
            tmp[np.where(tmp <= 0)] = np.nan
        im.set_data(tmp)

        # Update step line (grow over time)
        xs.append(path[frame]["x"])
        ys.append(path[frame]["y"])
        line.set_data(xs, ys)
        latest_point.set_offsets([[xs[-1], ys[-1]]])
        latest_point.set_facecolor(color_list[latent])
        return im, line
 

    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=num_frames,
        interval=300,
        blit=True
    )
 
    gif_writer = PillowWriter(fps=3)
    ani.save(os.path.join(out_dir, "env_map_" + uid + ".gif"), writer=gif_writer, dpi=800) 
    print(os.path.join(out_dir, "env_map_" + uid + ".gif"))
 
    plt.close(fig) 
    plt.clf()


def plot_rewards(f, paths, numcol='cyan', selected_color=None, figsize=6, ax = None, state_name = ""):
    '''
    Plot the maze defined in m with a function f overlaid in color
    f[]: array of something as a function of place in the maze, e.g. cell occupancy
        If f is None then the shading is omitted
    grid: The grid representation of env
    numcol: color for the numbers. If numcol is None the numbers are omitted
    figsize: in inches
    selected_color: a tuple specifying the RGBA color to be used for the colormap
    Returns: the axes of the plot.
    '''
    f = normalize(f)
    if selected_color is None:
        selected_color = "red"

    #col = np.array([[0, 1, 1, 1], [1, selected_color[0], selected_color[1], selected_color[2]]])
    norm = plt.Normalize(np.min(f), np.max(f))
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', [(1, 1, 1, 1), selected_color])
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])

    for j, r in enumerate(paths):
        print(f.shape, len(paths), j, r)
        x = r[-1]["x"]
        y = r[-1]["x"]
        if f is not None:
            ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1.0, 1.0, lw=0,
                                            fc=custom_cmap(norm(f[j])), ec='gray'))
                
        #plt.colorbar(sm,ticks=[0, 1], fraction=0.046, pad=0.04)
        ax.set_title(state_name, fontsize=20)

        # plt.axis('off')

def normalize(vals):
    """
    normalize to (0, max_val)
    input:
      vals: 1d array
    """
    min_val = np.nanmin(vals)
    max_val = np.nanmax(vals)
    return (vals - min_val) / (max_val - min_val)

def gen_invalid_inds(prev_state_map, n_state, n_action, n_latent):
    invalid_indices = np.ones((n_latent, n_state), dtype=bool)

    for x in range(n_state):
        for prev_x_i in np.arange(n_latent):
            if prev_x_i < len(prev_state_map[x]):
                invalid_indices[prev_x_i, x] = False

    return invalid_indices 


 


def run_plots(yml_conf):

    plot_traj = yml_conf["plot_trajs"]
    plot_gifs = yml_conf["plot_gifs"]

    seed = yml_conf["seed"]

    lon_bounds = yml_conf["lon_bounds"]
    lat_bounds = yml_conf["lat_bounds"]

    n_hidden = yml_conf["n_hidden_init"]

    trans_prob_fpath = yml_conf["trans_probs"]
    out_dir = yml_conf["out_dir"]
    run_uid = yml_conf["run_uid"]

    trans_prob = sparse.load_npz(trans_prob_fpath)

    trans_prob = trans_prob.todense()

    paths_fpaths = yml_conf["paths"]
    paths = []
    max_len = -1

    key = "1F90648_(4111)"  #"2017CA-Bmu-00825" #TODO fix
    df_uid = yml_conf["df_run_uid"] + "_" + key
    with open(os.path.join(yml_conf["df_dir"], df_uid + "_grid.pkl"), "rb") as f:
        grid = pickle.load(f)
  
    #with open(os.path.join(yml_conf["prev_envs"][0]), "rb") as f:
    #    prev_env = pickle.load(f)
 
    env_map_fname = yml_conf["envs"]
    with open(env_map_fname, "rb") as f:
        envs_init = pickle.load(f)

    if not os.path.exists(paths_fpaths[5]):#TODO
       gen_grid_point_paths(yml_conf) 
    
    uids_init = yml_conf["uids"]

    uids = []
    envs = []
    prev_envs = []
    print(len(paths_fpaths))
    for i in range(len(paths_fpaths)):
        paths_tmp = np.load(paths_fpaths[i], allow_pickle=True)
        envs_tmp = envs_init[uids_init[i]]

        #with open(os.path.join(yml_conf["prev_envs"][i]), "rb") as f:
        #    prev_env_tmp = pickle.load(f)

        #print(paths_tmp)
        for j in range(len(paths_tmp)):
            max_len = max(max_len, len(paths_tmp[j]))

        print(envs_tmp[0][0].shape, len(envs_tmp))
        envs.extend(envs_tmp)
        for e in range(len(envs_tmp)):
            uids.append(uids_init[i])
        paths.extend(paths_tmp)
        #prev_envs.extend(prev_env_tmp) 

    #for now, at least, all path lengths have to be the same for SWIRL :(
    paths_final = []
    envs_final = []
    uids_final = []
    #prev_envs_final = []
    for j in range(len(paths)):
        if len(paths[j]) == max_len:
            paths_final.append(paths[j])
            envs_final.append(envs[j])
            uids_final.append(uids[j])
            #prev_envs_final.append(prev_envs[j])
 
    #del prev_env
    #print(len(prev_envs_final), len(prev_envs_final[0]))
    #prev_envs = prev_envs_final
    paths = paths_final
    uids = uids_final
    envs = envs_final

    print(len(paths))

    paths = np.array(paths) #[:-1])
    envs = np.array(envs)
    uids = np.array(uids)


    #actions = np.load(actions_fpath, allow_pickle=True)
    #positions = np.load(positions_fpath, allow_pickle=True)

    n_states, n_actions, _ = trans_prob.shape

    prev_state_map_fname = os.path.join(out_dir, run_uid + "_prev_state_map.pkl")
    prev_state_map = np.load(prev_state_map_fname, allow_pickle=True)
    invalid_indices = gen_invalid_inds(prev_state_map, n_states, n_actions, n_hidden)

    ##maze_info = np.load(folder + '/maze_info.npz', allow_pickle=True) #Dont need this here, but useful reference for more complex env reps.
  
    temps = jnp.array([0.01] + [1] * (n_hidden - 1))

    # Load S-2 params
    print("Load params and set reward values")
    fname = run_uid + "_" + str(n_hidden) + '_' + str(seed) + "_naturenet_iter2.npz"
    fname = os.path.join(out_dir, fname)
    params2 = jnp.load(fname, allow_pickle=True)
    new_logpi02, new_log_Ps2, new_Rs2, new_reward2, LL_list2 = params2['new_logpi0'], params2['new_log_Ps'], params2['new_Rs'], params2['new_reward'], params2['LL_list']

    params3 = jnp.load(yml_conf["ll2_fname"], allow_pickle=True)
    jax_path_vmap=params3["jax_path_vmap"]
    

    #temps = jnp.array([0.01] + [1] * (n_hidden - 1))

    print("HERE REWARD", new_reward2.shape)
    reward2_filtered = np.copy(new_reward2[:, 0,:]).reshape((n_hidden, n_states, n_actions))
    print(invalid_indices.shape, reward2_filtered.shape)
 
    #reward2_filtered[invalid_indices,:] = np.nan
 
    color_list = ["red", "green", "blue", "brown", "cyan", "orange", "black", "magenta"]
    print(reward2_filtered.shape)
    converted_map = np.nanmean(reward2_filtered, axis=-1)
    print(converted_map.shape) 
    plot_reward_heatmaps(converted_map, yml_conf["out_dir"])
 
    learnt_zs = np.array(jax_path_vmap)

    print("HERE CONVERTED MAP", converted_map.shape)
    if plot_gifs:
        for j in range(len(paths)):
            converted_map = np.nanmean(reward2_filtered, axis=-1)
            plot_projection_examples(paths[j], envs[j], learnt_zs[j], uids[j] + "_path" + str(j), out_dir, color_list, lon_bounds, lat_bounds)
            #plot_env_heat_map_traj_animations(paths[j], envs[j], converted_map, learnt_zs[j], uids[j] + "_path" + str(j), out_dir, color_list)
            #plot_env_map_traj_animations(paths[j], envs[j], learnt_zs[j], uids[j] + "_path" + str(j), out_dir, color_list)
            


        """
        for i in range(n_hidden):
            title = "Action_" + str(i)
            color = color_list[i]
            converted_map = np.nanmean(reward2_filtered[i,:,:], axis=-1)    
            #TODO - read in env map and uid
            for j in range(len(paths)):
                #plot_env_maps(envs[j], converted_map, uids[j] + "_path" + str(j) + "_latent" + str(i), out_dir)
                #plot_env_prev_maps_traj_animations(paths[j], envs[j], prev_envs[j], uids[j] + "_path" + str(j) + "_latent" + str(i), out_dir)
                plot_env_heat_map_traj_animations(paths[j], envs[j], converted_map, uids[j] + "_path" + str(j) + "_latent" + str(i), out_dir)
                if i == 0:
                    plot_env_map_traj_animations(paths[j], envs[j], uids[j] + "_path" + str(j) + "_latent" + str(i), out_dir)
                    plot_env_maps(envs[j], converted_map, uids[j] + "_path" + str(j) + "_latent" + str(i), out_dir)
                    #plot_env_prev_maps_traj_animations(paths[j], envs[j], prev_envs[j], uids[j] + "_path" + str(j) + "_latent" + str(i), out_dir)
        """

    if plot_traj:

        print(converted_map.shape)
        plot_density(learnt_zs, envs, paths, converted_map, n_hidden, out_dir, color_list)

        figs = []
        axs = []
        for i in range(n_hidden):
            fig, ax = plot_map_with_bounds(lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1])
            figs.append(fig)
            axs.append(ax)
        figs, axs, lines_list = plot_trajs(learnt_zs, paths, grid, n_hidden, axs=axs, figs=figs)
  
 
        plt.clf()
        #plt.clear()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()

    #Translate config to dictionary 
    yml_conf = read_yaml(args.yaml)

    run_plots(yml_conf)



