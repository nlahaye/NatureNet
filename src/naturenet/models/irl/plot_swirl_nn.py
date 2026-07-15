
import operator

import pandas as pd
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

from datetime import datetime

import holoviews as hv
from holoviews import opts


hv.extension("bokeh") 

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
        axes.axhline(color='white', linewidth=0.5) #'black', linewidth=0.5)
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


def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    print(x_midpts, y_midpts, x.max(), y.max())

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def get_season(month):
 
    if month < 4:
        return 1
    elif month < 7:
        return 2
    elif month < 10:
        return 3
    else:
        return 4


def plot_time_series(learnt_zs, env_maps, paths, f, n_hidden, out_dir, color_list, static_data, uids):

    env_states = []
    heat_states = []
    dist_states = []
    turn_angles = []
    coasts = []
    ships = []
    baths = []

    bath = np.squeeze(static_data['Bathymetry']['scenes'][0])
    coast = np.squeeze(static_data['Coastal_Dist']['scenes'][0])
    ship = np.squeeze(static_data['Ship_Density']['scenes'][0])

    plt.imshow(bath)
    plt.savefig("BATH.png")
    plt.clf()

    plt.imshow(coast)
    plt.savefig("COASTS.png")
    plt.clf()

    plt.imshow(ship)
    plt.savefig("SHIP.png")
    plt.clf()


    max_dist = -1
    min_angles = 361
    max_angles = -1
    max_env = 0
    for i in range(n_hidden):
        f[i] = normalize(f[i])

    f = np.round(f, decimals=2)
    for i in range(learnt_zs.shape[0]):

        env_states.append([])
        heat_states.append([])
        dist_states.append([])
        turn_angles.append([])
        coasts.append([])
        baths.append([])
        ships.append([])
    

        print((learnt_zs.shape, len(paths), len(learnt_zs), len(env_maps), paths[i]), len(learnt_zs[i]), len(env_maps[i]), i)
        if len(paths[i]) < 25 or len(learnt_zs[i]) < 25 or len(env_maps[i]) < 25:
            continue
        zs = learnt_zs[i]
        path = paths[i]
        env_map = env_maps[i]
        for k in range(learnt_zs.shape[1]):
            print(k, k,path[k]["y"],path[k]["x"])
            if  path[k]["y"] < 0 or path[k]["x"] < 0:
                continue
            env_val = int(env_map[k,path[k]["y"],path[k]["x"]])
            if k == 0:
                dist = 0
            else:
                dist = math.sqrt((path[k-1]["y"] - path[k]["y"])**2 + (path[k-1]["x"] - path[k]["x"])**2)

            if k > 1:
                dy = path[k-1]["y"] - path[k]["y"]
                dx = path[k-1]["x"] - path[k]["x"]

                step_len = np.hypot(dx, dy)
 
                # Absolute movement angle in radians, measured from +x axis
                bearing = np.arctan2(dy, dx)
 
                # Relative turn angle, if previous location is known
                prev_dx = path[k-1]["x"] - path[k-2]["x"]
                prev_dy = path[k-1]["y"] - path[k-2]["y"]
 
                # Guard against zero-length previous step
                if prev_dx == 0 and prev_dy == 0:
                    turn_angle = 0.0
                else:
                    prev_bearing = np.arctan2(prev_dy, prev_dx)
                    turn_angle = wrap_to_pi(bearing - prev_bearing)
            else:
                turn_angle = 0.0

            turn_angles[i].append(turn_angle)
            baths[i].append(bath[path[k]["y"],path[k]["x"]])
            ships[i].append(ship[path[k]["y"],path[k]["x"]])
            coasts[i].append(coast[path[k]["y"],path[k]["x"]])

            max_dist = max(max_dist, dist)
            max_angles = max(max_angles, turn_angle)
            min_angles = min(0, 0, 0, 0, 0, 0, 0, 0, 0, min_angles, turn_angle)
            max_env = max(max_env, env_val)

            dist = np.round(dist * (0.009000090001*111.11), decimals=2) #degrees to km at the equator
            dist_states[i].append(dist)
            env_states[i].append(env_val)
            heat_states[i].append(f[zs[k], env_val])
        print(len(heat_states[i]), len(env_states[i]), len(dist_states[i]), len(baths[i]), len(ships[i]), len(coasts[i]), i, "HERE LENS") 



    pds = {}

    cmap = ListedColormap(color_list)

    print(len(env_states[1]), "HERE1")

    plt.clf()
    for ind in range(len(env_states)):
        x = list(range(0, len(env_states[ind])))
        y = [ind]*len(x)
        c = learnt_zs[ind]
        plt.scatter(x, y, c=c, cmap=cmap)
    plt.savefig(os.path.join(out_dir, "time_series_zs_plot.png"), bbox_inches='tight')


    print(len(env_states[1]), "HERE2")
 
    plt.clf()
    for ind in range(len(env_states)):
        x = list(range(0, len(env_states[ind])))
        print(len(env_states[ind]), ind, min(x), max(x))
        plt.plot(x, env_states[ind])
        plt.ylim(0, max_env)
    plt.savefig(os.path.join(out_dir, "env_map_time_series.png"), bbox_inches='tight')

    print(len(env_states[1]), "HERE3")


    df_dir = "/data/nlahaye/NatureNet/Blue_Whale_v1/"
    df_uid = "whale_v1"
    #df_dir = "/data/nlahaye/NatureNet/Hammerhead_Out_v1/"
    #df_uid = "hammerhead_v1"
 
    #out_dir = "/data/nlahaye/NatureNet/Blue_Whale_v1/"

    scenes_per_uid = {}
    movement_dfs = None
    with open(os.path.join(df_dir, df_uid + '_dfs.pkl'), "rb") as f:
        movement_dfs = pickle.load(f)

    time_dif = {}
    month = {}
    season = {}
 
    for uid in movement_dfs:
        current_time = None
     
        movement_dfs_uid = movement_dfs[uid]
        print("WTD UID", uid)
        min_dt = None
        for dind in range(len(movement_dfs_uid)):

            if len(movement_dfs_uid[dind]) < 99: #TODO generalize - removing end of paths that weren't included in training/eval
                continue
            print(uid, dind, len(movement_dfs_uid[dind]), len(movement_dfs_uid[dind]) < 90, "ERRORS HERE")
            movement_df = movement_dfs_uid[dind]

            dttm = movement_df.iloc[0]['TimeValue'] #["date"]
            #if len(dttm) == 10:
            #    dttm = dttm + " 00:00:00"

            if min_dt is None:
                min_dt = datetime.strptime(dttm, "%Y-%m-%dT%H:%M:%SZ") #"%Y-%m-%d %H:%M:%S") #['TimeValue'],"%Y-%m-%dT%H:%M:%SZ")
            else:
                min_dt = min(min_dt, datetime.strptime(dttm, "%Y-%m-%dT%H:%M:%SZ")) #"%Y-%m-%d %H:%M:%S")) #['TimeValue'],"%Y-%m-%dT%H:%M:%SZ"))  

        for dind in range(len(movement_dfs_uid)):

            if len(movement_dfs_uid[dind]) < 99: #TODO generalize - removing end of paths that weren't included in training/eval
                continue
            print(uid, dind, len(movement_dfs_uid[dind]), len(movement_dfs_uid[dind]) < 90, "ERRORS HERE2")
            movement_df = movement_dfs_uid[dind]
            if uid not in time_dif:
                print("HERE UID", uid)
                time_dif[uid] = []
                month[uid] = []
                season[uid] = []
            act_index = 0
            for index, row in movement_df.iterrows():

                #if len(row["date"]) == 10:
                #    row["date"] = row["date"] + " 00:00:00"

                if act_index == 0 or act_index == len(movement_df) - 1: #system currently cuts off first and last sample
                    act_index += 1
                    continue
                act_index += 1
                current_time = datetime.strptime(row['TimeValue'],"%Y-%m-%dT%H:%M:%SZ") #row["date"], "%Y-%m-%d %H:%M:%S") #['TimeValue'],"%Y-%m-%dT%H:%M:%SZ")
                print(current_time, min_dt, uid, "HERE TIME DIFF", len(movement_df), len(movement_dfs_uid), act_index, len(time_dif[uid]))
                time_df = (current_time - min_dt).total_seconds() / 60.0 / 60.0 
                time_dif[uid].append(time_df)
                month[uid].append(current_time.month)
                season[uid].append(get_season(current_time.month))

    plt.clf()
    uid = uids[0]
    current_uid = uids[0]

    uid_ind = 0
    ind = 0
    x = []
    y = []
    c = []


    while ind <= len(env_states):
  
        if ind < len(env_states):
            current_uid = uids[ind]
 
        print(uid, current_uid, ind, len(env_states), "UIDS_HERE") 
        if  current_uid == uid and ind < len(env_states):
                if uid_ind == 0:   
                    x = list(range(uid_ind, uid_ind + len(env_states[ind])))
                        
                    y = [ind]*len(env_states[ind])
                    c = learnt_zs[ind]
              
                    print(len(c), len(y), len(x), uid, uid_ind, "TEST1")
 
                    uid_ind = uid_ind + len(env_states[ind])
                else:
                    x.extend(list(range(uid_ind, uid_ind + len(env_states[ind]))))
                    y.extend([y[-1]]*len(env_states[ind]))
                    c = np.concatenate((c, learnt_zs[ind]), axis = 0)

                    print(len(c), len(y), len(x), uid, uid_ind, "TEST1_2")

                    uid_ind = uid_ind + len(env_states[ind])
        else:

            print(len(c), len(y), len(x), uid, uid_ind, "TEST1_3")

            pds[uid] = pd.DataFrame(index=x)
            print(pds.keys(), time_dif.keys())
            pds[uid]['Time'] = time_dif[uid]
            pds[uid]['month'] = month[uid]
            pds[uid]['seasaon'] = season[uid]
            pds[uid]['latent'] = c

            plt.scatter(x, y, c=c, cmap=cmap)
            plt.savefig(os.path.join(out_dir, "time_series_zs_plot_traj_color_" + uid + ".png"), bbox_inches='tight')

            plt.clf()
            plt.plot(x, c)
            plt.savefig(os.path.join(out_dir, "time_series_zs_plot_traj_" + uid + ".png"), bbox_inches='tight')

            plt.plot(x, time_dif[uid])
            plt.savefig(os.path.join(out_dir, "time_series_time_diff_traj_" + uid + ".png"), bbox_inches='tight')

            plt.clf()
            plt.plot(x, month[uid])
            plt.savefig(os.path.join(out_dir, "time_series_month_traj_" + uid + ".png"), bbox_inches='tight')

            plt.clf()
            plt.plot(x, season[uid])
            plt.savefig(os.path.join(out_dir, "time_series_season_traj_" + uid + ".png"), bbox_inches='tight')

            tmp_ind = ind
            if ind >= len(env_states):
                tmp_ind = ind-1

            plt.clf()
            uid_ind = 0
            x = list(range(uid_ind, uid_ind + len(env_states[tmp_ind])))
            y = [tmp_ind]*len(env_states[tmp_ind])
            c = learnt_zs[tmp_ind]
            uid_ind = uid_ind + len(env_states[tmp_ind])
        uid = current_uid
        ind = ind + 1


    print(len(env_states[1]), "HERE4")

    plt.clf()
    uid = uids[0]
    current_uid = uids[0]

    uid_ind = 0

    ind = 0
    x = []
    y = []
    while ind <= len(env_states):
        if ind < len(env_states):
            current_uid = uids[ind]
 
        if  current_uid == uid and ind < len(env_states):
                if uid_ind == 0:   
                    x = list(range(uid_ind, uid_ind + len(env_states[ind])))
                    y = copy.deepcopy(env_states[ind])


                    print(len(c), len(y), len(x), uid, uid_ind, "TEST2")

                    uid_ind = uid_ind + len(env_states[ind])
                else:
                    x.extend(list(range(uid_ind, uid_ind + len(env_states[ind]))))
                    y.extend(copy.deepcopy(env_states[ind]))

                    print(len(c), len(y), len(x), uid, uid_ind, "TEST2_1")

                    uid_ind = uid_ind + len(env_states[ind])
        else:
            plt.plot(x, y)


            print(len(c), len(y), len(x), uid, uid_ind, "TEST2_2")

            pds[uid]['env'] = y

            plt.ylim(0, max_env)
            plt.savefig(os.path.join(out_dir, "env_map_time_series_traj_" + uid + ".png"), bbox_inches='tight')
            plt.clf()

            tmp_ind = ind
            if ind >= len(env_states):
                tmp_ind = ind-1

            uid_ind = 0
            x = list(range(uid_ind, uid_ind + len(env_states[tmp_ind])))   
            y = copy.deepcopy(env_states[tmp_ind])
            uid_ind = uid_ind + len(env_states[tmp_ind])
        uid = current_uid
        ind = ind + 1

    print(len(env_states[1]))

    del ind


    plt.clf()
    for ind in range(len(env_states)):
        print(len(x), len(dist_states[ind]), ind)
        x = list(range(0, len(env_states[ind])))
        print(len(x), len(env_states[ind]), len(dist_states[ind]), ind)
        plt.plot(x, dist_states[ind])
        plt.ylim(0, max_dist)
    plt.savefig(os.path.join(out_dir, "dist_time_series.png"), bbox_inches='tight')


    plt.clf()
    uid = uids[0]
    current_uid = uids[0]

    uid_ind = 0
    ind = 0
    x = []
    y = []
    while ind <= len(env_states):

        if ind < len(env_states):
            current_uid = uids[ind]
   
        if  current_uid == uid and ind < len(env_states):
                if uid_ind == 0:
                    x = list(range(uid_ind, uid_ind + len(env_states[ind])))
                    y = copy.deepcopy(dist_states[ind])
                    uid_ind = uid_ind + len(env_states[ind])
                else:
                    x.extend(list(range(uid_ind, uid_ind + len(env_states[ind]))))
                    y.extend(dist_states[ind])
                    uid_ind = uid_ind + len(env_states[ind])
        else:
            plt.plot(x, y)
            pds[uid]['distance_traveled'] = y
            plt.ylim(0, max_dist)
            plt.savefig(os.path.join(out_dir, "dist_time_series_traj_" + uid + ".png"), bbox_inches='tight')
            plt.clf()

            tmp_ind = ind
            if ind >= len(env_states):
                tmp_ind = ind-1

            uid_ind = 0
            x = list(range(uid_ind, uid_ind + len(env_states[tmp_ind])))   
            y = copy.deepcopy(dist_states[tmp_ind])
            uid_ind = uid_ind + len(env_states[tmp_ind])
        uid = current_uid
        ind = ind + 1
 

    plt.clf()
    for ind in range(len(env_states)):
        x = list(range(0, len(env_states[ind])))
        plt.plot(x, heat_states[ind])
        plt.ylim(0, 1)
    plt.savefig(os.path.join(out_dir, "heat_time_series.png"), bbox_inches='tight')    

    plt.clf()
    uid = uids[0]
    current_uid = uids[0]

    uid_ind = 0

    ind = 0
    x = []
    y = []
    while ind <= len(env_states):
        if ind < len(env_states):
            current_uid = uids[ind] 
  
        if  current_uid == uid and ind < len(env_states):
                if uid_ind == 0:
                    x = list(range(uid_ind, uid_ind + len(env_states[ind])))
                    y = copy.deepcopy(heat_states[ind])
                    uid_ind = uid_ind + len(env_states[ind])
                else:
                    x.extend(list(range(uid_ind, uid_ind + len(env_states[ind]))))
                    y.extend(heat_states[ind])
                    uid_ind = uid_ind + len(env_states[ind])
        else:
            plt.plot(x, y)
            plt.ylim(0, 1)
            pds[uid]['reward_likelihood'] = y
            plt.savefig(os.path.join(out_dir, "heat_time_series_traj_" + uid + ".png"), bbox_inches='tight')
            plt.clf()

            tmp_ind = ind
            if ind >= len(env_states):
                tmp_ind = ind-1

            uid_ind = 0
            x = list(range(uid_ind, uid_ind + len(env_states[tmp_ind])))
            y = copy.deepcopy(heat_states[tmp_ind])
            uid_ind =  uid_ind + len(env_states[tmp_ind])

        uid = current_uid
        ind = ind + 1


    plt.clf()
    for ind in range(len(env_states)):
        x = list(range(0, len(env_states[ind])))
        plt.plot(x,  turn_angles[ind])
        plt.ylim(min_angles, max_angles)
    plt.savefig(os.path.join(out_dir, "turn_angles_time_series.png"), bbox_inches='tight')

    plt.clf()
    uid = uids[0]
    current_uid = uids[0]

    uid_ind = 0

    ind = 0
    x = []
    y = []
    while ind <= len(env_states):
 
        if ind < len(env_states):
            current_uid = uids[ind]
 
        if  current_uid == uid and ind < len(env_states):
                if uid_ind == 0:
                    x = list(range(uid_ind, uid_ind + len(env_states[ind])))
                    y = copy.deepcopy(turn_angles[ind])
                    uid_ind = uid_ind + len(env_states[ind])
                else:
                    x.extend(list(range(uid_ind, uid_ind + len(env_states[ind]))))
                    y.extend(turn_angles[ind])
                    uid_ind = uid_ind + len(env_states[ind])
        else:
            plt.plot(x, y)
            pds[uid]['turn_angle'] = y

            plt.ylim(min_angles, max_angles)
            plt.savefig(os.path.join(out_dir, "turn_angles_time_series_traj_" + uid + ".png"), bbox_inches='tight')
            plt.clf()

            tmp_ind = ind
            if ind >= len(env_states):
                tmp_ind = ind-1

            uid_ind = 0

            x = list(range(uid_ind, uid_ind + len(env_states[tmp_ind])))
            y = copy.deepcopy(turn_angles[tmp_ind])
            uid_ind = uid_ind + len(env_states[tmp_ind])
        uid = current_uid
        ind = ind + 1


    plt.clf()
    for ind in range(len(env_states)):
        x = list(range(0, len(env_states[ind])))
        plt.plot(x,  ships[ind])
        plt.ylim(0, 1)
    plt.savefig(os.path.join(out_dir, "ship_density_time_series.png"), bbox_inches='tight')

    plt.clf()
    uid = uids[0]
    current_uid = uids[0]

    uid_ind = 0

    ind = 0
    x = []
    y = []
    while ind <= len(env_states):
        if ind < len(env_states):
            current_uid = uids[ind]
     
        if  current_uid == uid and ind < len(env_states):
                if uid_ind == 0:
                    x = list(range(uid_ind, uid_ind + len(env_states[ind])))
                    y = copy.deepcopy(ships[ind])
                    uid_ind = uid_ind + len(env_states[ind])
                else:
                    x.extend(list(range(uid_ind, uid_ind + len(env_states[ind]))))
                    y.extend(ships[ind])
                    uid_ind = uid_ind + len(env_states[ind])
        else:
            plt.plot(x, y)
            plt.ylim(0, 1)
            pds[uid]['ship_density'] = y
            plt.savefig(os.path.join(out_dir, "ship_density_time_series_traj_" + uid + ".png"), bbox_inches='tight')
            plt.clf()

            tmp_ind = ind
            if ind >= len(env_states):
                tmp_ind = ind-1

            uid_ind = 0

            x = list(range(uid_ind, uid_ind + len(env_states[tmp_ind])))
            y = copy.deepcopy(ships[tmp_ind]) 
            uid_ind = uid_ind + len(env_states[tmp_ind])
        uid = current_uid
        ind = ind + 1


    plt.clf()
    for ind in range(len(env_states)):
        x = list(range(0, len(env_states[ind])))
        plt.plot(x,  baths[ind])
        plt.ylim(np.min(baths), np.max(baths))
    plt.savefig(os.path.join(out_dir, "bathymetry_time_series.png"), bbox_inches='tight')

    plt.clf()
    uid = uids[0]
    current_uid = uids[0]

    uid_ind = 0

    ind = 0
    x = []
    y = []
    while ind <= len(env_states):
 
        if ind < len(env_states):
            current_uid = uids[ind]

        if  current_uid == uid and ind < len(env_states):
                if uid_ind == 0:
                    x = list(range(uid_ind, uid_ind + len(env_states[ind])))
                    y = copy.deepcopy(baths[ind])
                    uid_ind = uid_ind + len(env_states[ind])
                else:
                    x.extend(list(range(uid_ind, uid_ind + len(env_states[ind]))))
                    y.extend(baths[ind])
                    uid_ind = uid_ind + len(env_states[ind])
        else:
            plt.plot(x, y)

            pds[uid]['bathymetry'] = y

            plt.ylim(np.min(baths), np.max(baths))
            plt.savefig(os.path.join(out_dir, "bathymetry_time_series_traj_" + uid + ".png"), bbox_inches='tight')
            plt.clf()

            tmp_ind = ind
            if ind >= len(env_states):
                tmp_ind = ind-1

            uid_ind = 0

            x = list(range(uid_ind, uid_ind + len(env_states[tmp_ind])))
            y = copy.deepcopy(baths[tmp_ind])
            uid_ind = uid_ind + len(env_states[tmp_ind])
        uid = current_uid
        ind = ind + 1


    plt.clf()
    for ind in range(len(env_states)):
        x = list(range(0, len(env_states[ind])))
        plt.plot(x,  coasts[ind])
        plt.ylim(np.min(coasts), np.max(coasts))
    plt.savefig(os.path.join(out_dir, "dist_to_coast_time_series.png"), bbox_inches='tight')

    plt.clf()
    uid = uids[0]
    current_uid = uids[0]

    uid_ind = 0

    ind = 0
    x = []
    y = []
    while ind <= len(env_states):
        if ind < len(env_states):
            current_uid = uids[ind]
  
        if  current_uid == uid and ind < len(env_states):
                if uid_ind == 0:
                    x = list(range(uid_ind, uid_ind + len(env_states[ind])))
                    y = copy.deepcopy(coasts[ind])
                    uid_ind = uid_ind + len(env_states[ind])
                else:
                    x.extend(list(range(uid_ind, uid_ind + len(env_states[ind]))))
                    y.extend(coasts[ind])
                    uid_ind = uid_ind + len(env_states[ind])
        else:
            plt.plot(x, y)
            plt.ylim(np.min(coasts), np.max(coasts))
            plt.savefig(os.path.join(out_dir, "dist_to_coast_time_series_traj_" + uid + ".png"), bbox_inches='tight')
            plt.clf()

            pds[uid]['distance_to_coast'] = y

            tmp_ind = ind
            if ind >= len(env_states):
                tmp_ind = ind-1

            uid_ind = 0

            x = list(range(uid_ind, uid_ind + len(env_states[tmp_ind])))
            y = copy.deepcopy(coasts[tmp_ind]) 
            uid_ind = uid_ind + len(env_states[tmp_ind])
        uid = current_uid
        ind = ind + 1


    for uid in pds:
        pds[uid].to_excel(os.path.join(out_dir, uid + "_time_series_pd.xlsx"), header=True, index=False)


def plot_density(learnt_zs, env_maps, paths, f, n_hidden, out_dir, color_list, static_data):

    env_states = []
    heat_states = []
    dist_states = []
    turn_angles = []
    baths = []
    ships = []
    coasts = []

    z_env_hist = {}

    max_dist = -1
    max_angles = -1
    min_angles = 361
    max_env = 0

    bath = np.squeeze(static_data['Bathymetry']['scenes'][0])
    coast = np.squeeze(static_data['Coastal_Dist']['scenes'][0])
    ship = np.squeeze(static_data['Ship_Density']['scenes'][0])
    print("HERE COAST", coast.min(), coast.max(), coast.mean(), coast.std())  
 

    for i in range(n_hidden):
        env_states.append([])
        heat_states.append([])
        dist_states.append([])
        turn_angles.append([])
        baths.append([])
        coasts.append([])
        ships.append([])

        f[i] = normalize(f[i])

        z_env_hist[i] = {}

    f = np.round(f, decimals=2)
    for i in range(learnt_zs.shape[0]):
        print( len(paths[i]), len(learnt_zs[i]), len(env_maps[i]), i)
        if len(paths[i]) < 97 or len(learnt_zs[i]) < 97 or len(env_maps[i]) < 97:
            continue
        zs = learnt_zs[i]
        path = paths[i]
        env_map = env_maps[i]
        for k in range(learnt_zs.shape[1]):
            print(k, k,path[k]["y"],path[k]["x"])
            if  path[k]["y"] < 0 or path[k]["x"] < 0:
                continue
            env_val = int(env_map[k,path[k]["y"],path[k]["x"]])
            print(zs[k], z_env_hist, env_val, "HERE", k,path[k]["y"], path[k]["x"], ship.shape, bath.shape, "HERE ISSUES")
            if env_val not in z_env_hist[zs[k]]:
                z_env_hist[zs[k]][env_val] = 1
            else:
                z_env_hist[zs[k]][env_val] = z_env_hist[zs[k]][env_val] + 1
            if k == 0:
                dist = 0
            else:
                dist = math.sqrt((path[k-1]["y"] - path[k]["y"])**2 + (path[k-1]["x"] - path[k]["x"])**2)
            max_dist = max(max_dist, dist)
            max_env = max(max_env, env_val)
            dist = np.round(dist * (0.009000090001*111.11), decimals=2) #degrees to km at the equator
            dist_states[zs[k]].append(dist)  
            print(dist, zs[k]) 
            env_states[zs[k]].append(env_val)
            print(f[zs[k], env_val], zs[k], env_val)
            heat_states[zs[k]].append(f[zs[k], env_val])


            print("HERE BATH ISSUE", zs[k], len(baths), path[k]["y"],path[k]["x"], bath.shape)
            baths[zs[k]].append(bath[path[k]["y"],path[k]["x"]])
            ships[zs[k]].append(ship[path[k]["y"],path[k]["x"]])
            coasts[zs[k]].append(coast[path[k]["y"],path[k]["x"]])

            if k > 1:
                dy = path[k-1]["y"] - path[k]["y"]
                dx = path[k-1]["x"] - path[k]["x"]

                step_len = np.hypot(dx, dy)

                # Absolute movement angle in radians, measured from +x axis
                bearing = np.arctan2(dy, dx)

                prev_dx = path[k-1]["x"] - path[k-2]["x"]
                prev_dy = path[k-1]["y"] - path[k-2]["y"]

                # Guard against zero-length previous step
                if prev_dx == 0 and prev_dy == 0:
                    turn_angle = 0.0
                else:
                    prev_bearing = np.arctan2(prev_dy, prev_dx)
                    turn_angle = wrap_to_pi(bearing - prev_bearing)
            else:
                turn_angle = 0.0

            max_angles = max(max_angles, turn_angle)
            min_angles = min(0, min_angles, turn_angle)
            turn_angles[zs[k]].append(turn_angle)  
            

    print(dist_states)
    print(heat_states)
    plt.rc('font', size=18) 
    ax = plt.gca()
    latent = 0
    for i in range(n_hidden):
        latent = latent + 1
        if len(env_states[i]) == 0:
            continue
        sns.kdeplot(env_states[i], ax=ax, label="Latent State " + str(latent), color=color_list[i], alpha=0.5, fill=True, linewidth=0, clip=(0,max_env))
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(out_dir, "env_kdeplots.png"), bbox_inches='tight')
 
    plt.clf()

    plt.boxplot(env_states, labels=list(range(1,len(env_states)+1)))
    plt.legend()
    plt.tight_layout(h_pad=0.1)
    plt.savefig(os.path.join(out_dir, "env_dist_bplot.png"), bbox_inches='tight')
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

    plt.boxplot(heat_states, labels=list(range(1,len(heat_states)+1)))
    plt.legend()
    plt.tight_layout(h_pad=0.1)
    plt.savefig(os.path.join(out_dir, "env_heat_dist_bplot.png"), bbox_inches='tight')
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

    plt.boxplot(dist_states, labels=list(range(1,len(dist_states)+1)))
    plt.legend()
    plt.tight_layout(h_pad=0.1)
    plt.savefig(os.path.join(out_dir, "dist_bplot.png"), bbox_inches='tight')
    plt.clf()

    latent = 0
    ax = plt.gca()
    for i in range(n_hidden):
        latent = latent + 1
        print(len(env_states[i]), len(turn_angles[i]), i)
        if len(env_states[i]) == 0:
            continue
        sns.kdeplot(turn_angles[i], ax=ax, label="Latent State " + str(latent), color=color_list[i], alpha=0.5, fill=True, linewidth=0, clip=(min_angles, max_angles))
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(out_dir, "angle_kdeplots.png"), bbox_inches='tight')
    plt.clf() 

    plt.boxplot(turn_angles, labels=list(range(1,len(turn_angles)+1)))
    plt.legend()
    plt.tight_layout(h_pad=0.1)
    plt.savefig(os.path.join(out_dir, "angle_dist_bplot.png"), bbox_inches='tight')
    plt.clf()


    latent = 0
    ax = plt.gca()
    for i in range(n_hidden):
        latent = latent + 1
        print(len(env_states[i]), len(baths[i]), i)
        if len(env_states[i]) == 0:
            continue
        sns.kdeplot(baths[i], ax=ax, label="Latent State " + str(latent), color=color_list[i], alpha=0.5, fill=True, linewidth=0, clip=(bath.min(), bath.max()))
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(out_dir, "bathymetry_kdeplots.png"), bbox_inches='tight')
    plt.clf()

    plt.boxplot(baths, labels=list(range(1,len(baths)+1)))
    plt.legend()
    plt.tight_layout(h_pad=0.1)
    plt.savefig(os.path.join(out_dir, "bathhymetry_bplot.png"), bbox_inches='tight')
    plt.clf()


    latent = 0
    ax = plt.gca()
    for i in range(n_hidden):
        latent = latent + 1 
        print(len(env_states[i]), len(ships[i]), i)
        if len(env_states[i]) == 0:
            continue
        print(min(ships[i]), max(ships[i]), "SHIPS")
        sns.kdeplot(ships[i], ax=ax, label="Latent State " + str(latent), color=color_list[i], alpha=0.5, fill=True, linewidth=0, clip=(0, ship.max()))
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(out_dir, "ship_density_kdeplots.png"), bbox_inches='tight')
    plt.clf()

    plt.boxplot(ships, labels=list(range(1,len(ships)+1)))
    plt.legend()
    plt.tight_layout(h_pad=0.1)
    plt.savefig(os.path.join(out_dir, "ship_density_bplot.png"), bbox_inches='tight')
    plt.clf()


    latent = 0
    ax = plt.gca()
    for i in range(n_hidden):
        latent = latent + 1
        print(len(env_states[i]), len(coasts[i]), i)
        if len(env_states[i]) == 0:
            continue
        print(min(coasts[i]), max(coasts[i]), "COASTS")
        sns.kdeplot(coasts[i], ax=ax, label="Latent State " + str(latent), color=color_list[i], alpha=0.5, fill=True, linewidth=0, clip=(0, coast.max()))
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(out_dir, "dist_to_coast_kdeplots.png"), bbox_inches='tight')
    plt.clf()


    plt.boxplot(coasts, labels=list(range(1,len(coasts)+1)))
    plt.legend()
    plt.tight_layout(h_pad=0.1)
    plt.savefig(os.path.join(out_dir, "env_to_coast_bplot.png"), bbox_inches='tight')
    plt.clf()

    return z_env_hist


def plot_reward_heatmaps(f, out_dir):

    f = normalize(f) 
    dists = []
    for j in range(f.shape[0]):
        f_tmp = np.squeeze(f[j,:])
        new_arr_size = math.ceil(math.sqrt(f_tmp.shape[0]))
        new_arr = np.zeros((new_arr_size**2))
        new_arr[:f_tmp.shape[0]] = f_tmp
        dists.append(np.ravel(f_tmp))

        im = plt.imshow(new_arr.reshape(new_arr_size, new_arr_size), cmap="jet", interpolation='none', vmin=0, vmax=1)
        plt.colorbar(im, location='bottom', pad=0.05, ticks=[np.min(new_arr), np.max(new_arr)],
                 format='${x:.1f}$')
        plt.tight_layout(h_pad=0.1)
        plt.savefig(os.path.join(out_dir, "reward_heatmap_" + str(j) + ".png"), bbox_inches='tight')
        plt.clf()

    plt.boxplot(dists, labels=list(range(1,f.shape[0]+1)))
    plt.legend()
    plt.tight_layout(h_pad=0.1)
    plt.savefig(os.path.join(out_dir, "reward_dist_bplot.png"), bbox_inches='tight')
    plt.clf()


def gen_cmap():
 
    palette_100 = sns.color_palette(cc.glasbey, n_colors=300)
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
    (line,) = ax.step(xs, ys, where="post", color="black", linewidth=3, zorder=1) #"black", linewidth=3, zorder=1)

    latest_point = ax.scatter(
        xs, ys, s=150,           # size controls marker size
    color=color_list[latents[0]], edgecolor="black", zorder=2 #"black", zorder=2
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
            color=color_list[latents[0]], edgecolor="white", zorder=2 #"black", zorder=2
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
    (line,) = ax.step(xs, ys, where="post", color="white", linewidth=2, zorder=1) #"black", linewidth=2, zorder=1)

    latest_point = ax.scatter(
        xs, ys, s=150,           # size controls marker size
    color=color_list[latents[0]], edgecolor="white", zorder=2 #"black", zorder=2
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

#def gen_invalid_inds(prev_state_map, n_state, n_action, n_latent):
#    invalid_indices = np.ones((n_latent, n_state), dtype=bool)
#
#    for x in range(n_state):
#        for prev_x_i in np.arange(n_latent):
#            if prev_x_i < len(prev_state_map[x]):
#                invalid_indices[prev_x_i, x] = False
#
#    return invalid_indices 


 


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

    key = "2017CA-Bmu-00826"  #"235283_6" #"2017CA-Bmu-00826"  #"1F90648_(4111)" #TODO
    df_uid = yml_conf["df_run_uid"] + "_" + key
    with open(os.path.join(yml_conf["df_dir"], df_uid + "_grid.pkl"), "rb") as f:
        grid = pickle.load(f)
  
    #with open(os.path.join(yml_conf["prev_envs"][0]), "rb") as f:
    #    prev_env = pickle.load(f)
 
    env_map_fname = yml_conf["envs"]
    with open(env_map_fname, "rb") as f:
        envs_init = pickle.load(f)

    if not os.path.exists(paths_fpaths[0]):
       gen_grid_point_paths(yml_conf) 
    
    uids_init = yml_conf["uids"]

    uids = []
    envs = []
    prev_envs = []
    print(len(paths_fpaths))
    for i in range(len(paths_fpaths)):
        paths_tmp = np.load(paths_fpaths[i], allow_pickle=True)
        envs_tmp = envs_init[uids_init[i]]

        #for k1 in range(len(envs_tmp)):
        #    for k2 in range(len(envs_tmp[k1])):
        #        print(envs_tmp[k1][k2].min(), envs_tmp[k1][k2].max(), envs_tmp[k1][k2].mean(), "ERROR")
                
        #with open(os.path.join(yml_conf["prev_envs"][i]), "rb") as f:
        #    prev_env_tmp = pickle.load(f)

        #print(paths_tmp)
        for j in range(len(paths_tmp)):
            max_len = max(max_len, len(paths_tmp[j]))

        print(envs_tmp[0][0].shape, len(envs_tmp))
        envs.extend(envs_tmp)
        for e in range(len(envs_tmp)):
            uids.append(uids_init[i])
            print("HERE FPATHS", i, paths_fpaths[i], e, len(envs_tmp), uids_init[i], uids[-1], len(uids))
        paths.extend(paths_tmp)
        #prev_envs.extend(prev_env_tmp) 

    #for now, at least, all path lengths have to be the same for SWIRL :(
    paths_final = []
    envs_final = []
    uids_final = []
    #prev_envs_final = []
    for j in range(len(paths)):
        print("HERE FINALIZE", j, len(paths[j]), max_len)
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

    #prev_state_map_fname = os.path.join(out_dir, run_uid + "_prev_state_map.pkl")
    #prev_state_map = np.load(prev_state_map_fname, allow_pickle=True)
    #invalid_indices = gen_invalid_inds(prev_state_map, n_states, n_actions, n_hidden)

    ##maze_info = np.load(folder + '/maze_info.npz', allow_pickle=True) #Dont need this here, but useful reference for more complex env reps.
  
    temps = jnp.array([0.01] + [1] * (n_hidden - 1))

    # Load S-2 params
    print("Load params and set reward values")
    fname = str(n_hidden) + '_' + str(seed) + "_MLP_S1_net1.npz"

    fname = os.path.join(out_dir, "s1_complex", fname)
    params2 = jnp.load(fname, allow_pickle=True)
    new_logpi02, new_log_Ps2, new_Rs2, new_R_state, LL_list2 = params2['new_logpi0'], params2['new_log_Ps'], params2['new_Rs'], params2['new_R_state'], params2['LL_list']

 
    fname = os.path.join(out_dir, "s1_complex", run_uid + "_" + str(n_hidden) + '_' + str(seed) + "_naturenet_MLP_S1_new1_LL.npz") 
    params3 = jnp.load(fname, allow_pickle=True)
    jax_path_vmap=params3["learnt_zs"]

    fname = os.path.join(out_dir, "s1_complex", run_uid + "_" + str(n_hidden) + '_' + str(seed) + "_naturenet_MLP_S1_new1_reward_filtered.npz")
    rwrd = jnp.load(fname, allow_pickle=True)
    reward2_filtered = rwrd["reward_filtered"]
    #print(invalid_indices.shape, reward2_filtered.shape)
 
    #reward2_filtered[invalid_indices,:] = np.nan
 
    color_list = ["red", "green", "blue", "brown", "cyan", "orange", "black", "magenta", "lightcoral", "goldenrod", "olive", "rosybrown", "silver", "sandybrown", "teal", "springgreen", "hotpink", "indigo", "darkkhaki", "navy", "red", "green", "blue", "brown", "cyan", "orange", "black", "magenta", "lightcoral", "goldenrod", "olive", "rosybrown", "silver", "sandybrown", "teal", "springgreen", "hotpink", "indigo", "darkkhaki", "navy"]
    print(reward2_filtered.shape)
    converted_map = np.nanmean(reward2_filtered, axis=-1)
    print(converted_map.shape) 
    plot_reward_heatmaps(converted_map, yml_conf["out_dir"])
 
    learnt_zs = np.array(jax_path_vmap)

    print("HERE CONVERTED MAP", converted_map.shape)
    if plot_gifs:
        for j in range(len(paths)):
            print( len(paths[j]), len(learnt_zs[j]), len(envs[j]), j)
            if len(paths[j]) < 25 or len(learnt_zs[j]) < 25 or len(envs[j]) < 25:
                print("CONTINUING", j, uids[j])
                continue
            #converted_map = np.nanmean(reward2_filtered, axis=-1)
            plot_env_heat_map_traj_animations(paths[j], envs[j], converted_map, learnt_zs[j], uids[j] + "_path" + str(j), out_dir, color_list)
            plot_env_map_traj_animations(paths[j], envs[j], learnt_zs[j], uids[j] + "_path" + str(j), out_dir, color_list)
            


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
                    #plot_env_maps(envs[j], converted_map, uids[j] + "_path" + str(j) + "_latent" + str(i), out_dir)
                    ##plot_env_prev_maps_traj_animations(paths[j], envs[j], prev_envs[j], uids[j] + "_path" + str(j) + "_latent" + str(i), out_dir)
        """

    if plot_traj:

        static_data = None #TODO - generalize and input fpath
        #with open("/data/nlahaye/NatureNet_Env/output_static/static_movement_kernel_info_prelim_scene_map.pkl",'rb') as f:
        #with open("/data/nlahaye/NatureNet_Env/output_static/static_movement_kernel_info_prelim_scene_map_shark.pkl",'rb') as f:
        with open("/data/nlahaye/NatureNet_Env/output_static/static_movement_kernel_info_prelim_scene_map.pkl",'rb') as f:
            static_data = pickle.load(f)
   
        print(converted_map.shape)
        z_env_hist = plot_density(learnt_zs, envs, paths, converted_map, n_hidden, out_dir, color_list, static_data) #, uids)

        for key in z_env_hist.keys():

            res = dict(sorted(z_env_hist[key].items(), key=operator.itemgetter(1), reverse=True)[:5])
            print("Latent state", key, "Top 5 env values", str(res))

        plot_time_series(learnt_zs, envs, paths, converted_map, n_hidden, out_dir, color_list, static_data, uids)



        figs = []
        axs = []
        #for i in range(n_hidden):
        #    fig, ax = plot_map_with_bounds(lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1])
        #    figs.append(fig)
        #    axs.append(ax)
        #figs, axs, lines_list = plot_trajs(learnt_zs, paths, grid, n_hidden, axs=axs, figs=figs)
  
 
        plt.clf()
        #plt.clear()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()

    #Translate config to dictionary 
    yml_conf = read_yaml(args.yaml)

    run_plots(yml_conf)



