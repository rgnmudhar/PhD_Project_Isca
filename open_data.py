"""
    This script plots (time and zonal) averages of various output variables averaged over X years'-worth of data from Isca
    Also plots differences between 2 datasets - important that they are of the same resolution (e.g. both T21 or both T42)
"""

from glob import glob
import imageio
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from open_winds import *
from shared_functions import *
from datetime import datetime

def plot(u, T, lvls, heat, lat, p, exp_name, vertical):
    print(datetime.now(), " - plotting")
    # Plots time and zonal-mean Zonal Wind Speed and Temperature
    fig, ax = plt.subplots(figsize=(6,6))

    if vertical == "a":
        csa = ax.contourf(lat, p, T, levels=lvls[0], cmap='Blues_r')
        csb = ax.contour(lat, p, u, colors='k', levels=lvls[1], linewidths=1)
        plt.contour(lat, p, heat, colors='g', linewidths=1, alpha=0.4, levels=11)
        plt.ylabel('Pressure (hPa)', fontsize='x-large')
        plt.ylim(max(p), upper_p) #goes to ~1hPa
        plt.yscale('log')

    if vertical == "b":
        # Use altitude rather than pressure for vertical
        z = altitude(p)
        upper_z = -7*np.log(upper_p/1000)
        u = use_altitude(u, z, lat, 'pfull', 'lat', r'ms$^{-1}$')
        T = use_altitude(T, z, lat, 'pfull', 'lat', 'K')
        H = use_altitude(heat, z, lat, 'pfull', 'lat', r'Ks$^{-1}$')
        csa = T.plot.contourf(levels=lvls[0], cmap='Blues_r', add_colorbar=False)
        csb = ax.contour(lat, z, u, colors='k', levels=lvls[1], linewidths=1)
        plt.contour(lat, z, H, colors='g', linewidths=1, alpha=0.4, levels=11)
        plt.ylabel('Pseudo-Altitude (km)', fontsize='x-large')
        plt.ylim(min(z), upper_z) #goes to ~1hPa

    ax.contourf(csa, colors='none')
    csb.collections[int(len(lvls[1])/2)].set_linewidth(1.5)
    cb = plt.colorbar(csa)
    cb.set_label(label='Temperature (K)', size='x-large')
    cb.ax.tick_params(labelsize='x-large')
    plt.xlabel(r'Latitude ($\degree$N)', fontsize='x-large')
    plt.xlim(0,90)
    plt.xticks([10, 30, 50, 70, 90], ['10', '30', '50', '70', '90'])    
    plt.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    plt.savefig(exp_name+'_zonal.pdf', bbox_inches = 'tight')

    return plt.close()

def plot_diff(vars, units, lvls, heat, lat, p, exp_name, vertical):
    # Plots differences in time and zonal-mean of variables (vars)
    lvls_diff = np.arange(-20, 22.5, 2.5)
    for i in range(len(vars)):
        print(datetime.now(), " - taking differences")
        x = diff_variables(vars[i], lat, p, units[i])
        print(datetime.now(), " - plotting")
        fig, ax = plt.subplots(figsize=(6,6))
        
        if vertical == "a":
            cs1 = ax.contourf(lat, p, x, levels=lvls_diff, cmap='RdBu_r')
            cs2 = ax.contour(lat, p, vars[i][0], colors='k', levels=lvls[i], linewidths=1, alpha=0.4)
            plt.contour(lat, p, heat, colors='g', linewidths=1, alpha=0.4, levels=11)
            plt.ylabel('Pressure (hPa)', fontsize='x-large')
            plt.ylim(max(p), upper_p) #goes to ~1hPa
            plt.yscale('log')

        if vertical == "b":
            # Use altitude rather than pressure for vertical
            z = altitude(p)
            upper_z = -7*np.log(upper_p/1000)
            var_z = use_altitude(vars[i][0], z, lat, 'pfull', 'lat', units[i])
            H = use_altitude(heat, z, lat, 'pfull', 'lat', r'Ks$^{-1}$')
            cs1 = x.plot.contourf(levels=lvls_diff, cmap='RdBu_r', add_colorbar=False)
            cs2 = ax.contour(lat, z, var_z, colors='k', levels=lvls[i], linewidths=1)
            plt.contour(lat, z, H, colors='g', linewidths=1, alpha=0.4, levels=11)
            plt.ylabel('Pseudo-Altitude (km)', fontsize='x-large')
            plt.ylim(min(z), upper_z) #goes to ~1hPa

        ax.contourf(cs1, colors='none')        
        cs2.collections[int(len(lvls[i])/2)].set_linewidth(1.5)
        cb = plt.colorbar(cs1)
        cb.set_label(label='Difference ('+units[i]+')', size='x-large')
        cb.ax.tick_params(labelsize='x-large')        
        plt.xlabel(r'Latitude ($\degree$N)', fontsize='x-large')
        plt.xlim(0,90)
        plt.xticks([10, 30, 50, 70, 90], ['10', '30', '50', '70', '90'])        
        plt.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
        plt.savefig(exp_name+'_diff{:.0f}.pdf'.format(i), bbox_inches = 'tight')
        plt.close()

if __name__ == '__main__': 
    #Set-up data to be read in
    indir = '/disco/share/rm811/processed/'
    basis = 'PK_e0v4z13'
    perturb = '_q6m2y45l800u200'
    ctrl = basis+perturb
    # make ctrl the first experiment for plotting diffs
    exp = [ctrl,\
        basis+'_w15a4p800f800g50'+perturb]
    upper_p = 1 # hPa
    lvls = [np.arange(160, 330, 10), np.arange(-200, 205, 5)]

    plot_type = input("Plot a) single experiments or b) difference between 2 experiments?")
    height_type = input("Plot vs. a) log pressure or b) pseudo-altitude?")

    for i in range(len(exp)):
        print(datetime.now(), " - opening files ({0:.0f}/{1:.0f})".format(i, len(exp)))
        #Read in data to plot polar heat contours
        if i == 0:
            heat = 0
        elif i != 0:
            file = '/disco/share/rm811/isca_data/' + exp[i] + '/run0100/atmos_daily_interp.nc'
            ds = xr.open_dataset(file)
            heat = ds.local_heating.sel(lon=180, method='nearest').mean(dim='time')    
        uz = xr.open_dataset(indir+exp[i]+'_utz.nc', decode_times=False).ucomp[0]
        Tz = xr.open_dataset(indir+exp[i]+'_Ttz.nc', decode_times=False).temp[0]
        lat = uz.coords['lat'].data
        p = uz.coords['pfull'].data
        #MSF = calc_streamfn(xr.open_dataset(indir+exp[0]+'_vtz.nc', decode_times=False).vcomp[0], p, lat)  # Meridional Stream Function
        #MSF_xr = xr.DataArray(MSF, coords=[p,lat], dims=['pfull','lat'])  # Make into an xarray DataArray
        #MSF_xr.attrs['units']=r'kgs$^{-1}$'

        if plot_type =='a':
            plot(uz, Tz, lvls, heat, lat, p, exp[i], height_type)
        elif plot_type == 'b':
            if i == 0:
                print("skipping control")
            elif i != 0:
                u = [uz, xr.open_dataset(indir+exp[0]+'_utz.nc', decode_times=False).ucomp[0]]
                T = [Tz, xr.open_dataset(indir+exp[0]+'_Ttz.nc', decode_times=False).temp[0]]
                plot_diff([T, u], ['K', r'ms$^{-1}$'], [np.arange(160, 330, 10), np.arange(-200, 210, 10)],\
                    heat, lat, p, exp[i], height_type)

