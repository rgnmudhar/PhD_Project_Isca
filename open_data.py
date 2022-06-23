"""
    This script plots (time and zonal) averages of various output variables averaged over X years'-worth of data from Isca
    Also plots differences between 2 datasets - important that they are of the same resolution (e.g. both T21 or both T42)
"""

from glob import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from open_winds import *
from datetime import datetime

def plot_single(u, T, heat, lat, p, upper_p, exp_name):
    print(datetime.now(), " - plotting")
    # Plots of means from across the time period
    # Zonal Average Zonal Wind Speed and Temperature
    lvls2a = np.arange(160, 330, 10)
    lvls2b = np.arange(-200, 200, 5)
    fig2, ax = plt.subplots(figsize=(6,6))
    cs2a = ax.contourf(lat, p, T, levels=lvls2a, cmap='Blues_r')
    ax.contourf(cs2a, colors='none')
    cs2b = ax.contour(lat, p, u, colors='k', levels=lvls2b, linewidths=1)
    cs2b.collections[int(len(lvls2b)/2)].set_linewidth(1.5)
    cb = plt.colorbar(cs2a)
    cb.set_label(label='Temperature (K)', size='x-large')
    cb.ax.tick_params(labelsize='x-large')
    plt.contour(lat, p, heat, colors='g', linewidths=1, alpha=0.4, levels=11)
    plt.xlabel('Latitude', fontsize='x-large')
    plt.xlim(0,90)
    plt.xticks([10, 30, 50, 70, 90], ['10', '30', '50', '70', '90'])
    plt.ylabel('Pressure (hPa)', fontsize='x-large')
    plt.ylim(max(p), upper_p) #goes to ~1hPa
    plt.yscale('log')
    plt.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    plt.savefig(exp_name+'_zonal.pdf', bbox_inches = 'tight')

    """
    # Use altitude rather than pressure for vertical
    z = altitude(p)
    H = 8 #scale height km
    upper_z = -H*np.log(upper_p/1000)

    u = use_altitude(u, z, lat, 'pfull', 'lat', r'ms$^{-1}$')
    T = use_altitude(T, z, lat, 'pfull', 'lat', 'K')
    H = use_altitude(heat, z, lat, 'pfull', 'lat', r'Ks$^{-1}$')

    # Zonal Average Zonal Wind Speed and Temperature
    lvls2a = np.arange(160, 330, 5)
    lvls2b = np.arange(-200, 200, 5)
    fig2, ax = plt.subplots(figsize=(6,6))
    cs2a = T.plot.contourf(levels=lvls2a, cmap='Blues_r', add_colorbar=False)
    ax.contourf(cs2a, colors='none')
    cs2b = ax.contour(lat, z, u, colors='k', levels=lvls2b, linewidths=1)
    cs2b.collections[int(len(lvls2b)/2)].set_linewidth(1.5)
    #plt.clabel(cs2b, levels = lvls2b[::4], inline=1, fontsize='x-small')
    cb = plt.colorbar(cs2a)
    cb.set_label(label='Temperature (K)', size='x-large')
    cb.ax.tick_params(labelsize='x-large')
    plt.contour(lat, z, H, colors='g', linewidths=1, alpha=0.4, levels=11)
    plt.xlabel('Latitude', fontsize='x-large')
    plt.xlim(0,90)
    plt.xticks([10, 30, 50, 70, 90], ['10', '30', '50', '70', '90'])
    plt.ylabel('Pseudo-Altitude (km)', fontsize='x-large')
    plt.ylim(min(z), upper_z) #goes to ~1hPa
    plt.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    plt.savefig(exp_name+'_zonal.pdf', bbox_inches = 'tight')
    """

    return plt.close()

def plot_diff(u, T, heat, lat, p, upper_p, exp_name):
    # Set-up variables from the dataset
    print(datetime.now(), " - taking differences")
    u_diff, T_diff = diff_variables(u, T, lat, p)

    print(datetime.now(), " - plotting u diff")
    lvls1 = np.arange(-20, 22.5, 2.5)
    lvls2 = np.arange(-200, 200, 10)
    fig1, ax1 = plt.subplots(figsize=(6,6))
    cs1 = ax1.contourf(lat, p, u_diff, levels=lvls1, cmap='RdBu_r')
    ax1.contourf(cs1, colors='none')
    cs2 = ax1.contour(lat, p, u[0], colors='k', levels=lvls2, linewidths=1, alpha=0.4)
    cs2.collections[int(len(lvls2)/2)].set_linewidth(1.5)
    cb = plt.colorbar(cs1)
    cb.set_label(label=r'Difference (ms$^{-1}$)', size='x-large')
    cb.ax.tick_params(labelsize='x-large')
    plt.contour(lat, p, heat, colors='g', linewidths=1, alpha=0.4, levels=11)
    plt.xlabel(r'Latitude ($\degree$N)', fontsize='x-large')
    plt.xlim(0,90)
    plt.xticks([10, 30, 50, 70, 90], ['10', '30', '50', '70', '90'])
    plt.ylabel('Pressure (hPa)', fontsize='x-large')
    plt.ylim(max(p), upper_p) #goes to ~1hPa
    plt.yscale('log')
    plt.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    plt.savefig(exp_name+'_udiff.pdf', bbox_inches = 'tight')
    plt.close()

    print(datetime.now(), " - plotting T diff")
    lvls3 = np.arange(-20, 22.5, 2.5)
    lvls4 = np.arange(160, 330, 10)
    fig2, ax2 = plt.subplots(figsize=(6,6))
    cs3 = ax2.contourf(lat, p, T_diff, levels=lvls3, cmap='RdBu_r')
    ax2.contourf(cs3, colors='none')
    cs4 = ax2.contour(lat, p, T[0], colors='k', levels=lvls4, linewidths=1, alpha=0.4)
    cs4.collections[int(len(lvls4)/2)].set_linewidth(1.5)
    cb = plt.colorbar(cs3)
    cb.set_label(label=r'Temperature (K)', size='x-large')
    cb.ax.tick_params(labelsize='x-large')
    plt.contour(lat, p, heat, colors='g', linewidths=1, alpha=0.4, levels=11)
    plt.xlabel(r'Latitude ($\degree$N)', fontsize='x-large')
    plt.xlim(0,90)
    plt.xticks([10, 30, 50, 70, 90], ['10', '30', '50', '70', '90'])
    plt.ylabel('Pressure (hPa)', fontsize='x-large')
    plt.ylim(max(p), upper_p) #goes to ~1hPa
    plt.yscale('log')
    plt.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
    plt.savefig(exp_name+'_Tdiff.pdf', bbox_inches = 'tight')
    return plt.close()

if __name__ == '__main__': 
    #Set-up data to be read in
    indir = '/disco/share/rm811/processed/'
    basis = 'PK_e0v4z13'
    filename = 'w15a4p300f800g50_q6m2y45l800u200'
    exp = [basis+'_'+filename, basis+'_q6m2y45l800u200']

    #Read in data to plot polar heat contours
    file = '/disco/share/rm811/isca_data/' + basis + '_' + filename + '/run0100/atmos_daily_interp.nc'
    ds = xr.open_dataset(file)
    heat = ds.local_heating.sel(lon=180, method='nearest').mean(dim='time')

    plot_type = input("Plot a) single experiment or b) difference between 2 experiments?")

    print(datetime.now(), " - opening files")
    uz = xr.open_dataset(indir+exp[0]+'_utz.nc', decode_times=False).ucomp[0]
    Tz = xr.open_dataset(indir+exp[0]+'_Ttz.nc', decode_times=False).temp[0]
    lat = uz.coords['lat'].data
    p = uz.coords['pfull'].data
    upper_p = uz.coords['pfull'].sel(pfull=1, method='nearest') # in order to cap plots at pressure = 1hPa

    if plot_type =='a':
        plot_single(uz, Tz, heat, lat, p, upper_p, exp[0])
    elif plot_type == 'b':
        u = [uz, xr.open_dataset(indir+exp[1]+'_utz.nc', decode_times=False).ucomp[0]]
        T = [Tz, xr.open_dataset(indir+exp[1]+'_Ttz.nc', decode_times=False).temp[0]]
        plot_diff(u, T, heat, lat, p, upper_p, exp[0])

#Meridional Stream Function
#MSF = calc_streamfn(xr.open_dataset(indir+exp[0]+'_vtz.nc', decode_times=False).vcomp[0], p, lat) 
#MSF_xr = xr.DataArray(MSF, coords=[p,lat], dims=['pfull','lat'])  # Make into an xarray DataArray
#MSF_xr.attrs['units']=r'kgs$^{-1}$'