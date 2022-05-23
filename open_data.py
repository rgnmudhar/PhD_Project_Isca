"""
    This script plots (time and zonal) averages of various output variables averaged over X years'-worth of data from Isca
    Also plots differences between 2 datasets - important that they are of the same resolution (e.g. both T21 or both T42)
"""

from glob import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from shared_functions import *

def plot_single(ds, exp_name):
    # Set-up variables from the dataset
    tm = ds.coords['time'].data
    lat = ds.coords['lat'].data
    lon = ds.coords['lon'].data
    p = ds.coords['pfull'].data
    upper_p = ds.coords['pfull'].sel(pfull=1, method='nearest') # in order to cap plots at pressure = 1hPa
    u = uz(ds)
    T = Tz(ds)
    
    #Teq = Teqz(ds)

    #heat = ds.local_heating

    #MSF = v(ds, p, lat) # Zonal Average Meridional Stream Function
    #MSF_xr = xr.DataArray(MSF, coords=[p,lat], dims=['pfull','lat'])  # Make into an xarray DataArray
    #MSF_xr.attrs['units']='kg/s'
        
    #T_anm = T.mean(dim='time').mean(dim='lon')
    #P_surf_anm = P_surf.mean(dim='time').mean(dim='lon')
    #theta = T_potential(p,P_surf_anm,T_anm,lat) # Zonal Average Potential Temperature
    #theta_xr = xr.DataArray(theta, coords=[p,lat], dims=['pfull','lat'])  # Make into an xarray DataArray
    #theta_xr.attrs['units']='K'

    #z_surf = ds.zsurf.mean(dim='time') # Average Surface Topography 

    # Plots of means from across the time period
    # Zonal Average Zonal Wind Speed and Temperature
    lvls2a = np.arange(160, 330, 5)
    lvls2b = np.arange(-200, 200, 5)
    fig2, ax = plt.subplots(figsize=(8,6))
    cs2a = ax.contourf(lat, p, T, levels=lvls2a, cmap='RdBu_r')
    ax.contourf(cs2a, colors='none')
    cs2b = ax.contour(lat, p, u, colors='k', levels=lvls2b, linewidths=1)
    cs2b.collections[int(len(lvls2b)/2)].set_linewidth(1.5)
    cb = plt.colorbar(cs2a)
    cb.set_label(label='Temperature (K)', size='x-large')
    cb.ax.tick_params(labelsize='x-large')
    plt.xlabel('Latitude', fontsize='x-large')
    plt.xlim(-90,90)
    plt.xticks([-90, -45, 0, 45, 90], ['90S', '45S', '0', '45N', '90N'])
    plt.ylabel('Pressure (hPa)', fontsize='x-large')
    plt.ylim(max(p), upper_p) #goes to ~1hPa
    plt.yscale('log')
    plt.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    plt.savefig(exp_name+'_zonal.pdf', bbox_inches = 'tight')

    """
    # Use altitude rather than pressure for vertical
    z = altitude(p)
    H = 8 #scale height km
    p0 = 1000 #surface pressure hPa    
    upper_z = -H*np.log(upper_p/p0)

    u = use_altitude(u, z, lat, 'pfull', 'lat', r'ms$^{-1}$')
    T = use_altitude(T, z, lat, 'pfull', 'lat', 'K')
    Teq = use_altitude(Teq, z, lat, 'pfull', 'lat', 'K')

    # Zonal Average Zonal Wind Speed and Temperature
    lvls2a = np.arange(160, 330, 5)
    lvls2b = np.arange(-200, 200, 5)
    fig2, ax = plt.subplots(figsize=(8,6))
    cs2a = T.plot.contourf(levels=lvls2a, cmap='RdBu_r', add_colorbar=False)
    ax.contourf(cs2a, colors='none')
    cs2b = ax.contour(lat, z, u, colors='k', levels=lvls2b, linewidths=1)
    cs2b.collections[int(len(lvls2b)/2)].set_linewidth(1.5)
    #plt.clabel(cs2b, levels = lvls2b[::4], inline=1, fontsize='x-small')
    cb = plt.colorbar(cs2a)
    cb.set_label(label='Temperature (K)', size='x-large')
    cb.ax.tick_params(labelsize='x-large')
    plt.xlabel('Latitude', fontsize='x-large')
    plt.xlim(-90,90)
    plt.xticks([-90, -45, 0, 45, 90], ['90S', '45S', '0', '45N', '90N'])
    plt.ylabel('Pseudo-Altitude (km)', fontsize='x-large')
    plt.ylim(min(z), upper_z) #goes to ~1hPa
    plt.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    #plt.title('Mean Zonal Wind and Temperature', fontsize='x-large')
    plt.savefig(exp_name+'_zonal.pdf', bbox_inches = 'tight')
    """
    return plt.close()

def plot_diff(ds1, ds2, exp_name):
    # Set-up variables from the dataset
    lat = ds1.coords['lat'].data
    lon = ds1.coords['lon'].data
    p = ds1.coords['pfull'].data
    upper_p = ds1.coords['pfull'].sel(pfull=1, method='nearest') # in order to cap plots at pressure = 1hPa
    z = altitude(p)

    u_diff, T_diff = diff_variables(ds1, ds2, lat, p)
    u = uz(ds1)
    T = Tz(ds1)

    lvls1 = np.arange(-35, 37.5, 2.5)
    lvls2 = np.arange(-200, 200, 10)
    fig1, ax1 = plt.subplots(figsize=(8,6))
    cs1 = ax1.contourf(lat, p, u_diff, levels=lvls1, cmap='RdBu_r')
    ax1.contourf(cs1, colors='none')
    cs2 = ax1.contour(lat, p, u, colors='k', levels=lvls2, linewidths=1)
    cs2.collections[int(len(lvls2)/2)].set_linewidth(1.5)
    cb = plt.colorbar(cs1)
    cb.set_label(label=r'Difference (ms$^{-1}$)', size='x-large')
    cb.ax.tick_params(labelsize='x-large')
    plt.xlabel('Latitude', fontsize='x-large')
    plt.xlim(-90,90)
    plt.xticks([-90, -45, 0, 45, 90], ['90S', '45S', '0', '45N', '90N'])
    plt.ylabel('Pressure (hPa)', fontsize='x-large')
    plt.ylim(max(p), upper_p) #goes to ~1hPa
    plt.yscale('log')
    plt.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    #plt.title('Difference in Mean Zonal Wind', fontsize='x-large')
    plt.savefig(exp_name+'_udiff.pdf', bbox_inches = 'tight')
    plt.close()

    lvls3 = np.arange(-35, 37.5, 2.5)
    lvls4 = np.arange(160, 330, 10)
    fig2, ax2 = plt.subplots(figsize=(10,8))
    cs3 = ax2.contourf(lat, p, T_diff, levels=lvls3, cmap='RdBu_r')
    ax2.contourf(cs3, colors='none')
    cs4 = ax2.contour(lat, p, T, colors='k', levels=lvls4, linewidths=1)
    cs4.collections[int(len(lvls4)/2)].set_linewidth(1.5)
    cb = plt.colorbar(cs3)
    cb.set_label(label=r'Temperature (K)', size='x-large')
    cb.ax.tick_params(labelsize='x-large')
    plt.xlabel('Latitude', fontsize='x-large')
    plt.xlim(-90,90)
    plt.xticks([-90, -45, 0, 45, 90], ['90S', '45S', '0', '45N', '90N'])
    plt.ylabel('Pressure (hPa)', fontsize='x-large')
    plt.ylim(max(p), upper_p) #goes to ~1hPa
    plt.yscale('log')
    plt.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
    #plt.title('Difference in Mean Temperature', fontsize='x-large')
    plt.savefig(exp_name+'_Tdiff.pdf', bbox_inches = 'tight')
    return plt.close()

if __name__ == '__main__': 
    #Set-up data to be read in
    basis = 'PK_e0v4z13'
    exp_name = basis+'_w15a4p300f800g50_q6m2y45l800u200'
    exp = [exp_name, basis]
    time = 'daily'
    file_suffix = '_interp'
    years = 2 # user sets no. of years worth of data to ignore due to spin-up
    
    plot_type = input("Plot a) single experiment or b) difference between 2 experiments?")
    if plot_type =='a':
        ds = discard_spinup1(exp[0], time, file_suffix, years)
        plot_single(ds, exp_name)
    elif plot_type == 'b':
        ds1 = discard_spinup1(exp[0], time, file_suffix, years)
        ds2 = discard_spinup1(exp[1], time, file_suffix, years)
        plot_diff(ds1, ds2, exp_name)
