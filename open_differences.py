"""
    This script plots differences between 2 datasets.
    Important that they are of the same resolution (e.g. both T21 or both T42).
"""

from glob import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from shared_functions import *

def plots(ds1, ds2, exp_name):
    # Set-up variables from the dataset
    lat = ds1.coords['lat'].data
    lon = ds1.coords['lon'].data
    p = ds1.coords['pfull'].data
    upper_p = ds1.coords['pfull'].sel(pfull=1, method='nearest') # in order to cap plots at pressure = 1hPa
    z = altitude(p)
    H = 8 #scale height km
    p0 = 1000 #surface pressure hPa    
    upper_z = -H*np.log(upper_p/p0)

    u_diff, T_diff = diff_variables(ds1, ds2, lat, lon, z, p)
    u = use_altitude(uz(ds1), z, lat, 'pfull', 'lat', r'ms$^{-1}$')
    T = use_altitude(Tz(ds1), z, lat, 'pfull', 'lat', 'K')

    fig1, ax1 = plt.subplots(figsize=(10,8))
    lvls1 = np.arange(-35, 37.5, 2.5)
    lvls2 = np.arange(-200, 200, 10)
    cs1 = u_diff.plot.contourf(levels=lvls1, cmap='RdBu_r', add_colorbar=False)
    ax1.contour(cs1, colors='gainsboro', linewidths=0.5)
    cs2 = ax1.contour(lat, z, u, colors='k', levels=lvls2, linewidths=0.5, alpha=0.4)
    cs2.collections[int(len(lvls2)/2)].set_linewidth(0.75)
    plt.colorbar(cs1, label=r'Difference (ms$^{-1}$)')
    plt.xlabel('Latitude', fontsize='large')
    plt.xlim(-90,90)
    plt.xticks([-90, -45, 0, 45, 90], ['90S', '45S', '0', '45N', '90N'])
    plt.ylabel('Pseudo-Altitude (km)', fontsize='large')
    plt.ylim(min(z), upper_z) #goes to ~1hPa
    plt.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
    plt.title('Difference in Mean Zonal Wind', fontsize='x-large')
    plt.savefig(exp_name+'_udiff.png', bbox_inches = 'tight')
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(10,8))
    lvls3 = np.arange(-35, 37.5, 2.5)
    lvls4 = np.arange(160, 330, 10)
    cs3 = T_diff.plot.contourf(levels=lvls3, cmap='RdBu_r', add_colorbar=False)
    ax2.contour(cs3, colors='gainsboro', linewidths=0.5)
    cs4 = ax2.contour(lat, z, T, colors='k', levels=lvls4, linewidths=0.5, alpha=0.4)
    plt.colorbar(cs3, label='Difference (K)')
    plt.xlabel('Latitude', fontsize='large')
    plt.xlim(-90,90)
    plt.xticks([-90, -45, 0, 45, 90], ['90S', '45S', '0', '45N', '90N'])
    plt.ylabel('Pseudo-Altitude (km)', fontsize='large')
    plt.ylim(min(z), upper_z) #goes to ~1hPa
    plt.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
    plt.title('Difference in Mean Temperature', fontsize='x-large')
    plt.savefig(exp_name+'_Tdiff.png', bbox_inches = 'tight')
    plt.close()

if __name__ == '__main__': 
    #Set-up data to be read in
    exp_name = 'PK_eps0_vtx3_zoz13_w15a4p500f800g50'
    exp = [exp_name, 'PK_eps0_vtx3_zoz13_7y']
    time = 'daily'
    years = 2 # user sets no. of years worth of data to ignore due to spin-up
    ds1 = discard_spinup1(exp[0], time, '_interp', years)
    ds2 = discard_spinup1(exp[1], time, '_interp', 0)
    

    plots(ds1, ds2, exp_name)