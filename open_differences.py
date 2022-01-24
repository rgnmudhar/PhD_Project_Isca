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

    uz = diff_variables(ds1, ds2, lat, lon, z, p)

    fig1 = plt.figure(figsize=(10,8))
    lvls1 = np.arange(-20,22.5,2.5)
    ax1 = fig1.add_subplot(111)
    cs1 = uz.plot.contourf(levels=lvls1, cmap='RdBu_r', add_colorbar=False)
    ax1.contour(cs1, colors='gainsboro', linewidths=1)
    plt.colorbar(cs1, label=r'Difference (ms$^{-1}$)')
    plt.xlabel('Latitude', fontsize='large')
    plt.xlim(-90,90)
    plt.xticks([-90, -45, 0, 45, 90], ['90S', '45S', '0', '45N', '90N'])
    plt.ylabel('Pseudo-Altitude (km)', fontsize='large')
    plt.ylim(min(z), upper_z) #goes to ~1hPa
    plt.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
    plt.title('Difference in Mean Zonal Wind', fontsize='x-large')
    plt.savefig(exp_name+'_diff.png', bbox_inches = 'tight')
    plt.close()

if __name__ == '__main__': 
    #Set-up data to be read in
    exp = ['PK_eps0_vtx4_zoz18_7y','PK_eps10_vtx4_zoz18_7y']
    time = 'daily'
    years = 0 # user sets no. of years worth of data to ignore due to spin-up
    ds1 = discard_spinup1(exp[0], time, '_interp', years)
    ds2 = discard_spinup1(exp[1], time, '_interp', years)
    exp_name = 'PK_eps0-10_zoz18_vtx4'

    plots(ds1, ds2, exp_name)