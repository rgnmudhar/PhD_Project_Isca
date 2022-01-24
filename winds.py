"""
Plots time and zonally averaged zonal wind at X hPa for multiple experiments
"""

from codecs import utf_16_be_decode
from glob import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import cftime
from shared_functions import *

def plots(ds1, ds2, ds3, ds4, labels, fig_name):
    """
    Sets up all necessary coordinates and variables.
    Then plots time and zonal average zonal wind at the pressure level closest to XhPa
    """
    
    X = 900

    #Following opens ERA5 re-analysis data
    file_ra = '/disca/share/pm366/ERA-5/era5_var131_masked_zm.nc'
    ds_ra = nc.Dataset(file_ra)
    t_ra = ds_ra.variables['time']
    lev = ds_ra.variables['lev'][:].data
    p_ra = lev/100 # convert to hPa
    lat_ra = ds_ra.variables['lat'][:].data
    u_ra = ds_ra.variables['ucomp']

    #Following writes date/time of ERA-5 data
    #times = []
    #times.append(cftime.num2pydate(t_ra, t_ra.units, t_ra.calendar)) # convert to actual dates
    #times = np.array(times)

    #Following sets up variables from the Isca datasets
    lat = ds1.coords['lat'].data
    lon = ds1.coords['lon'].data
    p = ds1.coords['pfull'].data
    uz1 = ds1.ucomp.mean(dim='time').mean(dim='lon').sel(pfull=X, method='nearest')
    uz2 = ds2.ucomp.mean(dim='time').mean(dim='lon').sel(pfull=X, method='nearest')
    uz3 = ds3.ucomp.mean(dim='time').mean(dim='lon').sel(pfull=X, method='nearest')
    uz4 = ds4.ucomp.mean(dim='time').mean(dim='lon').sel(pfull=X, method='nearest')

    fig = plt.subplots(1,1, figsize=(10,8))
    plt.plot(lat, uz1, ':k', label=labels[0])
    plt.plot(lat, uz2, '-.k', label=labels[1])
    plt.plot(lat, uz3, '--k', label=labels[2])
    plt.plot(lat, uz4, 'k', label=labels[3])
    plt.plot(lat_ra, np.mean(u_ra[491:494,np.where(p_ra == X)[0],:], axis=0)[0], color='#27AE60', label='ERA5 (DJF)') # plot re-analysis uwind against latitude average over Nov-19 to Feb-20 (NH winter)
    plt.xlabel('Latitude', fontsize='large')
    plt.xlim(-90,90)
    plt.xticks([-90, -45, 0, 45, 90], ['90S', '45S', '0', '45N', '90N'])
    plt.ylabel(r'Wind Speed (ms$^{-1}$)', fontsize='large')
    plt.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
    plt.legend(loc='upper center' , bbox_to_anchor=(0.5, -0.07),fancybox=False, shadow=True, ncol=5, fontsize='large')
    plt.title('Time and Zonally Averaged Zonal Wind at ~{:.0f}hPa'.format(X), fontsize='x-large')
    plt.savefig(fig_name+'_winds.png', bbox_inches = 'tight')
    plt.close()

    return plt.show()

if __name__ == '__main__': 
    #Set-up data
    exp = ['PK_eps0_vtx1_zoz13_7y', 'PK_eps0_vtx2_zoz13_7y', 'PK_eps0_vtx3_zoz13_7y', 'PK_eps0_vtx4_zoz13_7y']
    time = 'daily'
    years = 0 # user sets no. of years worth of data to ignore due to spin-up
    file_suffix = '_interp'
    
    ds1 = discard_spinup1(exp[0], time, file_suffix, years)
    ds2 = discard_spinup1(exp[1], time, file_suffix, years)
    ds3 = discard_spinup1(exp[2], time, file_suffix, years)
    ds4 = discard_spinup1(exp[3], time, file_suffix, years)

    labels = [r'$\gamma$ = 1',r'$\gamma$ = 2',r'$\gamma$ = 3',r'$\gamma$ = 4']

    plots(ds1, ds2, ds3, ds4, labels, 'PK_eps0_zoz13')
