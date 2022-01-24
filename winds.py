"""
    This script plots time and zonally averaged zonal surface wind for multiple gamma (averaged over a year's-worth of data from Isca)
"""

from glob import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import cftime
from shared_functions import *

def plots(ds1, ds2, ds3, ds4, labels):
    #following opens ERA5 re-analysis data
    file_ra = '/disca/share/pm366/ERA-5/era5_var131_masked_zm.nc'
    ds_ra = nc.Dataset(file_ra)
    t_ra = ds_ra.variables['time']
    lev = ds_ra.variables['lev'][:].data
    p_ra = lev/100 # convert to hPa
    lat_ra = ds_ra.variables['lat'][:].data
    u_ra = ds_ra.variables['ucomp']
    #times = []
    #times.append(cftime.num2pydate(t_ra, t_ra.units, t_ra.calendar)) # convert to actual dates
    #times = np.array(times)

    #set-up variables from the datasets
    #u0 = ds0.ucomp
    u1 = ds1.ucomp
    u2 = ds2.ucomp
    u3 = ds3.ucomp
    u4 = ds4.ucomp
    #u_hs = ds_hs.ucomp

    lat = ds1.coords['lat'].data
    lon = ds1.coords['lon'].data
    p = ds1.coords['pfull'].data
    
    #Time and Zonal Average Zonal Wind Speed at the pressure level closest to XhPa
    X = 900
    #uz0 = u0.mean(dim='time').mean(dim='lon').sel(pfull=X, method='nearest')
    uz1 = u1.mean(dim='time').mean(dim='lon').sel(pfull=X, method='nearest')
    uz2 = u2.mean(dim='time').mean(dim='lon').sel(pfull=X, method='nearest')
    uz3 = u3.mean(dim='time').mean(dim='lon').sel(pfull=X, method='nearest')
    uz4 = u4.mean(dim='time').mean(dim='lon').sel(pfull=X, method='nearest')
    #uz_hs = u_hs.mean(dim='time').mean(dim='lon').sel(pfull=X, method='nearest')

    label1 = labels[0] #r'$\gamma$ = 1.0'
    label2 = labels[1] #r'$\gamma$ = 2.0'
    label3 = labels[2] #r'$\gamma$ = 3.0'
    label4 = labels[3] #r'$\gamma$ = 4.0'

    fig = plt.subplots(1,1, figsize=(12,10))
    #plt.plot(lat, uz0, 'k', label='no vortex', linewidth='3.0')
    plt.plot(lat, uz1, ':k', label=label1)
    plt.plot(lat, uz2, '-.k', label=label2)
    plt.plot(lat, uz3, '--k', label=label3)
    plt.plot(lat, uz4, 'k', label=label4)
    #plt.plot(lat, uz_hs, 'b', label='Held-Suarez')
    plt.plot(lat_ra, np.mean(u_ra[491:494,np.where(p_ra == X)[0],:], axis=0)[0], 'r', label='ERA5 (DJF)') # plot re-analysis uwind against latitude average over Nov-19 to Feb-20 (NH winter)
    plt.xlabel('Latitude')
    plt.xlim(-90,90)
    plt.ylabel('Wind Speed (m/s)')
    plt.legend()
    plt.title('Time and Zonally Averaged Zonal Wind at ~{:.0f}hPa'.format(X))

    return plt.show()

if __name__ == '__main__': 
    
    #Set-up data
    exp = ['PK_eps0_vtx3_zoz13_7y', 'PK_eps0_vtx4_zoz13_7y', 'PK_eps10_vtx3_zoz13_7y', 'PK_eps0_vtx3_zoz13_heat']
    time = 'daily'
    years = 0 # user sets no. of years worth of data to ignore due to spin-up
    file_suffix = '_interp'
    
    ds1 = discard_spinup1(exp[0], time, file_suffix, years)
    ds2 = discard_spinup1(exp[1], time, file_suffix, years)
    ds3 = discard_spinup1(exp[2], time, file_suffix, years)
    ds4 = discard_spinup1(exp[3], time, '', years)
    #ds_hs = xr.open_mfdataset(sorted(glob('../isca_data/held_suarez_default/run*/atmos_monthly.nc'))[13:72], decode_times = False)

    plots(ds1, ds2, ds3, ds4, exp)
