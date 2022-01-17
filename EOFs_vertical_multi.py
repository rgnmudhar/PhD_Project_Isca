"""
    Computes and plots the leading EOF of zonal wind as per Fig. 4 of Sheshadri and Plumb (2017)
    u anomaly is used to calculate the EOF, where u’(lat,p,time) = u(lat,p,time) - ubar(p), with ubar = zonal and time average.
    Based on Sheshadri and Plumb (2017), https://ajdawson.github.io/eofs/latest/examples/nao_xarray.html and code by William Seviour.
"""

import os
from glob import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from eofs.xarray import Eof
import statsmodels.api as sm

def add_phalf(exp_name, time, file_suffix, years):
    
    files = discard_spinup(exp_name, time, file_suffix, years)
    files_original = discard_spinup(exp_name, time, '', years)

    ds = xr.open_mfdataset(files, decode_times=False)
    ds_original = xr.open_mfdataset(files_original, decode_times=False)
    ds = ds.assign_coords({"phalf":ds_original.phalf})

    return ds

def discard_spinup(exp_name, time, file_suffix, years):
    # Ignore initial spin-up period of 2 years
    files = sorted(glob('../isca_data/'+exp_name+'/run*'+'/atmos_'+time+file_suffix+'.nc'))
    max_months = len(files)-1
    min_months = years*12
    files = files[min_months:max_months]

    return files

def calc_pc(ds, p_min, lat_min):
    # For EOFs follow Sheshadri & Plumb 2017, use p>100hPa, lat>20degN and zonal mean u
    uz = ds.ucomp\
        .sel(pfull=slice(p_min,1000))\
        .sel(lat=slice(lat_min,90))\
        .mean(dim='lon')

    # Calculate anomalies
    uz_anom = uz - uz.mean(dim='time')

    # sqrt(cos(lat)) weights due to different box sizes over grid
    sqrtcoslat = np.sqrt(np.cos(np.deg2rad(uz_anom.coords['lat'].values))) 

    # sqrt(dp) weights, select correct number of levels
    nplevs = uz_anom.coords['pfull'].shape[0]
    sqrtdp = np.sqrt(np.diff(ds.coords['phalf'].values[-nplevs-2:-1]))

    # calculate gridpoint weights
    wgts = np.outer(sqrtdp,sqrtcoslat)

    # Create an EOF solver to do the EOF analysis.
    solver = Eof(uz_anom.compute(), weights=wgts)

    # Retrieve the leading EOFs' PC time series
    # expressed  as the covariance between the leading PC time series and the input anomalies
    # By default PCs used are scaled to unit variance (dvided by square root of the eigenvalue)
    pc = solver.pcs(npcs=2, pcscaling=1)

    return pc

def plot(ds1, ds2, ds3, ds4, p_min, lat_min):
    # Plot equivalent to Fig 4a-c from Sheshadri & Plumb
    # First generate all necessary information for EOF analysis
    pc1 = calc_pc(ds1, p_min, lat_min)
    pc2 = calc_pc(ds2, p_min, lat_min)
    pc3 = calc_pc(ds3, p_min, lat_min)
    pc4 = calc_pc(ds4, p_min, lat_min)
    lags = 50 
    
    # Now plot on a single figure
    fig = plt.figure(figsize=(10,8))
    plt.acorr(pc1.sel(mode=0), maxlags=lags, usevlines=False, linestyle="-", marker=None, linewidth=1, label=r'$\gamma$ = 1.0')
    plt.acorr(pc2.sel(mode=0), maxlags=lags, usevlines=False, linestyle="-", marker=None, linewidth=1, label=r'$\gamma$ = 2.0')
    plt.acorr(pc3.sel(mode=0), maxlags=lags, usevlines=False, linestyle="-", marker=None, linewidth=1, label=r'$\gamma$ = 3.0')
    plt.acorr(pc4.sel(mode=0), maxlags=lags, usevlines=False, linestyle="-", marker=None, linewidth=1, label=r'$\gamma$ = 4.0')
    plt.legend()
    plt.axhline(0, color='k', linewidth=1)
    plt.axhline(1/np.e, color='k', linestyle=":")
    plt.xlim(-1*lags, lags)
    plt.xlabel('Lag (days)')
    plt.title('PC autocorrelation')

    plt.tight_layout()
    return plt.show()


if __name__ == '__main__':    
    exp = ['PK_eps0_vtx1_zoz18_7y', 'PK_eps0_vtx2_zoz18_7y', 'PK_eps0_vtx3_zoz18_7y', 'PK_eps0_vtx4_zoz18_7y']
    time = 'daily'
    years = 0 # user sets no. of years worth of data to ignore due to spin-up
    file_suffix = '_interp'
    
    ds1 = add_phalf(exp[0], time, file_suffix, years)
    ds2 = add_phalf(exp[1], time, file_suffix, years)
    ds3 = add_phalf(exp[2], time, file_suffix, years)
    ds4 = add_phalf(exp[3], time, file_suffix, years)

    # For EOFs follow Sheshadri & Plumb 2017, use p>100hPa, lat>20degN, zonal mean u
    p_min = 100  # hPa
    lat_min = 20  # degrees

    plot(ds1, ds2, ds3, ds4, p_min, lat_min)