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
from shared_functions import *

def eof_u(ds, p_min, lat_min):
    # For EOFs follow Sheshadri & Plumb 2017, use p>100hPa, lat>20degN and zonal mean u
    uz = ds.ucomp\
        .sel(pfull=slice(p_min,1000))\
        .sel(lat=slice(lat_min,90))\
        .mean(dim='lon')
    return uz

def eof_solver(ds, p_min, lat_min):
    uz = eof_u(ds, p_min, lat_min)

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

    return solver

def leading_eofs(solver):
    # Retrieve the leading 2 EOFs and leading PC time series
    # expressed  as the covariance between the leading PC time series and the input anomalies
    # By default PCs used are scaled to unit variance (dvided by square root of the eigenvalue)
    eofs = solver.eofsAsCovariance(neofs=2)
    pcs = solver.pcs(npcs=2, pcscaling=1)

    return eofs

def leading_pcs(solver):
    # Retrieve the leading 2 EOFs and leading PC time series
    # expressed  as the covariance between the leading PC time series and the input anomalies
    # By default PCs used are scaled to unit variance (dvided by square root of the eigenvalue)
    pcs = solver.pcs(npcs=2, pcscaling=1)

    return pcs

def variance(solver):
    # Fractional EOF mode variances.
    # The fraction of the total variance explained by each EOF mode, values between 0 and 1 inclusive.
    return solver.varianceFraction()

def findtau(ac):
    for i in range(len(ac)):
        if ac[i] - 1/np.e < 1e-5:
            tau = i
            break
    return tau

def AM_times(pcs, lags):
    # Find autocorrelation function in order to determine decorrelation time (i.e. time for correlation reduce to 1/e)
    ac1 = sm.tsa.acf(pcs.sel(mode=0).values, nlags=lags)
    ac2 = sm.tsa.acf(pcs.sel(mode=1).values, nlags=lags)

    tau1 = findtau(ac1)
    tau2 = findtau(ac2)

    return tau1, tau2

def plot_single(ds, p_min, lat_min, exp_name, alt=True):
    # Plot equivalent to Fig 4a-c from Sheshadri & Plumb
    # First generate all necessary information for EOF analysis
    uz = eof_u(ds, p_min, lat_min)
    solver = eof_solver(ds, p_min, lat_min)
    eofs = leading_eofs(solver)
    pcs = leading_pcs(solver)
    variance_fractions = variance(solver)
    lags = 100 
    tau1, tau2 = AM_times(pcs, lags)

    # Set-up coordinates
    lat = ds.coords['lat'].sel(lat=slice(lat_min,90))
    p = ds.coords['pfull'].sel(pfull=slice(p_min,1000))
    z = altitude(p)

    # Set-up variables
    if alt==True:
        # Use altitude rather than pressure for vertical
        uz = use_altitude(uz.mean(dim='time'), z, lat, 'pfull', 'lat', 'm/s')
        eof1 = use_altitude(eofs.sel(mode=0), z, lat, 'pfull', 'lat', '')
        eof2 = use_altitude(eofs.sel(mode=1), z, lat, 'pfull', 'lat', '')
    else:
        uz = uz.mean(dim='time')
        eof1 = eofs.sel(mode=0)
        eof2 = eofs.sel(mode=1)
    
    # Now plot on a single figure
    fig = plt.figure(figsize=(15,5))
    fig.suptitle(exp_name, fontsize='large')

    ax1 = fig.add_subplot(1,3,1)
    if alt==True:
        eof1.plot.contourf(ax=ax1, cmap='RdBu_r', levels=21)
        uz.plot.contour(ax=ax1, colors='k', levels=15)
        ax1.set_ylabel('Pseudo-Altitude (km)')
    else:
        eof1.plot.contourf(ax=ax1, cmap='RdBu_r', yincrease=False, levels=21)
        uz.plot.contour(ax=ax1, colors='k', yincrease=False, levels=15)
    ax1.set_title(r'EOF1, {0:.0f}% of variance, $\tau$ = {1:.0f} days'\
        .format(100*variance_fractions.values[0],\
        tau1))
    

    ax2 = fig.add_subplot(1,3,2)
    if alt==True:
        eof2.plot.contourf(ax=ax2, cmap='RdBu_r', levels=21)
        uz.plot.contour(ax=ax2, colors='k', levels=15)
        ax2.set_ylabel('Pseudo-Altitude (km)')
    else:
        eof2.plot.contourf(ax=ax2, cmap='RdBu_r', yincrease=False, levels=21)
        uz.plot.contour(ax=ax2, colors='k', yincrease=False, levels=15)
    ax2.set_title(r'EOF2, {0:.0f}% of variance, $\tau$ = {1:.0f} days'\
        .format(100*variance_fractions.values[1],\
        tau2))

    ax3 = fig.add_subplot(1,3,3)
    ax3.acorr(pcs.sel(mode=0), maxlags=lags, usevlines=False, linestyle="-", marker=None, linewidth=1, label='EOF1')
    ax3.acorr(pcs.sel(mode=1), maxlags=lags, usevlines=False, linestyle="-", marker=None, linewidth=1, label='EOF2')
    ax3.legend()
    ax3.axhline(0, color='k', linewidth=1)
    ax3.axhline(1/np.e, color='k', linestyle=":")
    ax3.set_xlabel('Lag (days)')
    ax3.set_title('PC autocorrelation')

    plt.tight_layout()
    return plt.show()

def plot_multi(ds1, ds2, ds3, ds4, p_min, lat_min):
    # Plot equivalent to Fig 4a-c from Sheshadri & Plumb
    # First generate all necessary information for EOF analysis
    pc1 = leading_pcs(eof_solver(ds1, p_min, lat_min))
    pc2 = leading_pcs(eof_solver(ds2, p_min, lat_min))
    pc3 = leading_pcs(eof_solver(ds3, p_min, lat_min))
    pc4 = leading_pcs(eof_solver(ds4, p_min, lat_min))
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
    plot_type = input("a) single or b) multi?")

    time = 'daily'
    years = 0 # user sets no. of years worth of data to ignore due to spin-up
    file_suffix = '_interp'    

    # For EOFs follow Sheshadri & Plumb 2017, use p>100hPa, lat>20degN, zonal mean u
    p_min = 100  # hPa
    lat_min = 20  # degrees

    if plot_type=='a':
        exp_name = 'PK_eps0_vtx2_zoz18_7y'
        ds = add_phalf(exp_name, time, years
        plot_single(ds, p_min, lat_min, exp_name, alt=False)
    
    elif plot_type=='b':
        exp = ['PK_eps0_vtx1_zoz18_7y', 'PK_eps0_vtx2_zoz18_7y', 'PK_eps0_vtx3_zoz18_7y', 'PK_eps0_vtx4_zoz18_7y']
    
        ds1 = add_phalf(exp[0], time, file_suffix, years)
        ds2 = add_phalf(exp[1], time, file_suffix, years)
        ds3 = add_phalf(exp[2], time, file_suffix, years)
        ds4 = add_phalf(exp[3], time, file_suffix, years)

        plot_multi(ds1, ds2, ds3, ds4, p_min, lat_min)        


"""
# Plot the PC timeseries
t = ds.coords['time']
times=[]
for i in range(len(t)):
    times.append(t[i]-t[0])
fig2 = plt.figure(figsize=(6,5))
plt.plot(times, pcs[0, 0], linewidth=2)
ax = plt.gca()
ax.axhline(0, color='k')
ax.set_xlim(0, max(times))
ax.set_xlabel('Days')
ax.set_ylabel('Normalized Units')
ax.set_title('PC1 Time Series', fontsize='x-large')
 plt.show()
"""
