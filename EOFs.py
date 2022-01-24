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
    """
    Follows Sheshadri & Plumb 2017, uses p>100hPa, lat>20degN and zonal mean zonal wind.
    """
    uz = ds.ucomp\
        .sel(pfull=slice(p_min,1000))\
        .sel(lat=slice(lat_min,90))\
        .mean(dim='lon')
    return uz

def eof_solver(ds, p_min, lat_min):
    """
    Finds the zonal wind anomaly and latitude/pressure weightings.
    Returns the 'solver' for further EOF analyses.
    """
    uz = eof_u(ds, p_min, lat_min)

    # Calculate anomalies
    uz_anom = uz - uz.mean(dim='time')

    # sqrt(cos(lat)) weights due to different box sizes over grid
    sqrtcoslat = np.sqrt(np.cos(np.deg2rad(uz_anom.coords['lat'].values))) 

    # sqrt(dp) weights, select correct number of levels
    nplevs = uz_anom.coords['pfull'].shape[0]
    sqrtdp = np.sqrt(np.diff(ds.coords['phalf'].values[-nplevs-2:-1]))

    # Calculate gridpoint weights
    wgts = np.outer(sqrtdp,sqrtcoslat)

    # Create an EOF solver to do the EOF analysis.
    solver = Eof(uz_anom.compute(), weights=wgts)

    return solver

def leading_eofs(solver):
    """
    Retrieve the leading 2 EOFs expressed  as the covariance between the leading PC time series and the input anomalies
    """
    eofs = solver.eofsAsCovariance(neofs=2)

    return eofs

def leading_pcs(solver):
    """
    Retrieve the leading PC time series.
    By default PCs used are scaled to unit variance (divided by square root of the eigenvalue).
    """
    pcs = solver.pcs(npcs=2, pcscaling=1)

    return pcs

def variance(solver):
    """
    Fractional EOF mode variances.
    The fraction of the total variance explained by each EOF mode, values between 0 and 1 inclusive.
    """

    return solver.varianceFraction()

def findtau(ac):
    """
    Finds the time for correlation reduce to 1/e.
    """
    for i in range(len(ac)):
        if ac[i] - 1/np.e < 0:
            tau = i
            break

    return tau

def AM_times(pcs, lags):
    """
    Finds autocorrelation function in order to determine decorrelation time (tau)
    """
    ac1 = sm.tsa.acf(pcs.sel(mode=0).values, nlags=lags)
    ac2 = sm.tsa.acf(pcs.sel(mode=1).values, nlags=lags)

    tau1 = findtau(ac1)
    tau2 = findtau(ac2)

    return tau1, tau2

def plot_single(ds, p_min, lat_min, exp_name, alt=True):
    """
    Plots equivalent to Fig 4a-c from Sheshadri & Plumb.
    """
    # First generate all necessary information for EOF analysis
    uz = eof_u(ds, p_min, lat_min)
    solver = eof_solver(ds, p_min, lat_min)
    eofs = leading_eofs(solver)
    pcs = leading_pcs(solver)
    variance_fractions = variance(solver)
    lags = 50 
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
    fig = plt.figure(figsize=(18,6))
    #fig.suptitle(exp_name, fontsize='large')

    ax1 = fig.add_subplot(1,3,1)
    if alt==True:
        eof = eof1.plot.contourf(ax=ax1, cmap='RdBu_r', levels=21, add_colorbar=False)
        uz.plot.contour(ax=ax1, colors='k', levels=15)
        ax1.set_ylabel('Pseudo-Altitude (km)', fontsize='large')
    else:
        eof = eof1.plot.contourf(ax=ax1, cmap='RdBu_r', yincrease=False, levels=21, add_colorbar=False)
        uz.plot.contour(ax=ax1, colors='k', yincrease=False, levels=15)
        ax1.set_ylabel('Pressure (hPa)', fontsize='large')
        ax1.set_ylim(925,100)

    ax1.set_xlabel(r'Latitude ($\degree$N)', fontsize='large')
    ax1.tick_params(axis='both', labelsize = 'medium', which='both', direction='in')
    ax1.set_title(r'EOF1, {0:.0f}% of variance, $\tau$ = {1:.0f} days'\
        .format(100*variance_fractions.values[0],\
        tau1), fontsize='x-large')
    

    ax2 = fig.add_subplot(1,3,2)
    if alt==True:
        eof2.plot.contourf(ax=ax2, cmap='RdBu_r', levels=21, add_colorbar=False)
        uz.plot.contour(ax=ax2, colors='k', levels=15)
        ax2.set_ylabel('Pseudo-Altitude (km)', fontsize='large')
    else:
        eof2.plot.contourf(ax=ax2, cmap='RdBu_r', yincrease=False, levels=21, add_colorbar=False)
        uz.plot.contour(ax=ax2, colors='k', yincrease=False, levels=15)
        ax2.set_ylabel('Pressure (hPa)', fontsize='large')
        ax2.set_ylim(925,100)
    
    ax2.set_xlabel(r'Latitude ($\degree$N)', fontsize='large')
    ax2.tick_params(axis='both', labelsize = 'medium', which='both', direction='in')
    ax2.set_title(r'EOF2, {0:.0f}% of variance, $\tau$ = {1:.0f} days'\
        .format(100*variance_fractions.values[1],\
        tau2), fontsize='x-large')

    ax3 = fig.add_subplot(1,3,3)
    ax3.acorr(pcs.sel(mode=0), maxlags=lags, usevlines=False, color='#C0392B', linestyle="-", marker=None, linewidth=1, label='EOF1')
    ax3.acorr(pcs.sel(mode=1), maxlags=lags, usevlines=False, color='#27AE60', linestyle="-", marker=None, linewidth=1, label='EOF2')
    ax3.legend(fancybox=False, shadow=True, ncol=1, fontsize='large')
    ax3.axhline(0, color='k', linewidth=0.5)
    ax3.axvline(0, color='k', linewidth=0.5)
    ax3.axhline(1/np.e, color='#D2D0D3', linestyle=":")
    ax3.set_xlim(-1*lags, lags)
    ax3.set_xlabel('Lag (days)', fontsize='large')
    ax3.set_ylim(-0.2,1)
    ax3.tick_params(axis='both', labelsize = 'medium', which='both', direction='in')
    ax3.set_title('PC autocorrelation', fontsize='x-large')


    plt.savefig(exp_name+'_eofs.png', bbox_inches = 'tight')
    plt.close()

    return plt.show()

def plot_multi(ds1, ds2, ds3, ds4, p_min, lat_min, labels, fig_name):
    # Plot equivalent to Fig 4a-c from Sheshadri & Plumb
    # First generate all necessary information for EOF analysis
    pc1 = leading_pcs(eof_solver(ds1, p_min, lat_min))
    pc2 = leading_pcs(eof_solver(ds2, p_min, lat_min))
    pc3 = leading_pcs(eof_solver(ds3, p_min, lat_min))
    pc4 = leading_pcs(eof_solver(ds4, p_min, lat_min))
    lags = 190 
    
    # Now plot on a single figure
    fig = plt.figure(figsize=(10,8))
    plt.acorr(pc1.sel(mode=0), maxlags=lags, usevlines=False, color='#2980B9', linestyle="--", marker=None, linewidth=1, label=labels[0])
    plt.acorr(pc2.sel(mode=0), maxlags=lags, usevlines=False, color='#2980B9', linestyle="-", marker=None, linewidth=1, label=labels[1])
    plt.acorr(pc3.sel(mode=0), maxlags=lags, usevlines=False, color='k', linestyle="--", marker=None, linewidth=1, label=labels[2])
    plt.acorr(pc4.sel(mode=0), maxlags=lags, usevlines=False, color='k', linestyle="-", marker=None, linewidth=1, label=labels[3])
    plt.legend(loc='upper center' , bbox_to_anchor=(0.5, -0.07), fancybox=False, shadow=True, ncol=2, fontsize='large')
    plt.axhline(0, color='k', linewidth=0.5)
    plt.axvline(0, color='k', linewidth=0.5)
    plt.axhline(1/np.e, color='#D2D0D3', linestyle=":")
    plt.xlim(-1*lags, lags)
    plt.xlabel('Lag (days)', fontsize='large')
    plt.ylim(-0.4,1)
    plt.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
    plt.title('EOF1 PC autocorrelation', fontsize='x-large')
    plt.savefig(fig_name+'_pcs.png', bbox_inches = 'tight')
    plt.close()

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
        exp_name = 'PK_eps10_vtx4_zoz18_7y'
        ds = add_phalf(exp_name, time, file_suffix, years)
        plot_single(ds, p_min, lat_min, exp_name, alt=False)
    
    elif plot_type=='b':
        exp = ['PK_eps0_vtx4_zoz18_7y', 'PK_eps10_vtx4_zoz18_7y', 'PK_eps0_vtx4_zoz13_7y', 'PK_eps10_vtx4_zoz13_7y']
    
        ds1 = add_phalf(exp[0], time, file_suffix, years)
        ds2 = add_phalf(exp[1], time, file_suffix, years)
        ds3 = add_phalf(exp[2], time, file_suffix, years)
        ds4 = add_phalf(exp[3], time, file_suffix, years)

        #labels = [r'$\gamma$ = 1',r'$\gamma$ = 2',r'$\gamma$ = 3',r'$\gamma$ = 4']
        labels = [r'$\epsilon = 0, p_{trop} \sim 100$ hPa', r'$\epsilon = 10, p_{trop} \sim 100$ hPa', r'$\epsilon = 0, p_{trop} \sim 200$ hPa', r'$\epsilon = 10, p_{trop} \sim 200$ hPa']

        plot_multi(ds1, ds2, ds3, ds4, p_min, lat_min, labels, 'PK_eps0+10_zoz13+10_vtx4')        


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
