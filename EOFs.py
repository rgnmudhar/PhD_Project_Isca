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

def eof_u(uz, p_min, lat_min):
    """
    Follows Sheshadri & Plumb 2017, uses p>100hPa, lat>20degN and zonal mean zonal wind.
    """
    u_new = uz.sel(pfull=slice(p_min,1000)).sel(lat=slice(lat_min,90))

    return u_new

def eof_solver(uz, p_min, lat_min):
    """
    Finds the zonal wind anomaly and latitude/pressure weightings.
    Returns the 'solver' for further EOF analyses.
    """
    u = eof_u(uz, p_min, lat_min)

    # Calculate anomalies
    u_anom = u - u.mean(dim='time')

    # sqrt(cos(lat)) weights due to different box sizes over grid
    sqrtcoslat = np.sqrt(np.cos(np.deg2rad(u_anom.coords['lat'].values))) 

    # sqrt(dp) weights, select correct number of levels
    nplevs = u_anom.coords['pfull'].shape[0]
    sqrtdp = np.sqrt(np.diff(ds.coords['phalf'].values[-nplevs-2:-1]))

    # Calculate gridpoint weights
    wgts = np.outer(sqrtdp,sqrtcoslat)

    # Create an EOF solver to do the EOF analysis.
    solver = Eof(u_anom.compute(), weights=wgts)

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

def plot_single(uz, utz, p_min, lat_min, exp_name):
    """
    Plots equivalent to Fig 4a-c from Sheshadri & Plumb.
    """
    # First generate all necessary information for EOF analysis
    u = eof_u(uz, p_min, lat_min)
    solver = eof_solver(uz, p_min, lat_min)
    eofs = leading_eofs(solver)
    pcs = leading_pcs(solver)
    variance_fractions = variance(solver)
    lags = 50 
    tau1, tau2 = AM_times(pcs, lags)

    # Set-up coordinates
    lat = uz.coords['lat'].sel(lat=slice(lat_min,90))
    p = uz.coords['pfull'].sel(pfull=slice(p_min,1000))

    # Set-up variables
    eof1 = eofs.sel(mode=0)
    eof2 = eofs.sel(mode=1)
    
    # Now plot on a single figure
    fig = plt.figure(figsize=(19,6))
    ulvls = np.arange(-200, 200, 5)
    #fig.suptitle(exp_name, fontsize='large')

    ax1 = fig.add_subplot(1,3,1)
    eof_1 = eof1.plot.contourf(ax=ax1, cmap='RdBu_r', yincrease=False, levels=21, add_colorbar=False)
    plt.colorbar(eof_1)
    utz.plot.contour(ax=ax1, colors='k', yincrease=False, levels=ulvls)
    ax1.set_xlim(lat_min,90)
    ax1.set_ylabel('Pressure (hPa)', fontsize='x-large')
    ax1.set_ylim(925,100)

    ax1.set_xlabel(r'Latitude ($\degree$N)', fontsize='x-large')
    ax1.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
    ax1.set_title(r'EOF1, {0:.0f}% of variance, $\tau$ = {1:.0f} days'\
        .format(100*variance_fractions.values[0],\
        tau1), fontsize='x-large')
    
    ax2 = fig.add_subplot(1,3,2)
    eof_2 = eof2.plot.contourf(ax=ax2, cmap='RdBu_r', yincrease=False, levels=21, add_colorbar=False)
    plt.colorbar(eof_2)
    utz.plot.contour(ax=ax2, colors='k', yincrease=False, levels=ulvls)
    ax2.set_xlim(lat_min,90)
    ax2.set_ylabel('Pressure (hPa)', fontsize='x-large')
    ax2.set_ylim(925,100)
    
    ax2.set_xlabel(r'Latitude ($\degree$N)', fontsize='x-large')
    ax2.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
    ax2.set_title(r'EOF2, {0:.0f}% of variance, $\tau$ = {1:.0f} days'\
        .format(100*variance_fractions.values[1],\
        tau2), fontsize='x-large')

    ax3 = fig.add_subplot(1,3,3)
    ax3.acorr(pcs.sel(mode=0), maxlags=lags, usevlines=False, color='#C0392B', linestyle="-", marker=None, linewidth=1, label='EOF1')
    ax3.acorr(pcs.sel(mode=1), maxlags=lags, usevlines=False, color='#2980B9', linestyle="-", marker=None, linewidth=1, label='EOF2')
    ax3.legend(fancybox=False, shadow=True, ncol=1, fontsize='large')
    ax3.axhline(0, color='k', linewidth=0.5)
    ax3.axvline(0, color='k', linewidth=0.5)
    ax3.axhline(1/np.e, color='#D2D0D3', linestyle=":")
    ax3.set_xlim(-1*lags, lags)
    ax3.set_xlabel('Lag (days)', fontsize='x-large')
    ax3.set_ylim(-0.2,1)
    ax3.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
    ax3.set_title('PC autocorrelation', fontsize='x-large')
    plt.savefig(exp_name+'_eofs.pdf', bbox_inches = 'tight')

    return plt.close()

def plot_multi(uz, lags, p_min, lat_min, labels, colors, style, cols, fig_name):
    """
    Plots equivalent to Fig 4c from Sheshadri & Plumb to compare experiments.
    """
    # Plot on a single figure
    fig = plt.figure(figsize=(10,8))
    for i in range(len(uz)):
        pc = leading_pcs(eof_solver(uz[i], p_min, lat_min))
        plt.acorr(pc.sel(mode=0), maxlags=lags, usevlines=False, color=colors[i], linestyle=style[i], marker=None, linewidth=1, label=labels[i])
    plt.legend(loc='upper center' , bbox_to_anchor=(0.5, -0.07), fancybox=False, shadow=True, ncol=cols, fontsize='large')
    plt.axhline(0, color='k', linewidth=0.5)
    plt.axvline(0, color='k', linewidth=0.5)
    plt.axhline(1/np.e, color='#D2D0D3', linestyle=":")
    plt.xlim(-1*lags, lags)
    plt.xlabel('Lag (days)', fontsize='large')
    plt.ylim(-0.2,1)
    plt.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
    plt.title('EOF1 PC autocorrelation', fontsize='x-large')
    plt.savefig(fig_name+'_pcs.pdf', bbox_inches = 'tight')

    return plt.close()

if __name__ == '__main__': 
    plot_type = input("a) single or b) multi?")

    indir = '/disco/share/rm811/processed/'
    basis = 'PK_e0v4z13'
    exp = ['test', 'test'] #[basis, basis+'_q6m2y45l800u200']

    # For EOFs follow Sheshadri & Plumb 2017, use p>100hPa, lat>20degN
    p_min = 100  # hPa
    lat_min = 20  # degrees

    if plot_type=='a':
        ds = add_phalf(indir+exp[0], '_zmean.nc')
        uz = ds.ucomp
        utz = xr.open_dataset(indir+exp[0]+'_tzmean.nc', decode_times=False).ucomp[0]
        plot_single(uz, utz, p_min, lat_min, exp[0])
    
    elif plot_type=='b':
        uz = []
        for i in range(len(exp)):
            ds = add_phalf(indir+exp[i], '_zmean.nc')
            uz.append(ds.ucomp)

        labels = ['label1', 'label2']
        colors = ['#B30000', '#FF9900']
        style = ['-', ':']
        cols = len(exp)

        plot_multi(uz, 75, p_min, lat_min, labels, colors, style, cols, exp[0])    
