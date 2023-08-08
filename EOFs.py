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
from matplotlib.lines import Line2D
from eofs.xarray import Eof
import statsmodels.api as sm
from shared_functions import *
from datetime import datetime

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

def return_tau(uz):
    """
    Finds leading AM timescale.
    """
    # For EOFs follow Sheshadri & Plumb 2017, use p>100hPa, lat>20degN
    p_min = 100  # hPa
    lat_min = 20  # degrees

    # First generate all necessary information for EOF analysis
    u = eof_u(uz, p_min, lat_min)
    solver = eof_solver(uz, p_min, lat_min)
    eofs = leading_eofs(solver)
    pcs = leading_pcs(solver)
    variance_fractions = variance(solver)
    lags = 200 
    tau1, tau2 = AM_times(pcs, lags)

    return tau1

def plot_single(uz, utz, p_min, lat_min, exp_name):
    """
    Plots equivalent to Fig 4a-c from Sheshadri & Plumb.
    """
    # First generate all necessary information for EOF analysis
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
    #Set-up data to be read in
    indir = '/disco/share/rm811/processed/'
    basis = 'PK_e0v4z13'
    var_type = 0 #input("Plot a) depth, b) width, c) location, d) strength, e) vortex experiments or f) test?")
    if var_type == 'a':
        extension = '_depth'
    elif var_type == 'b':
        extension = '_width'
    elif var_type == 'c':
        extension = '_loc'
    elif var_type == 'd':
        extension = '_strength'
    elif var_type == 'e':
        basis = 'PK_e0vXz13'
        extension = '_vtx'
    elif var_type == 'f':
        extension = '_test'
    exp = ['test', 'test2', 'test3'] #return_exp(extension)[0]
    
    # For EOFs follow Sheshadri & Plumb 2017, use p>100hPa, lat>20degN
    p_min = 100  # hPa
    lat_min = 20  # degrees

    plot_type = input("a) individual experiment, b) multiple PCs, or c) plot for report?")

    if plot_type=='a':
        for i in range(len(exp)):
            print(datetime.now(), " - opening files ({0:.0f}/{1:.0f})".format(i+1, len(exp)))
            ds = add_phalf(indir+exp[i], '_uz.nc')
            uz = ds.ucomp
            utz = xr.open_dataset(indir+exp[i]+'_utz.nc', decode_times=False).ucomp[0]
            plot_single(uz, utz, p_min, lat_min, exp[i])
    
    elif plot_type=='b':
        uz = []
        for i in range(len(exp)):
            ds = add_phalf(indir+exp[i], '_uz.nc')
            uz.append(ds.ucomp)

        labels = ['label1', 'label2']
        colors = ['#B30000', '#FF9900']
        style = ['-', ':']
        cols = len(exp)

        plot_multi(uz, 75, p_min, lat_min, labels, colors, style, cols, exp[0])
    
    elif plot_type =='c':
        # For creating a plot that shows SPV speed and AM timescale for various experiments
        exp = ['PK_e0v1z13', 'PK_e0v2z13', 'PK_e0v3z13', 'PK_e0v4z13',\
        'PK_e0v1z18', 'PK_e0v2z18', 'PK_e0v3z18', 'PK_e0v4z18',\
        'PK_e0v4z18_h3000m2l25u65']
        symbols =  ['o', 's', '*']
        colors = ['k', '#00B300', '#0099CC', '#B30000']
        labels = [r'$p_{oz} \sim 200$ hPa', r'$p_{oz} \sim 100$ hPa', '+ asymmetry', r'$\gamma = 1$ K km$^{-1}$', r'$\gamma = 2$ K km$^{-1}$',r'$\gamma = 3$ K km$^{-1}$', r'$\gamma = 4$ K km$^{-1}$']
        legend_elements = [Line2D([0], [0], marker=symbols[0], color='w', label=labels[0], markerfacecolor='k', markersize=10),\
                    Line2D([0], [0], marker=symbols[1], color='w', label=labels[1], markerfacecolor='k', markersize=10),\
                    Line2D([0], [0], marker=symbols[2], color='w', label=labels[2], markerfacecolor='#4D0099', markersize=15),\
                    Line2D([0], [0], color=colors[0], lw=5, label=labels[3]),\
                    Line2D([0], [0], color=colors[1], lw=5, label=labels[4]),\
                    Line2D([0], [0], color=colors[2], lw=5, label=labels[5]),\
                    Line2D([0], [0], color=colors[3], lw=5, label=labels[6])]
    
        print(datetime.now(), " - finding SPV and tau values")
        tau = []
        vtx = []
        for i in range(len(exp)):
            ds = add_phalf(indir+exp[i], '_uz.nc')
            uz = ds.ucomp
            tau.append(return_tau(uz))
            SPV = open_file('../Files/', exp[i], 'u10')
            vtx.append(np.mean(SPV))
        
        print(datetime.now(), " - plotting")
        fig, ax = plt.subplots(figsize=(6,6))
        a = ax.axhline(25, linestyle='--', linewidth=1, color='k')
        b = ax.axhline(50, linestyle='--', linewidth=1, color='k')
        ax.axvline(10, linestyle='--', linewidth=1, color='k')
        ax.axvline(30, linestyle='--', linewidth=1, color='k')
        ax.scatter(tau[:4], vtx[:4], s=50, c=colors, marker=symbols[0])
        ax.scatter(tau[4:8], vtx[4:8], s=50, c=colors, marker=symbols[1])
        ax.scatter(tau[-1], vtx[-1], s=75, c='#4D0099', marker=symbols[2])
        ax.set_xlabel(r'EOF1 $\tau$ (days)', fontsize='x-large')
        ax.set_ylabel(r'10 hPa, 60 N Zonal Wind Mean (ms$^{-1}$)', fontsize='x-large')
        ax.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
        ax.legend(handles=legend_elements, fontsize='large')
        sq = plt.Rectangle((10,25), 20, 25, fc='#00B300', alpha=0.2)
        plt.gca().add_patch(sq)
        plt.savefig('vtx_vs_tau_v4h3000z18.pdf', bbox_inches = 'tight')
        plt.close()
