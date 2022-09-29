"""
Script for functions involving winds - near-surface, tropospheric jet and stratospheric polar vortex.
"""

from glob import glob
import xarray as xr
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import datetime
from open_winds import *
from shared_functions import *

def plot_winds(indir, exp, labels, colors, style, cols, name, p):
    """
    Plots time and zonal average zonal wind at the pressure level closest to XhPa
    """

    print(datetime.now(), " - plotting winds")
    #Following plots the data and saves as a figure
    fig = plt.subplots(1,1, figsize=(10,6))
    plt.axhline(0, color='#D2D0D3', linewidth=0.5)
    for i in range(len(exp)):
        uz = xr.open_dataset(indir+exp[i]+'_utz.nc', decode_times=False).ucomp[0]
        lat = uz.lat
        plt.plot(uz.lat, uz.sel(pfull=p, method='nearest'), color=colors[i], linestyle=style[i], label=labels[i])   
    plt.xlabel('Latitude', fontsize='large')
    plt.xlim(-90,90)
    plt.xticks([-90, -45, 0, 45, 90], ['90S', '45S', '0', '45N', '90N'])
    plt.ylabel(r'Wind Speed (ms$^{-1}$)', fontsize='large')
    plt.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
    plt.legend(loc='upper center' , bbox_to_anchor=(0.5, -0.07),fancybox=False, shadow=True, ncol=cols, fontsize='large')
    plt.savefig(name+'_winds.pdf', bbox_inches = 'tight')
    return plt.close()

def plot_jet(u, p, lvls, name):
    """
    Plots time-mean zonal wind on polar stereographic lat-lon grid
    """
    print(datetime.now(), " - plotting winds")
    #Following plots the data and saves as a figure
    u_p = u.sel(pfull=p, method='nearest')
    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    cs = ax.contourf(u.lon, u.lat, u_p,\
        cmap='RdBu_r', levels=lvls, transform = ccrs.PlateCarree())
    cb = plt.colorbar(cs, pad=0.1, extend='both')
    cb.set_label(label=r'Mean Zonal Wind (m s$^{-1}$)', size='x-large')
    cb.ax.tick_params(labelsize='x-large')
    ax.set_global()
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    ax.set_extent([-180, 180, 0, 90], crs=ccrs.PlateCarree())
    plt.savefig(name+'_jet{:.0f}.pdf'.format(p), bbox_inches = 'tight')
    return plt.close()

def plot_vtx(dir, exp, labels, colors, style, cols, fig_name):
    """
    Plots strength of winds at 60N, 10 hPa only.
    Best for 2 datsets, the second of which has its SSW statistics as a plot subtitle
    """
    
    print(datetime.now(), " - plotting SPV")
    fig, ax = plt.subplots(figsize=(8,6))
    for i in range(len(exp)):
        SPV = open_file(dir, exp[i], 'SPV')
        ax.plot(SPV, color=colors[i], linewidth=1, linestyle=style[i], label=labels[i])
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlim(1,len(SPV))
    ax.set_xlabel('Months Simulated', fontsize='x-large')       
    ax.set_ylabel(r'10 hPa, 60 N Zonal Wind Mean (ms$^{-1}$)', color='k', fontsize='x-large')
    ax.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    ax.set_xticks([10*(12*30), 20*(12*30), 30*(12*30), 40*(12*30)], [10*12, 20*12, 30*12, 40*12])
    plt.legend(loc='upper center' , bbox_to_anchor=(0.5, 0.11), fancybox=False, shadow=False, ncol=cols, fontsize='x-large')
    plt.savefig(fig_name+'_vtx.pdf', bbox_inches = 'tight')
    
    return plt.close()

def SPVvexp1(mean, mode, sd, err, p, labels, xlabel, name):
    """
    Plots the mean and standard deviation of SPV against (heating) experiment.
    """
    print(datetime.now(), " - plotting average and s.d. vs experiment at {:.0f} hPa".format(p))
    fig, ax = plt.subplots(figsize=(10,6))
    ax.errorbar(labels[1:], mean[1:], yerr=err[1:], fmt='o', linewidth=1.25, capsize=5, color='#B30000', linestyle=':', label='mean')
    ax.plot(labels[1:], mode[1:], marker='o', linewidth=1.25, color='#B30000', linestyle='-.', label='mode')
    plt.legend(loc='upper center' , bbox_to_anchor=(0.5, 1), fancybox=False, shadow=False, ncol=2, fontsize='xx-large')
    ax.set_xticks(labels)
    ax.set_xlabel(xlabel, fontsize='xx-large')
    ax.set_ylabel(r'zonal wind average (m s$^{-1}$)', fontsize='xx-large', color='#B30000')
    #ax.set_ylim(36,42)
    ax.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
    ax2 = ax.twinx()
    ax2.plot(labels[1:], sd[1:], marker='o', linewidth=1.25, color='#4D0099', linestyle=':', label='S.D.')
    #ax2.axhline(sd[0], color='#4D0099', linewidth=0.5)
    #ax2.text(5.6, sd[0]-0.6, 'Control', color='#4D0099', fontsize='x-large')
    ax2.set_ylabel(r'zonal wind standard deviation (m s$^{-1}$)', color='#4D0099', fontsize='xx-large')
    #ax2.set_ylim(12,22)
    ax2.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
    plt.savefig(name+'_{:.0f}stats1.pdf'.format(p), bbox_inches = 'tight')
    return plt.close()

def SPVvexp2(skew, kurt, p, labels, xlabel, name):
    """
    Plots the skew and kurtosis of zonal winds at p hPa against (heating) experiment.
    """
    print(datetime.now(), " - plotting skew and kurtosis vs experiment at {:.0f} hPa".format(p))
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(labels[1:], skew[1:], marker='o', linewidth=1.25, color='#B30000', linestyle=':')
    ax.set_xticks(labels)
    ax.set_xlabel(xlabel, fontsize='x-large')
    ax.set_ylabel('{:.0f} hPa,'.format(p)+r' 60$\degree$N zonal wind skewness', fontsize='xx-large', color='#B30000')
    ax.axhline(skew[0], color='#B30000', linewidth=0.5)
    ax.text(-0.25, skew[0]+0.05, 'control', color='#B30000', fontsize='xx-large')
    ax.set_ylim(-0.5, 0.5)
    ax.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
    ax2 = ax.twinx()
    ax2.plot(labels[1:], kurt[1:], marker='o', linewidth=1.25, color='#4D0099', linestyle=':')
    ax2.set_ylabel('{:.0f} hPa,'.format(p)+r' 60$\degree$N zonal wind kurtosis', color='#4D0099', fontsize='xx-large')
    ax2.axhline(kurt[0], color='#4D0099', linewidth=0.5)
    ax2.text(5.75, kurt[0]+0.05, 'control', color='#4D0099', fontsize='xx-large')
    ax2.set_xlim(-0.5, 6.5)
    ax2.set_ylim(-0.9, 0.9)
    ax2.fill_between(range(-1,8), -1, 0, facecolor ='gainsboro', alpha = 0.4)
    ax2.text(1, -0.75, 'Negative Skew, Lighter Tails', color='#666666', fontsize='xx-large')
    ax2.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
    plt.savefig(name+'_{:.0f}stats2.pdf'.format(p), bbox_inches = 'tight')
    return plt.close()

def windsvexp(dir, labels, xlabel, p, name):
    """
    Uses jet_locator functions to find location and strength of maximum stratopsheric vortex (10 hPa).
    Then plots this against (heating) experiment.
    """
    print(datetime.now(), " - plotting wind maxima vs experiment")
    fig, ax = plt.subplots(figsize=(10,6))
    ax.errorbar(labels[1:], open_file(dir, name, 'maxwinds'+p)[1:], yerr=open_file(dir, name, 'maxwinds_sd'+p)[1:], fmt='o', linewidth=1.25, capsize=5, color='#B30000', linestyle=':')
    ax.set_xticks(labels)
    ax.set_xlabel(xlabel, fontsize='x-large')
    ax.set_ylabel(r'Strength of Max. 10 hPa Zonal Wind (ms$^{-1}$)', color='#B30000', fontsize='x-large')
    ax.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    ax2 = ax.twinx()
    ax2.errorbar(labels[1:], open_file(dir, name, 'maxlats'+p)[1:], yerr=open_file(dir, name, 'maxlats_sd'+p)[1:], fmt='o', linewidth=1.25, capsize=5, color='#4D0099', linestyle=':')
    ax2.set_ylabel(r'Laitude of Max. 10 hPa Zonal Wind ($\degree$N)', color='#4D0099', fontsize='x-large')
    ax2.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    plt.savefig(name+'_windsvexp'+p+'.pdf', bbox_inches = 'tight')

    return plt.close()

def SSWsvexp(dir, exp, x, xlabel, fig_name):
    """
    Plots SSW frequency against (heating) experiment.
    """
    print(datetime.now(), " - finding SSWs")
    SSWs, errors = find_SSWs(dir, exp)
    og = SSWs[0]
    og_err = errors[0]
    print(SSWs, errors)
    print(datetime.now(), " - plotting SSWs vs experiment")
    fig, ax = plt.subplots(figsize=(10,6))
    ax.errorbar(x[1:], SSWs[1:], yerr=errors[1:], fmt='o', linewidth=1.25, capsize=5, color='#B30000', linestyle=':')
    ax.set_xlim(-0.5,6.5)
    ax.set_xticks(x[1:])
    ax.set_xlabel(xlabel, fontsize='xx-large')
    ax.set_ylabel(r'SSWs per 100 days', fontsize='xx-large')
    ax.axhline(0.42, color='#4D0099', linewidth=0.5)
    ax.text(5.35, 0.42+0.01, 'reanalysis', color='#4D0099', fontsize='xx-large')
    ax.axhline(og, color='#666666', linewidth=0.5)
    ax.fill_between(range(-1,8), (og - og_err), (og + og_err), facecolor ='gainsboro', alpha = 0.4)
    ax.text(5.7, og+0.01, 'control', color='#666666', fontsize='xx-large')
    ax.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
    plt.savefig(fig_name+'_SSWsvheat.pdf', bbox_inches = 'tight')

    return plt.close()

def SSWsvexp_multi(dir, exp, x, xlabel, legend, colors, fig_name):
    """
    Plots SSW frequency against (heating) experiment for 3 sets of experiments
    """
    print(datetime.now(), " - finding SSWs")
    og, og_err = find_SSWs(dir, [exp[0]])
    og = og[0]
    og_err = og_err[0]
    print(datetime.now(), " - plotting SSWs vs experiment")
    fig, ax = plt.subplots(figsize=(10,6))
    for i in np.arange(1,len(exp),1):
        SSWs, errors = find_SSWs(dir, exp[i])
        ax.errorbar(x[1:], SSWs, yerr=errors, fmt='o', linewidth=1.25, capsize=5, color=colors[i-1], linestyle=':', label=legend[i-1])
    ax.set_xlim(-0.25,2.25)
    ax.set_xticks(x[1:])
    ax.set_xlabel(xlabel, fontsize='x-large')
    ax.set_ylim(0,0.58)
    ax.set_ylabel(r'SSWs per 100 days', fontsize='x-large')
    ax.axhline(0.42, color='#4D0099', linewidth=0.5)
    ax.text(1.85, 0.425, 'ERA-Interim', color='#4D0099', fontsize='x-large')
    ax.axhline(og, color='#666666', linewidth=0.5)
    ax.fill_between(range(-1,5), (og - og_err), (og + og_err), facecolor ='gainsboro', alpha = 0.4)
    ax.text(2, 0.335, 'Control', color='#666666', fontsize='x-large')
    ax.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    plt.legend(loc='upper center' , bbox_to_anchor=(0.25, 0.995),fancybox=False, shadow=True, ncol=2, fontsize='large')
    plt.savefig(fig_name+'_SSWsvheat.pdf', bbox_inches = 'tight')

    return plt.close()

if __name__ == '__main__': 
    #Set-up data to be read in
    indir = '/disco/share/rm811/processed/'
    outdir = '../Files/'
    basis = 'PK_e0v4z13'
    var_type = input("Plot a) depth, b) width, c) location, d) strength, or e) topography experiments?")
    if var_type == 'a':
        extension = '_depth'
    elif var_type == 'b':
        extension = '_width'
    elif var_type == 'c':
        extension = '_loc'
    elif var_type == 'd':
        extension = '_strength'
    elif var_type == 'e':
        extension = '_topo'
    exp, labels, xlabel = return_exp(extension)
    n = len(exp)

    #User choices for plotting - subjects
    level = input('Plot a) near-surface winds, \
        b) tropospheric jet(s), \
        c) lower stratosphere, \
        or d) stratospheric polar vortex?')

    colors = ['k', '#B30000', '#FF9900', '#FFCC00', '#00B300', '#0099CC', '#4D0099', '#CC0080']
    reds = ['k', '#fedbcb', '#fcaf94', '#fc8161', '#f44f39', '#d52221', '#aa1016', '#67000d']
    blues = ['k', '#dbe9f6', '#bbd6eb', '#88bedc', '#549ecd',  '#2a7aba', '#0c56a0', '#08306b']
    legend = [r'A = 2 K day$^{-1}$', r'A = 4 K day$^{-1}$', r'A = 6 K day$^{-1}$', r'A = 8 K day$^{-1}$'] 
    #legend = [r'$p_{top} = 800$ hPa', r'$p_{top} = 600$ hPa', r'$p_{top} = 400$ hPa']

    if level == 'a':
        p = 900 #hPa
        style = ['-', ':']
        cols = len(exp)
        plot_winds(outdir, indir, exp, labels, colors, style, cols, exp[0], p)
    elif level == 'b':
        p = [850, 500] #hPa
        lvls = [np.arange(-25, 27.5, 2.5), np.arange(50, 55, 5)]
        #windsvexp(outdir, labels, xlabel, str(p), basis+extension)
        for j in range(len(p)):
            for i in range(n):
                print(datetime.now(), " - finding winds ({0:.0f}/{0:.0f})".format(i+1, n))
                u = xr.open_dataset(indir+exp[i]+'_ut.nc', decode_times=False).ucomp[0]
                plot_jet(u, p[j], lvls[j], exp[i])
    elif level == 'c':
        p = 100
        me, mo, sd, e, sk, k = plot_pdf('u', indir, exp, '_uz.nc', '', [p], labels, r"zonal-mean zonal wind (m s$^{-1}$)", blues, basis+extension)
        SPVvexp1(me[0], mo[0], sd[0], e[0], p, labels, xlabel, basis+extension)
        SPVvexp2(sk[0], k[0], p, labels, xlabel, basis+extension)
    elif level == 'd':
        p = 10 # pressure level at which we want to find the SPV (hPa)
        #User choice for plotting - type
        plot_type = input('Plot a) SPV @ 10hPa, 60N over time, \
            b) 10 hPa max. wind and lat vs experiment, \
            c) distribution and stats, or \
            d) SSW frequency vs experiment, \
            e) s.d. as a function of lat and p?')
        if plot_type == 'a':
            exp = exp[:2]
            labels = ['zonally symmetric', 'off-pole']
            colors = ['#0099CC', '#B30000']
            style = ['-', '-']
            cols = 2
            plot_vtx(outdir, exp, labels, colors, style, cols, exp[0])
        elif plot_type == 'b':
            windsvexp(outdir, labels, xlabel, str(p), basis+extension)
        elif plot_type == 'c':
            me, mo, sd, e, sk, k = plot_pdf('u', indir, exp, '_uz.nc', '', [p], labels, r"zonal-mean zonal wind (m s$^{-1}$)", blues, basis+extension)
            SPVvexp1(me[0], mo[0], sd[0], e[0], p, labels, xlabel, basis+extension)
            SPVvexp2(sk[0], k[0], p, labels, xlabel, basis+extension)
        elif plot_type == 'd':
            SSWsvexp(outdir, exp, labels, xlabel, basis+extension)
            #SSWsvexp_multi(outdir, exp, labels, xlabel, legend, ['#B30000', '#00B300', '#0099CC', 'k'], basis+extension)
        elif plot_type == 'e':
            ulvls = np.arange(-200, 200, 10)
            plot_what = input('Plot a) climatology or b) difference?)')
            for i in range(n):
                print(datetime.now(), " - finding s.d. ({0:.0f}/{1:.0f})".format(i+1, n))
                u = xr.open_dataset(indir+exp[i]+'_uz.nc', decode_times=False).ucomp
                utz = xr.open_dataset(indir+exp[i]+'_utz.nc', decode_times=False).ucomp[0]
                if plot_what == 'a':
                    lat, p, sd = find_sd(u)
                    NH_zonal(lat, p, sd, utz, np.arange(0, 42, 2), ulvls, 'Blues',\
                        r'zonal-mean zonal wind SD (ms$^{-1}$)', exp[i]+'_usd.pdf')
                elif plot_what == 'b':
                    if i == 0:
                        print("skipping control")
                    elif i != 0:
                        lat, p, sd1 = find_sd(u)
                        lat, p, sd2 = find_sd(xr.open_dataset(indir+exp[0]+'_uz.nc', decode_times=False).ucomp)
                        sd_diff = sd1 - sd2
                        NH_zonal(lat, p, sd_diff, utz, np.arange(-20, 22, 2), ulvls, 'RdBu_r',\
                            r'zonal-mean zonal wind SD (ms$^{-1}$)', exp[i]+'_usd_diff.pdf')
