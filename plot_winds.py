"""
Script for functions involving winds - near-surface, tropospheric jet and stratospheric polar vortex.
"""

from glob import glob
import imageio
import os
import xarray as xr
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import matplotlib.colors as cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
import cartopy.crs as ccrs
from datetime import datetime
from open_winds import *
from shared_functions import *

def plot_Xwinds(indir, exp, x, colors, style, cols, name, p):
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

def show_jet(u, p, lvls, name):
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
    fig, ax = plt.subplots(figsize=(10,6))
    for i in range(len(exp)):
        SPV = open_file(dir, exp[i], 'u10')
        ax.plot(SPV, color=colors[i], linewidth=1.5, linestyle=style[i], label=labels[i])
        ax.axhline(np.mean(SPV), color=colors[i], linewidth=1.5, linestyle=':')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlim(1, 2999)
    ax.set_xlabel('Days Simulated', fontsize='xx-large')
    ax.set_ylim(-35, 90) 
    ax.set_ylabel(r'U$_{10,60}$ Mean (ms$^{-1}$)', color='k', fontsize='xx-large')
    ax.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
    #ax.set_xticks([10*(12*30), 20*(12*30), 30*(12*30), 40*(12*30)], [10*12, 20*12, 30*12, 40*12])
    plt.legend(loc='lower center', fancybox=False, shadow=False, ncol=cols, fontsize='x-large')
    plt.savefig(fig_name+'_vtx.pdf', bbox_inches = 'tight')
    return plt.close()

def SPVvexp1(n, mean, mode, sd, err, p, x, xlabel, name):
    """
    Plots the mean and standard deviation of SPV against experiment.
    """
    print(datetime.now(), " - plotting average and s.d. vs experiment at {:.0f} hPa".format(p))
    markers = ['^', 'o']
    lines = ['--', '-', '-.']
    fig, ax = plt.subplots(figsize=(10,6))
    ax2 = ax.twinx()
    if n == 2:
        for i in range(n):
            ax.errorbar(x, mean[i], yerr=err[i], fmt=markers[i], linewidth=1.25, capsize=5, color='#B30000', linestyle=lines[i])
            ax.set_xticks(x)
            ax2.plot(x, sd[i], marker=markers[i], linewidth=1.25, color='#4D0099', linestyle=lines[i], label='S.D.')
            legend_elements = [Line2D([0], [0], marker=markers[0], color='k', label='no polar heat', markerfacecolor='k', markersize=10, linestyle=lines[0]),\
                    Line2D([0], [0], marker=markers[1], color='k', label='polar heat', markerfacecolor='k', markersize=10, linestyle=lines[1])]
            ax.legend(loc='upper left', handles=legend_elements, fontsize='x-large', fancybox=False, ncol=2)
    else:
        ax.errorbar(x[1:], mean[1:], yerr=err[1:], fmt=markers[1], linewidth=1.25, capsize=5, color='#B30000', linestyle=lines[0], label='mean')
        ax.set_xticks(x[1:])
        ax.plot(x[1:], mode[1:], marker=markers[1], linewidth=1.25, color='#B30000', linestyle=lines[2], label='mode')
        legend_elements = [Line2D([0], [0], marker=markers[1], color='#B30000', label='Mean', markerfacecolor='#B30000', markersize=5, linestyle=lines[0]),\
                    Line2D([0], [0], marker=markers[1], color='#B30000', label='Mode', markerfacecolor='#B30000', markersize=5, linestyle=lines[2])]
        ax.legend(loc='upper center', handles=legend_elements, fontsize='x-large', bbox_to_anchor=(0.5, 1), fancybox=False, shadow=False, ncol=2)
        ax2.plot(x[1:], sd[1:], marker=markers[1], linewidth=1.25, color='#4D0099', linestyle=lines[1], label='S.D.')
        #ax2.axhline(sd[0], color='#4D0099', linewidth=0.5)
        #ax2.text(5.6, sd[0]-0.6, 'Control', color='#4D0099', fontsize='x-large') 
    ax.set_xlabel(xlabel, fontsize='xx-large')
    ax.set_ylabel(r'U$_{%s,60}$ Average (m s$^{-1}$)'%(str(p)), fontsize='xx-large', color='#B30000')
    ax.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
    ax2.set_ylabel(r'U$_{%s,60}$ S.D. (m s$^{-1}$)'%(str(p)), color='#4D0099', fontsize='xx-large')
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
    ax.set_xlabel(xlabel, fontsize='xx-large')
    ax.set_ylabel(r'zonal wind skewness', fontsize='xx-large', color='#B30000')
    ax.axhline(skew[0], color='#B30000', linewidth=0.5)
    ax.text(-0.25, skew[0]+0.05, 'control', color='#B30000', fontsize='xx-large')
    ax.set_ylim(-0.5, 0.5)
    ax.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
    ax2 = ax.twinx()
    ax2.plot(labels[1:], kurt[1:], marker='o', linewidth=1.25, color='#4D0099', linestyle=':')
    ax2.set_ylabel('zonal wind kurtosis', color='#4D0099', fontsize='xx-large')
    ax2.axhline(kurt[0], color='#4D0099', linewidth=0.5)
    ax2.text(5.75, kurt[0]+0.05, 'control', color='#4D0099', fontsize='xx-large')
    ax2.set_xlim(-0.5, 6.5)
    ax2.set_ylim(-0.9, 0.9)
    ax2.fill_between(range(-1,8), -1, 0, facecolor ='gainsboro', alpha = 0.4)
    ax2.text(1, -0.75, 'Negative Skew, Lighter Tails', color='#666666', fontsize='xx-large')
    ax2.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
    plt.savefig(name+'_{:.0f}stats2.pdf'.format(p), bbox_inches = 'tight')
    return plt.close()

def GP09(exp):
    """
    As per Gerber & Polvani (2009) plot vtx_gamma vs. U_1060 and EDJ latitude
    """
    vals = np.arange(1,7,1)

    print(datetime.now(), " - finding values")
    edj_lats = []
    vtx = []
    for e in exp:
        edj_lats.append(find_EDJ('/disco/share/rm811/processed/', e)[1])
        vtx.append(np.mean(open_file('../Files/', e, 'u10')))
    
    print(datetime.now(), " - plotting")
    markers = ['^', 'o']
    lines = ['--', '-']
    fig, ax = plt.subplots(figsize=(6,6))
    ax2 = ax.twinx()
    ax.plot(vals, vtx, marker=markers[0], linewidth=1.25, color='#B30000', linestyle=lines[0])
    ax2.plot(vals, edj_lats, marker=markers[1], linewidth=1.25, color='#4D0099', linestyle=lines[1])
    ax.set_xlabel(r'Polar Vortex Lapse Rate, $\gamma$ (K km$^{-1}$)', fontsize='xx-large')
    ax.set_ylabel(r'U$_{10,60}$ Average (m s$^{-1}$)', fontsize='xx-large', color='#B30000')
    ax.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
    ax2.set_ylabel(r'Latitude of Max. U$_{850}$ ($\degree$N)', color='#4D0099', fontsize='xx-large')
    ax2.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
    plt.savefig('GP09_check.pdf', bbox_inches = 'tight')
    return plt.close()

def windsvexp(dir, x, xlabel, p, name):
    """
    Uses jet_locator functions to find location and strength of maximum stratopsheric vortex (10 hPa) or EDJ (850 hPa).
    Then plots this against experiment.
    """
    print(datetime.now(), " - plotting wind maxima vs experiment")
    markers = ['^', 'o']
    lines = ['--', '-']
    fig, ax = plt.subplots(figsize=(10,6))
    ax2 = ax.twinx()
    if len(name) == 2:
        for i in range(len(name)):
            ax.errorbar(x, open_file(dir, name[i], 'maxwinds'+p), yerr=open_file(dir, name[i], 'maxwinds_sd'+p),\
                    fmt=markers[i], linewidth=1.25, capsize=5, color='#B30000', linestyle=lines[i])
            ax2.errorbar(x, open_file(dir, name[i], 'maxlats'+p), yerr=open_file(dir, name[i], 'maxlats_sd'+p),\
                         fmt=markers[i], linewidth=1.25, capsize=5, color='#4D0099', linestyle=lines[i])
        ax.set_xticks(x)
        legend_elements = [Line2D([0], [0], marker=markers[0], color='k', label='no polar heat', markerfacecolor='k', markersize=10, linestyle=lines[0]),\
                    Line2D([0], [0], marker=markers[1], color='k', label='polar heat', markerfacecolor='k', markersize=10, linestyle=lines[1])]
        ax.legend(loc='upper center',handles=legend_elements, fontsize='x-large', fancybox=False, ncol=2)
        name = name[i]
    else:
        ax.errorbar(x[1:], open_file(dir, name, 'maxwinds'+p)[1:], yerr=open_file(dir, name, 'maxwinds_sd'+p)[1:],\
                    fmt=markers[1], linewidth=1.25, capsize=5, color='#B30000', linestyle=lines[1])
        ax2.errorbar(x[1:], open_file(dir, name, 'maxlats'+p)[1:], yerr=open_file(dir, name, 'maxlats_sd'+p)[1:], fmt=markers[1],\
                     linewidth=1.25, capsize=5, color='#4D0099', linestyle=lines[1]) 
        ax.set_xticks(x[1:])
    ax.set_xlabel(xlabel, fontsize='xx-large')
    ax.set_ylabel(r'Strength of Max. U$_{850}$ (ms$^{-1}$)', color='#B30000', fontsize='xx-large')
    ax.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
    ax2.set_ylabel(r'Laitude of Max. U$_{850}$ ($\degree$N)', color='#4D0099', fontsize='xx-large')
    ax2.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
    plt.savefig(name+'_windsvexp'+p+'.pdf', bbox_inches = 'tight')
    return plt.close()

def EDJ_loc():
    """
    Track impact of polar heating on EDJ location with and without an SPV
    """
    heat = '_w15a4p600f800g50'
    perturb = '_q6m2y45l800u200'
    combo = heat+perturb
    # Without a polar vortex
    exp_noheat = [['PK_e0v1z13_a0b0p2'+perturb,'PK_e0v1z13_a0b10p2'+perturb, 'PK_e0v1z13_a5b4p1'+perturb, 'PK_e0v1z13_a5b12p1'+perturb, 'PK_e0v1z13_a5b20p1'+perturb],\
                   ['PK_e0v1z13'+perturb, 'PK_e0v2z13'+perturb,'PK_e0v3z13'+perturb, 'PK_e0v4z13'+perturb,'PK_e0v5z13'+perturb, 'PK_e0v6z13'+perturb]]
    exp_heat = [['PK_e0v1z13_a0b0p2'+heat+perturb, 'PK_e0v1z13_a0b10p2'+heat+perturb, 'PK_e0v1z13_a5b4p1'+heat+perturb, 'PK_e0v1z13_a5b12p1'+heat+perturb, 'PK_e0v1z13_a5b20p1'+heat+perturb],\
                ['PK_e0v1z13'+combo, 'PK_e0v2z13'+combo,'PK_e0v3z13'+combo, 'PK_e0v4z13'+combo,'PK_e0v5z13'+combo, 'PK_e0v6z13'+combo]]
    n = len(exp_heat)

    print(datetime.now(), " - finding locations")
    edj_loc_noheat = []
    for i in exp_noheat:
        edj_lats = []
        for j in i:
            edj_lats.append(find_EDJ('/disco/share/rm811/processed/', j)[1])
        edj_loc_noheat.append(edj_lats)

    edj_loc_heat = []
    for i in exp_heat:
        edj_lats = []
        for j in i:
            edj_lats.append(find_EDJ('/disco/share/rm811/processed/', j)[1])
        edj_loc_heat.append(edj_lats)
    
    print(datetime.now(), " - finding response")
    edj_loc_shift = []
    for i in range(n):
        shifts = []
        for j in range(len(edj_loc_heat[i])):
            shifts.append(edj_loc_heat[i][j] - edj_loc_noheat[i][j])
        edj_loc_shift.append(shifts)
    
    print(datetime.now(), " - plotting")
    markers = ['^', 'o']
    colours = ['#B30000', '#4D0099']
    fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(edj_loc_noheat[0], edj_loc_shift[0], marker=markers[0], s=50, color=colours[0], label=r'Modify $T_{eq}$')
    ax.plot(np.unique(edj_loc_noheat[0]), np.poly1d(np.polyfit(edj_loc_noheat[0], edj_loc_shift[0], 1))(np.unique(edj_loc_noheat[0])), color=colours[0], linewidth=1.25, linestyle='--')
    ax.scatter(edj_loc_noheat[1], edj_loc_shift[1], marker=markers[1], s=50, color=colours[1], label=r'Modify $\gamma$')
    ax.plot(np.unique(edj_loc_noheat[1]), np.poly1d(np.polyfit(edj_loc_noheat[1], edj_loc_shift[1], 1))(np.unique(edj_loc_noheat[1])), color=colours[1], linewidth=1.25, linestyle='--')
    ax.set_xlabel(r'Climatological EDJ Location ($\degree$N)', fontsize='xx-large')
    ax.set_ylabel(r'EDJ Location Response ($\degree$N)', fontsize='xx-large')
    ax.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
    ax.legend(loc='lower left', fontsize='xx-large', fancybox=False, ncol=1)
    plt.savefig('EDJ_location_response_v1+perturb.pdf', bbox_inches = 'tight')
    return plt.close()

def SSWsvexp(dir, exp, x, xlabel, fig_name):
    """
    Plots SSW frequency against (heating) experiment.
    """
    print(datetime.now(), " - finding SSWs")
    SSWs, errors = find_SSWs(dir, exp)
    og = SSWs[0]
    og_err = errors[0]
    obs = 0.48
    obs_err = 0.19
    print(datetime.now(), " - plotting SSWs vs experiment")
    fig, ax = plt.subplots(figsize=(10,6))
    ax.errorbar(x[1:], SSWs[1:], yerr=errors[1:], fmt='o', linewidth=1.25, capsize=5, color='#B30000', linestyle=':')
    ax.set_xlim(-0.5,6.5)
    ax.set_xticks(x[1:])
    ax.set_xlabel(xlabel, fontsize='xx-large')
    ax.set_ylabel(r'SSWs per 100 days', fontsize='xx-large')
    ax.axhline(obs, color='#4D0099', linewidth=0.5)
    ax.text(5, obs+0.01, 'observations', color='#4D0099', fontsize='xx-large')
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

def report_vals(exp, label, u, SSW_flag=True):
    print(datetime.now(), " - finding stats")
    print(label+' mean: {0:.2f}'.format(np.mean(u)))
    x, p, mode = pdf(u)
    print(label+' mode: {0:.2f}'.format(mode))
    print(label+' s.d.: {0:.2f}'.format(np.std(u)))
    print(label+' min: {0:.2f}'.format(np.min(u)))
    print(label+' max: {0:.2f}'.format(np.max(u)))

    if SSW_flag == True:
        print(datetime.now(), " - finding SSWs")
        SSWs, SSWs_err = find_SSW(u)
        print(label+' SSWs: {0:.2f} ± {1:.2f}'.format(SSWs, SSWs_err))

def SPV_report_plot(exp, x, xlabel, name):
    """
    Plots SSW frequency and SPV s.d. against experiment.
    """
    n = len(exp)
    obs = 0.48
    obs_err = 0.19
    markers = ['^', 'o']
    lines = ['--', '-']
    fig, ax = plt.subplots(figsize=(10,6))
    if n == 2:
        SSWs_full = []
        errors = []
        for i in range(n):
            SSWs, errs = find_SSWs(outdir, exp[i])
            SSWs_full.append(SSWs)
            errors.append(errs)
        for i in range(n):
            print(datetime.now(), " - plotting SSWs vs experiment")
            ax.errorbar(x, SSWs_full[i], yerr=errors[i], fmt=markers[i], linewidth=1.5, capsize=5, color='#B30000', linestyle=lines[i])
        legend_elements = [Line2D([0], [0], marker=markers[0], color='k', label='no polar heat', markerfacecolor='k', markersize=10, linestyle=lines[0]),\
                    Line2D([0], [0], marker=markers[1], color='k', label='polar heat', markerfacecolor='k', markersize=10, linestyle=lines[1])]
        ax.legend(loc='lower right',handles=legend_elements, fontsize='x-large', fancybox=False, ncol=2)
        ax.set_ylim(min(min(SSWs_full[0])-0.05,0), max(SSWs_full[0])+0.2)
        ax.set_xlim(-0.5,len(exp[0])-0.5)
    else:
        # For polar heating experiments at fixed vortex strength
        print(datetime.now(), " - finding SSWs")
        SSWs, errors = find_SSWs(outdir, exp)
        print(datetime.now(), " - finding s.d.")
        sd = []
        for e in exp:
            u = xr.open_dataset(indir+e+'_uz.nc', decode_times=False).ucomp.sel(lat=60, method='nearest').sel(pfull=10, method='nearest')
            sd.append(np.std(u))
        print(datetime.now(), " - plotting SSWs and SPV s.d. vs experiment")
        ax.errorbar(x[1:], SSWs[1:], yerr=errors[1:], fmt=markers[1], linewidth=1.5, capsize=5, color='#B30000', linestyle=lines[1])
        ax2 = ax.twinx()
        ax2.plot(x[1:], sd[1:], marker=markers[1], linewidth=1.5, color='#4D0099', linestyle=lines[1], label='S.D.')
        ax.set_xticks(x[1:])
        og = SSWs[0]
        og_err = errors[0]
        ax.set_xlim(-0.5,6.5)
        ax.axhline(og, color='#666666', linewidth=1.5, linestyle=':')
        ax.fill_between(range(-1,8), (og - og_err), (og + og_err), facecolor ='gainsboro', alpha = 0.4)
        ax.text(5, og+0.01, 'control', color='#666666', fontsize='x-large')
        ax.set_ylim(min(min(SSWs)-0.05,0), max(SSWs)+0.2)
        ax2.set_ylim(min(sd)-1, max(sd)+1)
        ax2.set_ylabel(r'U$_{10,60}$ S.D. (m s$^{-1}$)', color='#4D0099', fontsize='xx-large')
        ax2.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
    ax.axhline(obs, color='#0c56a0', linewidth=1.5, linestyle=':')
    ax.text(5, obs+0.01, 'observations', color='#0c56a0', fontsize='x-large')
    ax.set_xlabel(xlabel, fontsize='xx-large')
    ax.set_ylabel('SSW Frequency (per 100 days)', fontsize='xx-large', color='#B30000')
    ax.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
    plt.savefig(name+'_SSWs.pdf', bbox_inches = 'tight')
    return plt.close()

def standardise(full, og, pressure, time):
    std = og
    for p in range(len(pressure)):
        sd = np.std(full[p])
        for t in range(len(time)):
            std[p,t] = og[p,t] / sd
    return std

def plot_SSW_comp(x, unit, name, lvls, label):
    norm = cm.TwoSlopeNorm(vmin=min(lvls), vmax=max(lvls), vcenter=0)
    fig, axes = plt.subplots(1, 1, figsize=(10,6))
    csa = axes.contourf(x.time, x.pfull, x, cmap='RdBu_r', norm=norm, extend='both', levels=lvls)
    cb  = fig.colorbar(csa, extend='both')
    cb.set_label(label=unit, size='xx-large')
    cb.ax.tick_params(labelsize='x-large')
    axes.set_xlabel('Lag (Days)', fontsize='xx-large')
    axes.set_ylabel('Pressure (hPa)', fontsize='xx-large')
    axes.set_ylim(max(x.pfull), 1)
    axes.set_yscale('log')
    axes.axvline(0, color='w')
    axes.axhline(300, color='w')
    axes.text(-17, 1.75, label, color='k', fontsize='xx-large')
    axes.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
    plt.savefig(name, bbox_inches = 'tight')
    return plt.close()

def SSW_comp(indir, outdir, exp, labels, unit, addon, lvls):
    polarcap = slice(65,90) # 65 to 90 N
    days = np.arange(-20, 61, 1) # Want a window from -20 to + 60 days
    for i in range(len(exp)):
        print(datetime.now(), " - {0:.0f}/{1:.0f} experiments".format(i+1, len(exp)))
        indices = identify_SSWs(outdir, exp[i])
        # Set-up polar cap pressure anomaly dataset 
        print(datetime.now(), " - finding polar cap anomalies")
        if var == 'a':
            ds = xr.open_dataset(indir+exp[i]+'_h.nc', decode_times=False).height
            ds_polarcap = ds.sel(lat=polarcap).mean('lat')
            ds_polarcap_z = ds_polarcap.mean('lon')
        elif var == 'b':
            ds = xr.open_dataset(indir+exp[i]+'_Tz.nc', decode_times=False).temp
            ds_polarcap_z = ds.sel(lat=polarcap).mean('lat')
        ds_polarcap_mean = ds_polarcap_z.mean('time')
        ds_polarcap_anom = ds_polarcap_z - ds_polarcap_mean
    
        print(datetime.now(), " - finding SSW windows")
        SSW_windows = []
        for idx in indices:
            if idx + min(days) < 0:
                start = 0
                end = idx + max(days) + 1
                new_days = np.arange(min(days) - (idx + min(days)), max(days)+1, 1)
                window = ds_polarcap_anom[start:end].assign_coords({'time' : new_days}).transpose()
                empty = xr.zeros_like(ds_polarcap_anom[:abs(idx + min(days))]).assign_coords({'time' : np.arange(min(days), min(window.time))})
                empty = xr.where(empty==0, np.nan, empty)
                window = xr.concat([empty, window], 'time').transpose('pfull', 'time')
            elif idx+max(days) > len(ds.time):
                excess = idx+max(days) - len(ds.time)
                start = idx + min(days)
                end = len(ds.time)-1
                new_days = np.arange(min(days), len(ds.time) - idx-1, 1)
                window = ds_polarcap_anom[start:end].assign_coords({'time' : new_days}).transpose()
                empty = xr.zeros_like(ds_polarcap_anom[:len(days) - (end - start)]).assign_coords({'time' : np.arange(max(window.time)+1, max(days)+1)})
                empty = xr.where(empty==0, np.nan, empty)
                window = xr.concat([window, empty], 'time').transpose('pfull', 'time')
            else:
                start = idx + min(days)
                end = idx + max(days) + 1
                window = ds_polarcap_anom[start:end].assign_coords({'time' : days}).transpose('pfull', 'time')
            SSW_windows.append(window) # Common time coordinates
    
        print(datetime.now(), " - standardising")
        SSW_windows_std = []
        for w in range(len(SSW_windows)):
            print(datetime.now(), " - {0:.0f}/{1:.0f} windows".format(w+1, len(SSW_windows)))
            window_std = standardise(ds_polarcap_z.transpose(), SSW_windows[w], ds.pfull, days)
            SSW_windows_std.append(window_std)
        """
            plot_SSW_comp(window_std, unit, 'SSWcomp_'+str(w+1)+'.png', lvls, w+1)

        print(datetime.now(), " - making gif")
        # Merge all plots into a GIF for visualisation
        image_list = glob('SSWcomp_*.png')
        list.sort(image_list, key = lambda x: int(x.split('_')[1].split('.png')[0]))
        IMAGES = []
        for w in range(len(SSW_windows)):
            IMAGES.append(imageio.imread(image_list[w]))
        imageio.mimsave(exp[i]+'_SSWcomp'+addon+'.gif', IMAGES, 'GIF', duration = 1/2)
        # Delete all temporary plots from working directory
        for w in range(len(SSW_windows)):
            os.remove(image_list[w])
        """
        print(datetime.now(), " - plotting composite")
        composite_std = xr.concat(SSW_windows_std, 'window').mean('window')
        plot_SSW_comp(composite_std, unit, exp[i]+'_SSWcomp'+addon+'.pdf', lvls, labels[i])

def find_responses1(extension):
    exp, labels, xlabel = return_exp(extension)
    labels = labels[1:]
    neck_response = []
    SPV_response = []
    for i in range(len(exp)):
        utz = xr.open_dataset(indir+exp[i]+'_utz.nc', decode_times=False).ucomp[0]
        if i == 0:
            neck_ctrl = calc_winds(utz, 70, 45, 55)
            SPV_ctrl = calc_winds(utz, 10, 60, 75)
        else:
            neck_exp = calc_winds(utz, 70, 45, 55)
            neck_response.append(neck_exp - neck_ctrl)
            SPV_exp = calc_winds(utz, 10, 60, 75)
            SPV_response.append(SPV_exp - SPV_ctrl)
    return labels, neck_response, SPV_response

def find_responses2(extension):
    exp, labels, xlabel = return_exp(extension)
    neck = []
    SPV = []
    neck_response = []
    SPV_response = []
    for j in range(len(exp[0])):
        utz = xr.open_dataset(indir+exp[0][j]+'_utz.nc', decode_times=False).ucomp[0]
        neck_noheat = calc_winds(utz, 70, 45, 55)
        SPV_noheat = calc_winds(utz, 10, 60, 75)

        utz = xr.open_dataset(indir+exp[1][j]+'_utz.nc', decode_times=False).ucomp[0]
        neck_heat = calc_winds(utz, 70, 45, 55)
        SPV_heat = calc_winds(utz, 10, 60, 75)
        neck_response.append(neck_heat - neck_noheat)
        SPV_response.append(SPV_heat - SPV_noheat)
        neck.append(neck_noheat)
        SPV.append(SPV_noheat)

    return labels, xlabel, neck_response, SPV_response, neck, SPV

def neck_winds(exp_type):
    if exp_type == 'heat':
        neck_response = []
        SPV_response = []
        labels = []
        xlabels = [r'$p_{top}$ (hPa)', r'A (K day$^{-1}$)', r'$\phi_w$ ($\degree$)']

        extension = '_depth'
        l, n, S = find_responses1(extension)
        neck_response.append(n)
        SPV_response.append(S)
        labels.append(l)
        
        extension = '_strength'
        l, n, S = find_responses1(extension)
        neck_response.append(n)
        SPV_response.append(S)
        labels.append(l)

        extension = '_width'
        l, n, S = find_responses1(extension)
        neck_response.append(n)
        SPV_response.append(S)
        labels.append(l)

        fig, ax = plt.subplots(figsize=(8,8))        
        colours = colours[1:]
        markers = ['o', 's', '^']
        for i in range(len(neck_response)):
            for j in range(len(neck_response[i])):
                ax.scatter(x=neck_response[i][j], y=SPV_response[i][j], marker=markers[i], s=100, c=reds[j], label=labels[i][j])
        #ax.plot(np.unique(neck_response), np.poly1d(np.polyfit(neck_response, SPV_response, 1))(np.unique(neck_response)),\
        #    color='k', linewidth=1.5, linestyle='--')
        #ax.axhline(0, color='k', linewidth=0.5)
        ax.set_xlabel(r'Change in neck winds: 70hPa, 45N-55N (m s$^{-1}$)', fontsize='xx-large')
        ax.set_ylabel(r'Change in SPV strength: 10hPa, 60N-75N (m s$^{-1}$)', fontsize='xx-large')
        legend_elements = [Line2D([0], [0], marker=markers[0], color='w', label=xlabels[0], markerfacecolor='k', markersize=15),\
                    Line2D([0], [0], marker=markers[1], color='w', label=xlabels[1], markerfacecolor='k', markersize=15),\
                    Line2D([0], [0], marker=markers[2], color='w', label=xlabels[2], markerfacecolor='k', markersize=15)]
        ax.legend(loc='upper left', handles=legend_elements, fontsize='xx-large', fancybox=False, shadow=False, ncol=1)
        ax.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
        plt.savefig(exp_type+'_neckcheck.pdf', bbox_inches = 'tight')

    elif exp_type == 'test': #'vtx':
        labels, xlabel, neck_response, SPV_response, neck, SPV = find_responses2('_test') #'_vtx')
        colours1 = blues[1:]
        fig, ax = plt.subplots(figsize=(8,8))        
        for i in range(len(neck_response)):
            ax.scatter(x=neck_response[i], y=SPV_response[i], marker='D', s=100, c=colours1[i], label=labels[i])
        ax.plot(np.unique(neck_response), np.poly1d(np.polyfit(neck_response, SPV_response, 1))(np.unique(neck_response)),\
            color='k', linewidth=1.5, linestyle='--')
        ax.axhline(0, color='k', linewidth=0.5)
        ax.set_xlabel(r'Change in neck winds: 70hPa, 45N-55N (m s$^{-1}$)', fontsize='xx-large')
        ax.set_ylabel(r'Change in SPV winds: 10hPa, 60N-75N (m s$^{-1}$)', fontsize='xx-large')
        ax.legend(title=xlabel, title_fontsize='xx-large', fontsize='xx-large', fancybox=False, shadow=False, ncol=1)
        ax.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
        plt.savefig(exp_type+'_neckcheck.pdf', bbox_inches = 'tight')
        plt.close()

        fig, ax = plt.subplots(figsize=(10,8))
        colours2 = reds[1:]
        greys = ['#dedede', '#c6c6c6', '#a7a7a7', '#868686', '#686868', '#484848']
        #ax2 = ax.twiny()
        for i in range(len(neck)):
            ax.scatter(x=neck[i], y=SPV_response[i], marker='D', s=100, c=colours1[i], label=labels[i])
        ax.plot(np.unique(neck), np.poly1d(np.polyfit(neck, SPV_response, 1))(np.unique(neck)),\
            color=colours1[-2], linewidth=1.5, linestyle='--')
        ax.axhline(0, color='k', linewidth=0.5)
        ax.set_xlabel(r'Climatological neck winds: 70hPa, 45N-55N (m s$^{-1}$)', fontsize='xx-large', color=colours1[-2])
        ax.set_ylabel(r'Change in SPV winds: 10hPa, 60N-75N (m s$^{-1}$)', fontsize='xx-large')
        ax.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in', labelcolor=colours1[-2])
        ax.tick_params(axis='y', labelcolor='k')
        ax.legend(title=xlabel, title_fontsize='xx-large', fontsize='xx-large', fancybox=False, shadow=False, ncol=1)
        #for i in range(len(SPV)):
        #    ax2.scatter(x=SPV[i], y=SPV_response[i], marker='o', s=100, c=colours2[i], label=labels[i])
        #ax2.plot(np.unique(SPV), np.poly1d(np.polyfit(SPV, SPV_response, 1))(np.unique(SPV)),\
        #    color=colours2[-2], linewidth=1.5, linestyle=':')
        #ax2.set_xlabel(r'Climatological SPV winds: 10hPa, 60N-75N (m s$^{-1}$)', fontsize='xx-large', color=colours2[-2])
        #ax2.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in', labelcolor=colours2[-2])
        #legend_elements = []
        #for i in range(len(labels)):
        #    legend_elements.append(Line2D([0], [0], marker='s', color='w', label=labels[i], markerfacecolor=greys[i], markersize=15))
        #ax.legend(handles=legend_elements, title=xlabel, title_fontsize='xx-large', fontsize='xx-large',\
        #          fancybox=False, shadow=False, ncol=1, handler_map={tuple: HandlerTuple(ndivide=None)})
        plt.savefig(exp_type+'_climatologycheck.pdf', bbox_inches = 'tight')

    return plt.close()

if __name__ == '__main__': 
    #Set-up data to be read in
    indir = '/disco/share/rm811/processed/'
    outdir = '../Files/'
    basis = 'PK_e0v4z13'
    var_type = input("Plot a) depth, b) width, c) location, d) strength, e) vortex experiments or f) test?")
    if var_type == 'a':
        extension = '_depth'
    elif var_type == 'b':
        extension = '_width'
    elif var_type == 'c':
        extension = '_loc2'
    elif var_type == 'd':
        extension = '_strength'
    elif var_type == 'e':
        basis = 'PK_e0vXz13'
        extension = '_vtx'
    elif var_type == 'f':
        extension = '_test'
    exp, labels, xlabel = return_exp(extension)
    n = len(exp)

    #User choices for plotting - subjects
    level = input('Plot a) near-surface, b) tropospheric, c) stratospheric wind stats, d) SPV stuff, or e) SPV and SSW values?')

    colors = ['k', '#B30000', '#FF9900', '#FFCC00', '#00B300', '#0099CC', '#4D0099', '#CC0080']
    reds = ['k', '#fedbcb', '#fcaf94', '#fc8161', '#f44f39', '#d52221', '#aa1016', '#67000d']
    blues = ['k', '#dbe9f6', '#bbd6eb', '#88bedc', '#549ecd',  '#2a7aba', '#0c56a0', '#08306b']
    legend = [r'A = 2 K day$^{-1}$', r'A = 4 K day$^{-1}$', r'A = 6 K day$^{-1}$', r'A = 8 K day$^{-1}$'] 
    #legend = [r'$p_{top} = 800$ hPa', r'$p_{top} = 600$ hPa', r'$p_{top} = 400$ hPa']

    if level == 'a':
        p = 900 #hPa
        style = ['-', ':']
        cols = len(exp)
        plot_Xwinds(outdir, indir, exp, labels, colors, style, cols, exp[0], p)
    elif level == 'b':
        alt = input("Plot a) top-down jet view, b) EDJ strength & lat vs experiment, c) recreate Gerber & Polvani 2009, or d) check EDJ response to polar heating? ")
        if alt == "a":
            p = [850, 500] #hPa
            lvls = [np.arange(-25, 27.5, 2.5), np.arange(50, 55, 5)]
            #windsvexp(outdir, labels, xlabel, str(p), basis+extension)
            for j in range(len(p)):
                for i in range(n):
                    print(datetime.now(), " - finding winds ({0:.0f}/{1:.0f})".format(i+1, n))
                    u = xr.open_dataset(indir+exp[i]+'_ut.nc', decode_times=False).ucomp[0]
                    show_jet(u, p[j], lvls[j], exp[i])
        elif alt == "b":
            p = 850 # hPa for EDJ
            if extension == '_vtx' or '_loc2':
                windsvexp(outdir, labels, xlabel, str(p), [basis+extension+'_noperturb', basis+extension+'_perturb'])
            else:
                windsvexp(outdir, labels, xlabel, str(p), 'PK_e0v0z13_noheat')
        elif alt == "c":
            perturb = '_q6m2y45l800u200' 
            exp = ['PK_e0v1z13', 'PK_e0v2z13', 'PK_e0v3z13', 'PK_e0v4z13', 'PK_e0v5z13', 'PK_e0v6z13']
            #exp = ['PK_e0v1z13'+perturb, 'PK_e0v2z13'+perturb,'PK_e0v3z13'+perturb, 'PK_e0v4z13'+perturb,'PK_e0v5z13'+perturb, 'PK_e0v6z13'+perturb]
            GP09(exp)
        elif alt == "d":
            EDJ_loc()
    elif level == 'c':
        alt = input("Plot a) neck, b) 100 hPa SPV or c) 10 hPa SPV winds?")
        if alt == "a":
            #neck winds as per Isla Simpson's SPARC talk @ 45-55N, 70 hPa
            p = 70
            lats = slice(45,55)
            lab = 'neck '
            neck_winds('test') #vtx')
        elif alt == "b":
            #lower SPV winds @ 60N, 100 hPa
            p = 100
            lats = 60
            lab = 'lower SPV '
        elif alt == "c":
            #lower SPV winds @ 60N, 10 hPa
            p = 10
            lats = 60
            lab = 'SPV '
            n = len(exp)
        """
        if n == 2:
            means = []
            modes = []
            SDs = []
            errs = []
            for i in range(n):
                me, mo, sd, e = plot_pdf('u', indir, exp[i], '_uz.nc', '', p, lats, labels, lab+r"zonal-mean zonal wind (m s$^{-1}$)", blues, basis+extension)[:4]
                means.append(me)
                modes.append(mo)
                SDs.append(sd)
                errs.append(e)
            SPVvexp1(n, means, modes, SDs, errs, p, labels, xlabel, basis+extension)
        else:
            me, mo, sd, e, sk, k = plot_pdf('u', indir, exp, '_uz.nc', '', p, lats, labels, lab+r"zonal-mean zonal wind (m s$^{-1}$)", blues, basis+extension)
            SPVvexp1(n, me, mo, sd, e, p, labels, xlabel, basis+extension)
            #SPVvexp2(sk, k, p, labels, xlabel, basis+extension)
        """
    elif level == 'd':
        p = 10 # pressure level at which we want to find the SPV (hPa)
        #User choice for plotting - type
        plot_type = input('Plot a) SPV over time, b) 10 hPa max. wind/lats, c) SSW frequencies, d) lat-p s.d., e) paper plot, or f) SSW composites?')
        if plot_type == 'a':
            exp = [exp[1], exp[-1]]
            labels = [labels[1], labels[-1]]
            colors = ['#88bedc', '#0c56a0']
            style = ['-', '-']
            cols = 2
            plot_vtx(outdir, exp, labels, colors, style, cols, exp[0])
        elif plot_type == 'b':
            if extension == '_vtx' or '_loc2':
                windsvexp(outdir, labels, xlabel, str(p), [basis+extension+'_noheat', basis+extension+'_heat'])
            else:
                windsvexp(outdir, labels, xlabel, str(p), basis+extension)
        elif plot_type == 'c':
            SSWsvexp(outdir, exp, labels, xlabel, basis+extension)
            #SSWsvexp_multi(outdir, exp, labels, xlabel, legend, ['#B30000', '#00B300', '#0099CC', 'k'], basis+extension)
        elif plot_type == 'd':
            ulvls = np.arange(-200, 200, 10)
            plot_what = input('Plot a) climatology or b) difference?)')
            for i in range(n):
                print(datetime.now(), " - finding s.d. ({0:.0f}/{1:.0f})".format(i+1, n))
                u = xr.open_dataset(indir+exp[i]+'_uz.nc', decode_times=False).ucomp
                utz = xr.open_dataset(indir+exp[i]+'_utz.nc', decode_times=False).ucomp[0]
                if plot_what == 'a':
                    lat, p, sd = find_sd(u)
                    NH_zonal(lat, p, sd, utz, np.arange(0, 42, 2), ulvls, 'Blues',\
                        r'zonal-mean zonal wind S.D. (ms$^{-1}$)', exp[i]+'_usd.pdf')
                elif plot_what == 'b':
                    if i == 0:
                        print("skipping control")
                    elif i != 0:
                        lat, p, sd1 = find_sd(u)
                        lat, p, sd2 = find_sd(xr.open_dataset(indir+exp[0]+'_uz.nc', decode_times=False).ucomp)
                        sd_diff = sd1 - sd2
                        NH_zonal(lat, p, sd_diff, utz, np.arange(-20, 22, 2), ulvls, 'RdBu_r',\
                            r'zonal-mean zonal wind S.D. (ms$^{-1}$)', exp[i]+'_usd_diff.pdf')
        elif plot_type == 'e':
            SPV_report_plot(exp, labels, xlabel, basis+extension)
        elif plot_type == 'f':
            var = input('a) GPH or b) temperature anomaly? ')
            if var == 'a':
                unit = 'Standardised PCH Anomaly (m)'
                addon = '-h'
                lvls = np.arange(-1, 3.25, 0.25)
            elif var == 'b':
                unit = 'Standardised PCT Anomaly (K)'
                addon = '-T'
                lvls = np.arange(-2, 2.25, 0.25)
            SSW_comp(indir, outdir, exp, labels, unit, addon, lvls)
    elif level == 'e':
        n = len(exp)
        u10_full = []
        u100_full = []
        SSWs = []
        for i in range(n):
            u10 = open_file(outdir, exp[i], 'u10')
            u100 = open_file(outdir, exp[i], 'u100')
            report_vals(exp[i], labels[i], u10)
            report_vals(exp[i], labels[i], u100, SSW_flag=False)
            count = 0
            for j in u100:
                if j < 0:
                    count += 1
            print(labels[i]+' days w/ westward winds (100 hPa): {0:.2f} %'.format(100*count/len(u100)))
            u10_full.append(u10)
            u100_full.append(u100)

        obs = input('Plot MERRA2 data? (y/n)')
        if obs == 'y':
            exp = ['obs_u1060', 'obs_u10060']
            labels = ['MERRA2 @ 10 hPa', 'MERRA2 @ 100 hPa']
            months = 'NDJF'
            u10 = open_file(outdir, exp[0], months)
            u100 = open_file(outdir, exp[1], months)
            report_vals(exp[0], labels[0], u10)
            report_vals(exp[1], labels[1], u100, SSW_flag=False)
            count = 0
            for j in u100:
                if j < 0:
                    count += 1
            print(labels[1]+' days w/ westward winds: {0:.2f} %'.format(100*count/len(u100)))

            plot1 = [u100, u100_full[0], u100_full[1]] 
            plot2 = [u10, u10_full[0], u10_full[1]]
            labels = ['a) MERRA2', r'b) $\gamma = 3$ K km$^{-1}$', r'c) $\gamma = 4$ K km$^{-1}$']
            lines = ['-', '--', ':']
            x_min = x_max = 0
            fig, ax = plt.subplots(figsize=(6,6))
            for i in range(len(plot1)):
                x = plot1[i]
                x_sort, f, m = pdf(x)
                if max(x) > x_max:
                    x_max = max(x)
                if min(x) < x_min:
                    x_min = min(x)
                ax.plot(x_sort, f, linewidth=1.25, color='#4D0099', linestyle=lines[i])
            ax.set_ylim(bottom=0)
            ax.set_ylabel('100 hPa', color='#4D0099', fontsize='x-large')
            ax.tick_params(axis='y', colors='#4D0099')

            ax2 = ax.twinx()
            for i in range(len(plot2)):
                x = plot2[i]
                x_sort, f, m = pdf(x)
                if max(x) > x_max:
                    x_max = max(x)
                if min(x) < x_min:
                    x_min = min(x)
                ax2.plot(x_sort, f, linewidth=1.25, color='#B30000', label=labels[i], linestyle=lines[i])
            ax2.set_ylim(bottom=0)
            ax2.set_ylabel('10 hPa', color='#B30000', fontsize='x-large')
            ax2.tick_params(axis='y', colors='#B30000')  

            ax.axvline(0, color='k', linewidth=0.25)
            ax.set_xlim(x_min, x_max)
            ax.set_xlabel(r'$60\degree$N zonal wind', fontsize='x-large')
            ax.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
            ax2.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
            plt.legend(fancybox=False, ncol=1, fontsize='x-large')
            plt.savefig('MERRA2vCtrl_'+months+'_uwindpdf.pdf', bbox_inches = 'tight')
            plt.show()
            plt.close()
        else:
            print('done')