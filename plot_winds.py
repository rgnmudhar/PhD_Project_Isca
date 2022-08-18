"""
Script for functions involving winds - near-surface, tropospheric jet and stratospheric polar vortex.
"""

from glob import glob
import xarray as xr
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
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

def plot_vtx(exp, labels, colors, style, cols, fig_name):
    """
    Plots strength of winds at 60N, 10 hPa only.
    Best for 2 datsets, the second of which has its SSW statistics as a plot subtitle
    """
    
    print(datetime.now(), " - plotting SPV")
    fig, ax = plt.subplots(figsize=(8,6))
    for i in range(len(exp)):
        SPV = open_file(exp[i], 'SPV')
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

def SPVvexp1(exp, exp_names, xlabel, name):
    """
    Plots the mean and standard deviation of SPV against (heating) experiment.
    """
    print(datetime.now(), " - plotting SPV mean and variance vs experiment")
    mean = []
    mode = []
    err = []
    sd = []
    for i in range(len(exp)):
        SPV = open_file(exp[i], 'SPV')
        mean.append(np.mean(SPV))
        err.append(np.std(SPV/np.sqrt(len(SPV))))
        sd.append(np.std(SPV))
        x, f, m = pdf(SPV)
        mode.append(m)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.errorbar(exp_names[1:], mean[1:], yerr=err[1:], fmt='o', linewidth=1.25, capsize=5, color='#B30000', linestyle=':', label='Mean')
    ax.plot(exp_names[1:], mode[1:], marker='o', linewidth=1.25, color='#B30000', linestyle='-.', label='Mode')
    plt.legend(loc='upper center' , bbox_to_anchor=(0.5, 1), fancybox=False, shadow=False, ncol=2, fontsize='x-large')
    ax.set_xticks(exp_names)
    ax.set_xlabel(xlabel, fontsize='x-large')
    ax.set_ylabel(r'10 hPa, 60 N Zonal Wind Average (ms$^{-1}$)', fontsize='x-large', color='#B30000')
    #ax.set_ylim(36,42)
    ax.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    ax2 = ax.twinx()
    ax2.plot(exp_names[1:], sd[1:], marker='o', linewidth=1.25, color='#4D0099', linestyle=':', label='S.D.')
    ax2.set_ylabel(r'10 hPa, 60 N Zonal Wind S.D. (ms$^{-1}$)', color='#4D0099', fontsize='x-large')
    #ax2.set_ylim(12,22)
    ax2.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    plt.savefig(name+'_SPVvheat.pdf', bbox_inches = 'tight')

    return plt.close()

def windsvexp(labels, xlabel, p, name):
    """
    Uses jet_locator functions to find location and strength of maximum stratopsheric vortex (10 hPa).
    Then plots this against (heating) experiment.
    """
    print(datetime.now(), " - plotting wind maxima vs experiment")
    fig, ax = plt.subplots(figsize=(10,6))
    ax.errorbar(labels[1:], open_file(name, 'maxwinds'+p)[1:], yerr=open_file(name, 'maxwinds_sd'+p)[1:], fmt='o', linewidth=1.25, capsize=5, color='#B30000', linestyle=':')
    ax.set_xticks(labels)
    ax.set_xlabel(xlabel, fontsize='x-large')
    ax.set_ylabel(r'Strength of Max. 10 hPa Zonal Wind (ms$^{-1}$)', color='#B30000', fontsize='x-large')
    ax.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    ax2 = ax.twinx()
    ax2.errorbar(labels[1:], open_file(name, 'maxlats'+p)[1:], yerr=open_file(name, 'maxlats_sd'+p)[1:], fmt='o', linewidth=1.25, capsize=5, color='#4D0099', linestyle=':')
    ax2.set_ylabel(r'Laitude of Max. 10 hPa Zonal Wind ($\degree$N)', color='#4D0099', fontsize='x-large')
    ax2.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    plt.savefig(name+'_windsvexp'+p+'.pdf', bbox_inches = 'tight')

    return plt.close()

def SSWsvexp(exp, x, xlabel, fig_name):
    """
    Plots SSW frequency against (heating) experiment.
    """
    print(datetime.now(), " - finding SSWs")
    SSWs, errors = find_SSWs(exp)
    og = SSWs[0]
    og_err = errors[0]
    print(SSWs, errors)
    print(datetime.now(), " - plotting SSWs vs experiment")
    fig, ax = plt.subplots(figsize=(10,6))
    ax.errorbar(x[1:], SSWs[1:], yerr=errors[1:], fmt='o', linewidth=1.25, capsize=5, color='#B30000', linestyle=':')
    ax.set_xlim(-0.5,3.5)
    ax.set_ylim(0.25,0.5)
    ax.set_xticks(x[1:])
    ax.set_xlabel(xlabel, fontsize='x-large')
    ax.set_ylabel(r'SSWs per 100 days', fontsize='x-large')
    ax.axhline(0.42, color='#4D0099', linewidth=0.5)
    ax.text(2.85, 0.425, 'ERA-Interim', color='#4D0099', fontsize='x-large')
    ax.axhline(og, color='#666666', linewidth=0.5)
    ax.fill_between(range(-1,8), (og - og_err), (og + og_err), facecolor ='gainsboro', alpha = 0.4)
    ax.text(3.1, 0.33, 'Control', color='#666666', fontsize='x-large')
    ax.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    plt.savefig(fig_name+'_SSWsvheat.pdf', bbox_inches = 'tight')

    return plt.close()

def SSWsvexp_multi(exp, x, xlabel, legend, colors, fig_name):
    """
    Plots SSW frequency against (heating) experiment for 3 sets of experiments
    """
    print(datetime.now(), " - finding SSWs")
    og, og_err = find_SSWs([exp[0]])
    og = og[0]
    og_err = og_err[0]
    print(datetime.now(), " - plotting SSWs vs experiment")
    fig, ax = plt.subplots(figsize=(10,6))
    for i in np.arange(1,len(exp),1):
        SSWs, errors = find_SSWs(exp[i])
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

def pdf(x, plot=False):
    x = np.sort(x)
    ae, loce, scalee = sps.skewnorm.fit(x)
    p = sps.skewnorm.pdf(x, ae, loce, scalee)
    if plot==True:
        s = np.std(x)
        mean = np.mean(x)
        f = (1 / (s * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean)/s)**2)
        plt.hist(x, bins = 50, density=True)
        plt.plot(x, f)
        plt.plot(x, p)
        plt.show()
    mode = x[int(np.argmax(p))]
    return x, p, mode

def plot_pdf(exp, labels, colors, name):
    print(datetime.now(), " - plotting SPV pdfs")
    x_min = x_max = 0 
    fig, ax = plt.subplots(figsize=(8,6))
    for i in range(len(exp)):
        SPV = open_file(exp[i], 'SPV')
        x, f, mode = pdf(SPV)
        if max(x) > x_max:
            x_max = max(x)
        if min(x) < x_min:
            x_min = min(x)
        ax.plot(x, f, linewidth=1.25, color=colors[i], label=labels[i])
    ax.set_xlim(x_min, x_max)
    ax.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    plt.legend(loc='upper right',fancybox=False, shadow=True, ncol=1, fontsize='large')
    plt.savefig(name+'_pdf.pdf', bbox_inches = 'tight')

    return plt.close()

def find_sd(indir, exp):
    print(datetime.now(), " - opening files")
    u = xr.open_dataset(indir+exp+'_uz.nc', decode_times=False).ucomp
    utz = xr.open_dataset(indir+exp+'_utz.nc', decode_times=False).ucomp
    p = u.coords['pfull']
    lat = u.coords['lat']
    sd = np.empty_like(u[0])
    print(datetime.now(), " - finding zonal mean s.d. over latitude-pressure")
    for i in range(len(p)):
        for j in range(len(lat)):
            sd[i,j] = np.std(u[:,i,j])
    return lat, p, utz, sd

def plot_sd(lat, p, u, sd, lvls, exp, colors):
    ulvls = np.arange(-200, 200, 10)
    print(datetime.now(), " - plotting")
    fig, ax = plt.subplots(figsize=(6,6))
    cs1 = ax.contourf(lat, p, sd, levels=lvls, cmap=colors)
    ax.contourf(cs1, colors='none')
    cs2 = ax.contour(lat, p, u[0], colors='k', levels=ulvls, linewidths=1, alpha=0.4)
    cs2.collections[int(len(ulvls)/2)].set_linewidth(1.5)
    cb = plt.colorbar(cs1)
    cb.set_label(label=r'Zonal-mean Zonal Wind S.D. (ms$^{-1}$)', size='x-large')
    cb.ax.tick_params(labelsize='x-large')
    plt.scatter(lat.sel(lat=60, method='nearest'), p.sel(pfull=10, method='nearest'), marker='x', color='#B30000')
    plt.xlabel(r'Latitude ($\degree$N)', fontsize='x-large')
    plt.xlim(0,90)
    plt.xticks([10, 30, 50, 70, 90], ['10', '30', '50', '70', '90'])
    plt.ylabel('Pressure (hPa)', fontsize='x-large')
    plt.ylim(max(p), 1) #goes to ~1hPa
    plt.yscale('log')
    plt.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    plt.savefig(exp+'_sd.pdf', bbox_inches = 'tight')
    return plt.close()

def SPVvexp2(exp, exp_names, xlabel, name):
    """
    Plots the skew and kurtosis of SPV against (heating) experiment.
    """
    print(datetime.now(), " - plotting SPV skew and kurtosis vs experiment")
    skew = []
    kurt = []
    for i in range(len(exp)):
        SPV = open_file(exp[i], 'SPV')
        skew.append(sps.skew(SPV))
        kurt.append(sps.kurtosis(SPV))
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(exp_names[1:], skew[1:], marker='o', linewidth=1.25, color='#B30000', linestyle=':')
    ax.set_xticks(exp_names)
    ax.set_xlabel(xlabel, fontsize='x-large')
    ax.set_ylabel(r'10 hPa, 60 N Zonal Wind Skewness', fontsize='x-large', color='#B30000')
    ax.axhline(skew[0], color='#B30000', linewidth=0.5)
    ax.text(-0.3, 0.07, 'Control', color='#B30000', fontsize='x-large')
    ax.set_ylim(-0.5, 0.5)
    ax.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    ax2 = ax.twinx()
    ax2.plot(exp_names[1:], kurt[1:], marker='o', linewidth=1.25, color='#4D0099', linestyle=':')
    ax2.set_ylabel(r'10 hPa, 60 N Zonal Wind Kurtosis', color='#4D0099', fontsize='x-large')
    ax2.axhline(kurt[0], color='#4D0099', linewidth=0.5)
    ax2.text(3, -0.57, 'Control', color='#4D0099', fontsize='x-large')
    ax2.set_xlim(-0.5, 3.5)
    ax2.set_ylim(-0.9, 0.9)
    ax2.fill_between(range(-1,8), -1, 0, facecolor ='gainsboro', alpha = 0.4)
    ax2.text(0.75, -0.5, 'Negative Skew, Lighter Tails', color='#666666', fontsize='x-large')
    ax2.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    plt.savefig(name+'_SPVstats.pdf', bbox_inches = 'tight')
    return plt.close()

if __name__ == '__main__': 
    #Set-up data to be read in
    indir = '/disco/share/rm811/processed/'
    basis = 'PK_e0v4z13'
    perturb = '_q6m2y45l800u200'
    exp = [basis+'_w15a4p800f800g50'+perturb, basis+'_a4x75y180w5v30p800_q6m2y45']
    extension = 0 #'_strengthp400'
    if extension == '_depth':
        exp = [basis+perturb,\
        basis+'_w15a4p900f800g50'+perturb,\
        basis+'_w15a4p800f800g50'+perturb,\
        basis+'_w15a4p700f800g50'+perturb,\
        basis+'_w15a4p600f800g50'+perturb,\
        basis+'_w15a4p500f800g50'+perturb,\
        basis+'_w15a4p400f800g50'+perturb,\
        basis+'_w15a4p300f800g50'+perturb]
        labels = ['no heat', '900', '800', '700', '600', '500', '400', '300']
        xlabel = 'Depth of Heating (hPa)'
    elif extension == '_width':
        exp = [basis+perturb,\
        basis+'_w10a4p800f800g50'+perturb,\
        basis+'_w15a4p800f800g50'+perturb,\
        basis+'_w20a4p800f800g50'+perturb,\
        basis+'_w25a4p800f800g50'+perturb,\
        basis+'_w30a4p800f800g50'+perturb,\
        basis+'_w35a4p800f800g50'+perturb,\
        basis+'_w40a4p800f800g50'+perturb]
        labels = ['no heat', '10', '15', '20', '25', '30', '35', '40']
        xlabel = r'Extent of Heating ($\degree$)'
    elif extension == '_strengthp400':    
        #exp = [basis+perturb,\
        exp = [basis+'_w15a2p400f800g50'+perturb,\
        basis+'_w15a4p400f800g50'+perturb,\
        basis+'_w15a6p400f800g50'+perturb,\
        basis+'_w15a8p400f800g50'+perturb]
        labels = ['no heat', '2', '4', '6', '8'] #['no heat', '800', '600', '400']
        xlabel = r'Strength of Heating (K day$^{-1}$)'
    elif extension == '_loc':   
        perturb = '_q6m2y45'
        exp = [basis+'_q6m2y45l800u200',\
            basis+'_a4x75y0w5v30p800'+perturb,\
            basis+'_a4x75y90w5v30p800'+perturb,\
            basis+'_a4x75y180w5v30p800'+perturb,\
            basis+'_a4x75y270w5v30p800'+perturb]
        labels = ['no heat', '0', '90', '180', '270']
        xlabel = r'Longitude of Heating ($\degree$E)'
    
    #User choices for plotting - subjects
    level = input('Plot a) near-surface winds, b) tropospheric jet, c) stratospheric polar vortex?')

    colors = ['#B30000', '#FF9900', '#FFCC00', '#00B300', '#0099CC', '#4D0099', '#CC0080', '#666666']
    legend = [r'A = 2 K day$^{-1}$', r'A = 4 K day$^{-1}$', r'A = 6 K day$^{-1}$', r'A = 8 K day$^{-1}$'] #[r'$p_{top} = 800$ hPa', r'$p_{top} = 600$ hPa', r'$p_{top} = 400$ hPa']

    if level == 'a':
        p = 900 #hPa
        style = ['-', ':']
        cols = len(exp)
        plot_winds(indir, exp, labels, colors, style, cols, exp[0], p)

    elif level == 'b':
        p = 850 #hPa
        windsvexp(labels, xlabel, str(p), basis+extension)

    elif level == 'c':
        p = 10 # pressure level at which we want to find the SPV (hPa)
        #User choice for plotting - type
        plot_type = input('Plot a) SPV @ 10hPa, 60N over time, \
            b) 10 hPa max. wind and lat vs experiment, \
            c) SPV mean and s.d. vs experiment, or \
            d) SSW frequency vs experiment, \
            e) distribution and stats, or \
            f) s.d. as a function of lat and p?')
        if plot_type == 'a':
            exp = exp[:2]
            labels = ['zonally symmetric', 'off-pole']
            colors = ['#0099CC', '#B30000']
            style = ['-', '-']
            cols = 2
            plot_vtx(exp, labels, colors, style, cols, exp[0])
        elif plot_type == 'b':
            windsvexp(labels, xlabel, str(p), basis+extension)
        elif plot_type == 'c':
            SPVvexp1(exp, labels, xlabel, basis+extension)
        elif plot_type == 'd':
            SSWsvexp(exp, labels, xlabel, basis+extension)
            #SSWsvexp_multi(exp, labels, xlabel, legend, ['#B30000', '#00B300', '#0099CC', 'k'], basis+extension)
        elif plot_type == 'e':
            plot_pdf(exp, labels, colors, basis+extension)
            SPVvexp2(exp, labels, xlabel, basis+extension)
        elif plot_type == 'f':
            plot_what = input('Plot a) climatology or b) difference?)')
            if plot_what == 'a':
                for i in range(len(exp)):
                    print(datetime.now(), " - ", exp[i])
                    lat, p, u, sd = find_sd(indir, exp[i])
                    plot_sd(lat, p, u, sd, np.arange(0, 42, 2), exp[i], 'Blues')
            elif plot_what == 'b':
                lat, p, u1, sd1 = find_sd(indir, exp[0])
                lat, p, u2, sd2 = find_sd(indir, exp[1])
                sd_diff = sd1 - sd2
                plot_sd(lat, p, u1, sd_diff, np.arange(-10, 11, 1), basis+'_y40-10'+perturb, 'RdBu_r')
