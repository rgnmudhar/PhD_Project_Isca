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
    err = []
    sd = []
    for i in range(len(exp)):
        SPV = open_file(exp[i], 'SPV')
        mean.append(np.mean(SPV))
        err.append(np.std(SPV/np.sqrt(len(SPV))))
        sd.append(np.std(SPV))
    print(mean)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.errorbar(exp_names[1:], mean[1:], yerr=err[1:], fmt='o', linewidth=1.25, capsize=5, color='#B30000', linestyle=':')
    ax.set_xticks(exp_names)
    ax.set_xlabel(xlabel, fontsize='x-large')
    ax.set_ylabel(r'10 hPa, 60 N Zonal Wind Mean (ms$^{-1}$)', fontsize='x-large', color='#B30000')
    ax.set_ylim(32,44)
    ax.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    ax2 = ax.twinx()
    ax2.plot(exp_names[1:], sd[1:], marker='o', linewidth=1.25, color='#4D0099', linestyle=':')
    ax2.set_ylabel(r'10 hPa, 60 N Zonal Wind S.D. (ms$^{-1}$)', color='#4D0099', fontsize='x-large')
    ax2.set_ylim(12,22)
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
    ax.set_xlim(-0.5,6.5)
    ax.set_xticks(x[1:])
    ax.set_xlabel(xlabel, fontsize='x-large')
    ax.set_ylabel(r'SSWs per 100 days', fontsize='x-large')
    ax.axhline(0.42, color='#4D0099', linewidth=0.5)
    ax.text(5.4, 0.425, 'ERA-Interim', color='#4D0099', fontsize='x-large')
    ax.axhline(og, color='#666666', linewidth=0.5)
    ax.fill_between(range(-1,8), (og - og_err), (og + og_err), facecolor ='gainsboro', alpha = 0.4)
    ax.text(5.75, 0.335, 'Control', color='#666666', fontsize='x-large')
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

def pdf(x):
    x = np.sort(x)
    s = np.std(x)
    m = np.mean(x)
    f = (1 / (s * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - m)/s)**2)
    return x, f

def skew(x):
    x = np.sort(x)
    s = np.std(x)
    m = np.mean(x)
    N = len(x)
    z = sum((x - m)**3) / ((N - 1) * s**3)
    return z

def plot_pdf(exp, labels, colors, name):
    print(datetime.now(), " - plotting SPV pdfs")
    x_min = x_max = 0 
    fig, ax = plt.subplots(figsize=(8,6))
    for i in range(len(exp)):
        SPV = open_file(exp[i], 'SPV')
        x, f = pdf(SPV)
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

def plot_stats(exp, labels, colors):
    x_min = x_max = 0
    fig, ax = plt.subplots(figsize=(8,6))
    for i in range(len(exp)):
        SPV = open_file(exp[i], 'SPV')
        x, f = pdf(SPV)
        if max(x) > x_max:
            x_max = max(x)
        if min(x) < x_min:
            x_min = min(x)
        ax.plot(x, f, linewidth=1.25, color=colors[i], linestyle=style[i], label=labels[i])
    ax.set_xlim(x_min, x_max)
    ax.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    plt.legend(loc='upper right',fancybox=False, shadow=True, ncol=1, fontsize='large')
    return plt.show()

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
    ax.text(-0.2, 0.02, 'Control', color='#B30000', fontsize='x-large')
    ax.set_ylim(-0.5, 0.5)
    ax.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    ax2 = ax.twinx()
    ax2.plot(exp_names[1:], kurt[1:], marker='o', linewidth=1.25, color='#4D0099', linestyle=':')
    ax2.set_ylabel(r'10 hPa, 60 N Zonal Wind Kurtosis', color='#4D0099', fontsize='x-large')
    ax2.axhline(kurt[0], color='#4D0099', linewidth=0.5)
    ax2.text(5.5, -0.57, 'Control', color='#4D0099', fontsize='x-large')
    ax2.set_xlim(-0.5, 6.5)
    ax2.set_ylim(-0.9, 0.9)
    ax2.fill_between(range(-1,8), -1, 0, facecolor ='gainsboro', alpha = 0.4)
    ax2.text(2, -0.3, 'Negative Skew, Lighter Tails', color='#666666', fontsize='x-large')
    ax2.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    plt.savefig(name+'_SPVstats.pdf', bbox_inches = 'tight')
    return plt.close()

if __name__ == '__main__': 
    #Set-up data to be read in
    indir = '/disco/share/rm811/processed/'
    basis = 'PK_e0v4z13'
    extension = '_depth'
    if extension == '_depth':
        exp = [basis+'_q6m2y45l800u200',\
        basis+'_w15a4p900f800g50_q6m2y45l800u200',\
        basis+'_w15a4p800f800g50_q6m2y45l800u200',\
        basis+'_w15a4p700f800g50_q6m2y45l800u200',\
        basis+'_w15a4p600f800g50_q6m2y45l800u200',\
        basis+'_w15a4p500f800g50_q6m2y45l800u200',\
        basis+'_w15a4p400f800g50_q6m2y45l800u200',\
        basis+'_w15a4p300f800g50_q6m2y45l800u200']
        labels = ['no heat', '900', '800', '700', '600', '500', '400', '300']
        xlabel = 'Depth of Heating (hPa)'
    elif extension == '_width':
        exp = [basis+'_q6m2y45l800u200',\
        basis+'_w10a4p800f800g50_q6m2y45l800u200',\
        basis+'_w15a4p800f800g50_q6m2y45l800u200',\
        basis+'_w20a4p800f800g50_q6m2y45l800u200',\
        basis+'_w25a4p800f800g50_q6m2y45l800u200',\
        basis+'_w30a4p800f800g50_q6m2y45l800u200',\
        basis+'_w35a4p800f800g50_q6m2y45l800u200',\
        basis+'_w40a4p800f800g50_q6m2y45l800u200']
        labels = ['no heat', '10', '15', '20', '25', '30', '35', '40']
        xlabel = r'Extent of Heating ($\degree$)'
    elif extension == '_strengthp800':    
        exp = [basis+'_q6m2y45l800u200',\
        basis+'_w15a2p800f800g50_q6m2y45l800u200',\
        basis+'_w15a4p800f800g50_q6m2y45l800u200',\
        basis+'_w15a6p800f800g50_q6m2y45l800u200',\
        basis+'_w15a8p800f800g50_q6m2y45l800u200']
        labels = ['no heat', '2', '4', '6', '8'] #['no heat', '800', '600', '400']
        xlabel = r'Strength of Heating (K day$^{-1}$)'
    
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
            c) SPV mean and variance vs experiment, or \
            d) SSW frequency vs experiment, or \
            e) distribution and stats?')
        if plot_type == 'a':
            exp = exp[:2]
            labels = ['no asymmetry', 'asymmetry']
            colors = ['#0099CC', '#B30000']
            style = ['-', '-']
            cols = 2
            plot_vtx(exp, labels, colors, style, cols, exp[0])
        elif plot_type == 'b':
            windsvexp(labels, xlabel, str(p), basis+extension)
        elif plot_type == 'c':
            SPVvexp1(exp, labels, xlabel, basis+extension)
        elif plot_type == 'd':
            #SSWsvexp(exp, labels, xlabel, basis+extension)
            SSWsvexp_multi(exp, labels, xlabel, legend, ['#B30000', '#00B300', '#0099CC', 'k'], basis+extension)
        elif plot_type == 'e':
            plot_pdf(exp, labels, colors, basis+extension)
            SPVvexp2(exp, labels, xlabel, basis+extension)
