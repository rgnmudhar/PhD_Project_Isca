"""
Script for functions involving winds - near-surface, tropospheric jet and stratospheric polar vortex.
"""

from glob import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import cftime
from datetime import datetime
from shared_functions import *

def winds_errs(files, p):
    """
    Uses jet_locator functions to find location and strength of the tropospheric jet (850 hPa).
    """
    print("calculating standard deviation errors")
    lats = []
    lats_sd = []
    maxs = []
    maxs_sd = []
    for i in range(len(files)):
        lat, max = jet_timeseries(files[i], np.arange(0,len(files[i])), p)
        lats.append(np.mean(lat))
        maxs.append(np.mean(max))
        lats_sd.append(np.std(lat))
        maxs_sd.append(np.std(max))
    return lats, lats_sd, maxs, maxs_sd

def calc_jet_lat_quad(u, lats, p, plot=False):
    """
    Function for finding location and strength of maximum given zonal wind u(lat) field.
    Based on Will Seviour code.
    """
    print("finding location and strength of maximum of zonal wind ")
    # Restrict to 3 points around maximum
    u_new = u.mean(dim='time').mean(dim='lon').sel(pfull=p, method='nearest')
    u_max = np.where(u_new == np.ma.max(u_new))[0][0]
    u_near = u_new[u_max-1:u_max+2]
    lats_near = lats[u_max-1:u_max+2]
    # Quadratic fit, with smaller lat spacing
    coefs = np.ma.polyfit(lats_near,u_near,2)
    fine_lats = np.linspace(lats_near[0], lats_near[-1],200)
    quad = coefs[2]+coefs[1]*fine_lats+coefs[0]*fine_lats**2
    # Find jet lat and max
    jet_lat = fine_lats[np.where(quad == max(quad))[0][0]]
    jet_max = coefs[2]+coefs[1]*jet_lat+coefs[0]*jet_lat**2
    # Plot fit?
    if plot:
        print(jet_max)
        print(jet_lat)
        plt.plot(lats_near, u_near)
        plt.plot(fine_lats, quad)
        plt.xlabel("Latitude")
        plt.ylabel("Wind Speed")
        plt.title("Jet Latitude at p={:.0f}hPa".format(p))
        plt.text(jet_lat-2, jet_max-1, "Jet max = {0:.2f}m/s at {1:.2f} deg latitude".format(jet_max, jet_lat))
        plt.show()
 
    return jet_lat, jet_max

def open_ra():
    """
    Opens ERA5 re-analysis data if required.
    """
    print("opening re-analysis data")
    #Following opens ERA5 re-analysis data
    file_ra = '/disca/share/pm366/ERA-5/era5_var131_masked_zm.nc'
    ds_ra = nc.Dataset(file_ra)
    t_ra = ds_ra.variables['time']
    lev = ds_ra.variables['lev'][:].data
    p_ra = lev/100 # convert to hPa
    lat_ra = ds_ra.variables['lat'][:].data
    u_ra = ds_ra.variables['ucomp']

    #Following writes date/time of ERA-5 data
    #times = []
    #times.append(cftime.num2pydate(t_ra, t_ra.units, t_ra.calendar)) # convert to actual dates
    #times = np.array(times)

    return lat_ra, u_ra, p_ra

def jet_timeseries(files, iter, p):
    """
    Steps through each dataset to find jet latitude/maximum over time.
    Amended to only look at NH tropospheric jet.
    """
    print("finding jet maxima over time")
    jet_maxima = []
    jet_lats = []
    for i in iter:
        file  = files[i]
        ds = xr.open_dataset(file, decode_times=False)
        lat = ds.coords['lat'].data
        u = ds.ucomp
        # restrict to NH:
        u = u[:,:,int(len(lat)/2):len(lat),:]
        lat = lat[int(len(lat)/2):len(lat)] 
        # find and store jet maxima and latitudes for each month
        jet_lat, jet_max = calc_jet_lat_quad(u, lat, p)
        jet_maxima.append(jet_max)
        jet_lats.append(jet_lat)

    return jet_lats, jet_maxima

def plot_winds(ds, labels, colors, style, cols, fig_name, X, ra):
    """
    Sets up all necessary coordinates and variables for re-analysis and simulation data.
    Then plots time and zonal average zonal wind at the pressure level closest to XhPa
    """
    
    #Following sets up variables from the Isca datasets
    lat = ds[0].coords['lat'].data
    lon = ds[0].coords['lon'].data
    p = ds[0].coords['pfull'].data
    uz = []
    for i in range(len(ds)):
        uz.append(ds[i].ucomp.mean(dim='time').mean(dim='lon').sel(pfull=X, method='nearest'))

    print("plotting near-surface winds")
    #Following plots the data and saves as a figure
    fig = plt.subplots(1,1, figsize=(10,8))
    plt.axhline(0, color='#D2D0D3', linewidth=0.5)
    for i in range(len(ds)):
        plt.plot(lat, uz[i], color=colors[i], linestyle=style[i], label=labels[i])   
    if ra == True:
        # plot re-analysis uwind against latitude average over Nov-19 to Feb-20 (NH winter) if required
        lat_ra, u_ra, p_ra = open_ra()
        plt.plot(lat_ra, np.mean(u_ra[491:494,np.where(p_ra == X)[0],:], axis=0)[0], color='#27AE60', label='ERA5 (DJF)')     
    plt.xlabel('Latitude', fontsize='large')
    plt.xlim(-90,90)
    plt.xticks([-90, -45, 0, 45, 90], ['90S', '45S', '0', '45N', '90N'])
    plt.ylabel(r'Wind Speed (ms$^{-1}$)', fontsize='large')
    plt.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
    plt.legend(loc='upper center' , bbox_to_anchor=(0.5, -0.07),fancybox=False, shadow=True, ncol=cols, fontsize='large')
    plt.title('Mean Zonal Winds at ~{:.0f}hPa'.format(X), fontsize='x-large')
    plt.savefig(fig_name+'_winds.png', bbox_inches = 'tight')
    return plt.close()

def plot_jet(files, p, labels, colors, style, cols, wind, fig_name):
    """
    Plots latitude and corresponding strength of maximum winds at some input
    """
    
    iter = np.arange(0,len(files[0]))
    lats = []
    maxs = []
    for i in range(len(files)):
        for j in range(len(files[i])):
            lat, max = jet_timeseries(files[i], iter, p)
            lats.append(lat)
            maxs.append(max)

    print("plotting tropospheric jet")
    fig, ax = plt.subplots(figsize=(12,8))
    for i in range(len(files)):
        ax.plot(iter+1, lats[i], color=colors[i], linewidth=1, linestyle=style[i], label=labels[i])
    ax.set_xlim(1,len(files[0]))
    ax.set_xlabel('Month', fontsize='large')       
    ax.set_ylabel(wind+r'(Latitude ($\degree$N)', fontsize='large')
    ax.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
    plt.legend(loc='upper center' , bbox_to_anchor=(0.5, -0.07), fancybox=False, shadow=True, ncol=cols, fontsize='large')
    plt.title('NH '+wind+'Latitude at p ~{0:.0f} hPa'.format(p), fontsize='x-large')
    plt.savefig(fig_name+'_maxlat.png', bbox_inches = 'tight')
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(12,8))
    for i in range(len(files)):
        ax.plot(iter+1, maxs[i], color=colors[i], linewidth=1, linestyle=style[i], label=labels[i])
    ax2.set_xlim(1,len(files[0]))
    ax2.set_xlabel('Month', fontsize='large')       
    ax2.set_ylabel(wind+r'Max (ms$^{-1}$)', color='k', fontsize='large')
    ax2.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
    plt.legend(loc='upper center' , bbox_to_anchor=(0.5, -0.07), fancybox=False, shadow=True, ncol=cols, fontsize='large')
    plt.title('NH '+wind+'Strength at p ~{0:.0f} hPa'.format(p), fontsize='x-large')
    plt.savefig(fig_name+'_maxwind.png', bbox_inches = 'tight')
    
    return plt.close()

def jetvexp(files, exp, p, xlabel, fig_name):
    """
    Uses jet_locator functions to find location and strength of the tropospheric jet (850 hPa).
    Then plots this against (heating) experiment.
    """
    lats, lats_sd, maxwinds, maxwinds_sd = winds_errs(files, p)

    fig, ax = plt.subplots(figsize=(12,8))
    ax.errorbar(exp, maxwinds[0:3], yerr=maxwinds_sd[0:3], fmt='o', linewidth=1.25, capsize=5, color='#C0392B', linestyle=':', label=r'$\gamma = 0$')
    ax.errorbar(exp, maxwinds[3:6], yerr=maxwinds_sd[3:6], fmt='o', linewidth=1.25, capsize=5, color='#C0392B', linestyle='--', label=r'$\gamma = 1$')
    ax.set_xticks(exp)
    ax.set_xlabel(xlabel, fontsize='large')
    ax.set_ylabel(r'Max. Jet Speed (ms$^{-1}$)', color='#C0392B', fontsize='large')
    ax.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
    ax2 = ax.twinx()
    ax2.errorbar(exp, lats[0:3], yerr=lats_sd[0:3], fmt='o', linewidth=1.25, capsize=5, color='#2980B9', linestyle=':', label=r'$\gamma = 0$')
    ax2.errorbar(exp, lats[3:6], yerr=lats_sd[3:6], fmt='o', linewidth=1.25, capsize=5, color='#2980B9', linestyle='--', label=r'$\gamma = 1$')
    ax2.set_ylabel(r'Max. Jet Latitude ($\degree$N)', color='#2980B9', fontsize='large')
    ax2.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
    plt.legend(loc='upper right' , fancybox=False, shadow=True, ncol=1, fontsize='large')
    plt.title(r'Max. NH Jet Strength and Location at $p \sim 850$ hPa', fontsize='x-large')
    plt.savefig(fig_name+'_jetvexp.png', bbox_inches = 'tight')

    return plt.close()

def calc_error(nevents, nyears):
    """
    For the SSW frequency finder from Will Seviour
    Returns the 95% error interval assuming a binomial distribution:
    e.g. http://www.sigmazone.com/binomial_confidence_interval.htm
    """
    p = nevents / float(nyears)
    e = 1.96 * np.sqrt(p * (1 - p) / nyears)
    return e

def find_SPV(exp):
    """
    Steps through each dataset to find vortex strength over time.
    Uses 60N and 10hPa as per SSW definiton.
    Also finds SSW statistics.
    """
    print("finding wind speeds at 60N, 10hPa")
    for i in range(len(exp)):
        files = discard_spinup2(exp[i], time, file_suffix, years)
        ds = xr.open_mfdataset(files, decode_times=False)
        SPV = ds.ucomp.mean(dim='lon').sel(pfull=10, method='nearest').sel(lat=60, method='nearest')
        textfile = open(exp[i]+'_SPV.txt', 'w')
        SPV_list = SPV.to_numpy().tolist()
        for j in SPV_list:
            textfile.write(str(j) + '\n')
        textfile.close()

def find_SSWs(SPV):
    #finding SSWs
    SPV_flag = np.select([SPV<0, SPV>0], [True, False], True)
    days = len(SPV)
    count = 0
    for k in range(days):
        if SPV[k] < 0:
            if SPV[k-1] > 0:
                subset = SPV_flag[k-20:k]
                if True not in subset:
                    count += 1
    winters = (12/4) * (days/360)
    SSWs_w = '{0:.2f} SSWs per winter (± {1:.2f})'.format((count/winters),calc_error(count, winters))
    SSWs_h = '{0:.2f} SSWs per 100 days (± {1:.2f})'.format((count/days)*100,calc_error(count, days)*100)
    print(SSWs_w, ', ', SSWs_h)

def plot_vtx(exp, SSWs, labels, colors, style, cols, fig_name):
    """
    Plots strength of winds at 60N, 10 hPa only.
    Best for 2 datsets, the second of which has its SSW statistics as a plot subtitle
    """
    
    print(datetime.now(), " - plotting SPV")
    fig, ax = plt.subplots(figsize=(10,6))
    for i in range(len(exp)):
        textfile = open(exp[i]+'_SPV.txt', 'r')
        SPV = textfile.read().replace('\n', ' ').split(' ')
        SPV = SPV[:len(SPV)-1]
        textfile.close()
        SPV = np.asarray([float(j) for j in SPV])
        if SSWs == True:
            find_SSWs(SPV)
        ax.plot(SPV, color=colors[i], linewidth=1, linestyle=style[i], label=labels[i])
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlim(1,len(SPV))
    ax.set_xlabel('Day', fontsize='x-large')       
    ax.set_ylabel(r'Zonal Wind Speed (ms$^{-1}$)', color='k', fontsize='x-large')
    ax.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    plt.legend(loc='upper center' , bbox_to_anchor=(0.5, -0.1), fancybox=False, shadow=True, ncol=cols, fontsize='large')
    #plt.suptitle(r'Vortex Strength at $p \sim 10$ hPa, $\theta \sim 60 \degree$N', fontsize='xx-large')
    plt.title('With Polar Heating', fontsize='x-large')
    plt.savefig(fig_name+'_vtx3.png', bbox_inches = 'tight')
    
    return plt.close()

def vtxvexp(files, exp, p, xlabel, fig_name):
    """
    Uses jet_locator functions to find location and strength of maximum stratopsheric vortex (10 hPa).
    Then plots this against (heating) experiment.
    """
    lats, lats_sd, maxwinds, maxwinds_sd = winds_errs(files, p)

    print("plotting SPV maxima over time")
    fig, ax = plt.subplots(figsize=(12,8))
    ax.errorbar(exp, maxwinds, yerr=maxwinds_sd, fmt='o', linewidth=1.25, capsize=5, color='#C0392B', linestyle=':')
    ax.set_xticks(exp)
    ax.set_xlabel(xlabel, fontsize='large')
    ax.set_ylabel(r'Max. SPV Speed (ms$^{-1}$)', color='#C0392B', fontsize='large')
    ax.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
    ax2 = ax.twinx()
    ax2.errorbar(exp, lats, yerr=lats_sd, fmt='o', linewidth=1.25, capsize=5, color='#2980B9', linestyle=':')
    ax2.set_ylabel(r'Max. SPV Latitude ($\degree$N)', color='#2980B9', fontsize='large')
    ax2.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
    plt.title(r'Max. NH SPV Strength and Location at $p \sim 10$ hPa', fontsize='x-large')
    plt.savefig(fig_name+'_SPVvheat.png', bbox_inches = 'tight')

    return plt.close()

if __name__ == '__main__': 
    #User set
    time = 'daily'
    years = 40 # user sets no. of years worth of data to ignore due to spin-up
    file_suffix = '_interp'

    #Set-up data to be read in
    basis = 'PK_e0v4z13'
    exp = [basis,\
        basis+'_q6m2y45l800u200']        
    
    #User choices for plotting - aesthetics
    label_type = input(r'Plot a) different $\gamma$, b) different $\epsilon, z_{oz}$, c) heat or d) diff. vs. original?')
    if label_type == 'a':
        labels = [r'$\gamma$ = 1',r'$\gamma$ = 2',r'$\gamma$ = 3',r'$\gamma$ = 4']
        colors = ['k', '#C0392B', '#27AE60', '#9B59B6']
        style = ['-', '-', '-', '-']
        cols = 4
        ra = True
    elif label_type == 'b':
        labels = [r'$\epsilon = 0, p_{trop} \sim 100$ hPa', r'$\epsilon = 10, p_{trop} \sim 100$ hPa', r'$\epsilon = 0, p_{trop} \sim 200$ hPa', r'$\epsilon = 10, p_{trop} \sim 200$ hPa']
        colors = ['#2980B9', '#2980B9', 'k', 'k']
        style = [':', '-', ':', '-']
        cols = 2
        ra = True
    elif label_type == 'c':
        labels = ['no heat', r'A = 2 K day$^{-1}$', r'A = 4 K day$^{-1}$', r'A = 8 K day$^{-1}$']
        colors = ['k', 'k', 'k', 'k']
        style = ['-', '--', '-.', ':']
        cols = 4
        ra = False
    elif label_type == 'd':
        labels = ['original', 'heating perturbation']
        colors = ['#2980B9', '#C0392B']
        style = ['-', '-']
        cols = 2
        ra = False

    #User choices for plotting - subjects
    level = input('Plot a) near-surface winds, b) tropospheric jet, c) stratospheric polar vortex?')
    if level == 'a':
        p = 900 #hPa
        plot_winds(discard_spinup1(exp, time, file_suffix, years),\
            labels, colors, style, cols, basis, p, ra)
    elif level == 'b':
        p = 850 # pressure level at which we want to find the jet (hPa)
        wind = 'Jet '
        plot_type = input('Plot a) jet max. and lat over time or b) jet max. and lat for different experiments?')
        if plot_type == 'a':
            plot_jet(discard_spinup2(exp, time, file_suffix, years),\
                p, labels, colors, style, cols, wind, basis)
        if plot_type == 'b':
            jetvexp(discard_spinup2(exp, time, file_suffix, years),\
                [0, 2, 4], p, r'Strength of Heating (K day$^{-1}$)', basis)
    elif level == 'c':
        p = 10 # pressure level at which we want to find the SPV (hPa)
        wind = 'SPV '
        plot_type = input('Plot a) SPV @ 10hPa, 60N over time or b) SPV max. and lat for different experiments?')
        if plot_type == 'a':
            find_SPV(exp)
            #plot_vtx(exp, True, labels, colors, style, cols, basis)
        elif plot_type == 'b':
            vtxvexp(discard_spinup2(exp, time, file_suffix, years),\
                [0, 0.5, 2, 4, 6, 8], p, r'Strength of Heating (K day$^{-1}$)', basis)
