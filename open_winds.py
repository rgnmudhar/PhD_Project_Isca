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

def calc_jet_lat_quad(u, lats, p, plot=False):
    """
    Function for finding location and strength of maximum given zonal wind u(lat) field.
    Based on Will Seviour code.
    """
    print("finding location and strength of maximum of zonal wind ")
    # Restrict to 3 points around maximum
    u_new = u.mean(dim='lon').sel(pfull=p, method='nearest')
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

def jet_timeseries(file, p):
    """
    Steps through each dataset to find jet latitude/maximum over time.
    Amended to only look at NH tropospheric jet.
    """
    print(datetime.now(), " - finding jet maxima over time")
    jet_maxima = []
    jet_lats = []
    ds = xr.open_dataset(file, decode_times=False)
    lat = ds.coords['lat'].data
    u = ds.ucomp
    # restrict to NH:
    u = u[:,:,int(len(lat)/2):len(lat),:]
    lat = lat[int(len(lat)/2):len(lat)] 
    for i in range(len(u)):
        # find and store jet maxima and latitudes for each month
        jet_lat, jet_max = calc_jet_lat_quad(u[i], lat, p)
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
    plt.savefig(fig_name+'_winds.pdf', bbox_inches = 'tight')
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
    plt.savefig(fig_name+'_maxlat.pdf', bbox_inches = 'tight')
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
    plt.savefig(fig_name+'_maxwind.pdf', bbox_inches = 'tight')
    
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
    plt.savefig(fig_name+'_jetvexp.pdf', bbox_inches = 'tight')

    return plt.close()

def save_file(exp, var, input):
    textfile = open('../Files/'+exp+'_'+input+'.txt', 'w')
    if isinstance(var, list):
        l = var
    else:
        l = var.to_numpy().tolist()
    for j in l:
        textfile.write(str(j) + '\n')
    return textfile.close()

def open_file(exp, input):
    textfile = open('../Files/'+exp+'_'+input+'.txt', 'r')
    list = textfile.read().replace('\n', ' ').split(' ')
    list = list[:len(list)-1]
    textfile.close()
    list = np.asarray([float(j) for j in list])
    return list

def winds_errs(exp, p, name):
    """
    Uses jet_locator functions to find location and strength of the tropospheric jet (850 hPa) or SPV (10 hPa).
    """
    print(datetime.now(), " - calculating standard deviation errors")
    maxlats = []
    maxwinds = []
    maxlats_sd = []
    maxwinds_sd = []
    for i in range(len(exp)):
        files = discard_spinup2(exp[i], time, file_suffix, years)
        lats = []
        maxs = []
        for j in range(len(files)):
            lat, max = jet_timeseries(files[j], p)
            lats.append(lat)
            maxs.append(max)
        maxlats.append(np.mean(lats))
        maxwinds.append(np.mean(maxs))
        maxlats_sd.append(np.std(lats)/np.sqrt(len(files)*30))
        maxwinds_sd.append(np.std(maxs)/np.sqrt(len(files)*30))
    save_file(name, maxlats, 'maxlats')
    save_file(name, maxwinds, 'maxwinds')
    save_file(name, maxlats_sd, 'maxlats_sd')
    save_file(name, maxwinds_sd, 'maxwinds_sd')

def find_SPV(exp):
    """
    Steps through each dataset to find vortex strength over time.
    Uses 60N and 10hPa as per SSW definiton.
    Saves as a file.
    """
    print(datetime.now(), " - finding wind speeds at 60N, 10hPa")
    for i in range(len(exp)):
        print(datetime.now(), " - ", exp[i])
        files = discard_spinup2(exp[i], time, file_suffix, years)
        ds = xr.open_mfdataset(files, decode_times=False)
        SPV = ds.ucomp.mean(dim='lon').sel(pfull=10, method='nearest').sel(lat=60, method='nearest')
        save_file(exp[i], SPV, 'SPV')

def plot_vtx(exp, labels, colors, style, cols, fig_name):
    """
    Plots strength of winds at 60N, 10 hPa only.
    Best for 2 datsets, the second of which has its SSW statistics as a plot subtitle
    """
    
    print(datetime.now(), " - plotting SPV")
    fig, ax = plt.subplots(figsize=(6,6))
    for i in range(len(exp)):
        SPV = open_file(exp[i], 'SPV')
        ax.plot(SPV, color=colors[i], linewidth=1, linestyle=style[i], label=labels[i])
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlim(1,len(SPV))
    ax.set_xlabel('Months Simulated', fontsize='x-large')       
    ax.set_ylabel(r'Zonal Wind (ms$^{-1}$)', color='k', fontsize='x-large')
    ax.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    ax.set_xticks([10*(12*30), 20*(12*30), 30*(12*30), 40*(12*30)], [10*12, 20*12, 30*12, 40*12])
    plt.legend(loc='upper center' , bbox_to_anchor=(0.5, 0.11), fancybox=False, shadow=False, ncol=cols, fontsize='x-large')
    plt.savefig(fig_name+'_vtx.pdf', bbox_inches = 'tight')
    
    return plt.close()

def SPVvexp(exp, exp_names, xlabel, name):
    """
    Uses jet_locator functions to find location and strength of maximum stratopsheric vortex (10 hPa).
    Then plots this against (heating) experiment.
    """
    print(datetime.now(), " - plotting SPV mean and variance v experiment")
    mean = []
    err = []
    sd = []
    for i in range(len(exp)):
        SPV = open_file(exp[i], 'SPV')
        mean.append(np.mean(SPV))
        err.append(np.std(SPV/np.sqrt(len(SPV)))) 
        sd.append(np.std(SPV))
    fig, ax = plt.subplots(figsize=(10,6))
    ax.errorbar(exp_names[1:], mean[1:], yerr=err[1:], fmt='o', linewidth=1.25, capsize=5, color='#B30000', linestyle=':')
    ax.set_xticks(exp_names)
    ax.set_xlabel(xlabel, fontsize='x-large')
    ax.set_ylim(30, 35)
    ax.set_ylabel(r'10 hPa, 60 N Zonal Wind Mean (ms$^{-1}$)', fontsize='x-large', color='#B30000')
    ax.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    ax2 = ax.twinx()
    ax2.plot(exp_names[1:], sd[1:], marker='o', linewidth=1.25, color='#4D0099', linestyle=':')
    ax2.set_ylim(13,19)
    ax2.set_ylabel(r'10 hPa, 60 N Zonal Wind S.D.(ms$^{-1}$)', color='#4D0099', fontsize='x-large')
    ax2.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    #plt.title(r'Max. NH SPV Strength and Location at $p \sim 10$ hPa, 60N', fontsize='xx-large')
    plt.savefig(name+'_SPVvheat.pdf', bbox_inches = 'tight')

    return plt.close()

def vtxvexp(exp, xlabel, name):
    """
    Uses jet_locator functions to find location and strength of maximum stratopsheric vortex (10 hPa).
    Then plots this against (heating) experiment.
    """
    print(datetime.now(), " - plotting SPV maxima v experiment")
    fig, ax = plt.subplots(figsize=(10,6))
    ax.errorbar(exp, open_file(name, 'maxwinds')[1:], yerr=open_file(name, 'maxwinds_sd')[1:], fmt='o', linewidth=1.25, capsize=5, color='#B30000', linestyle=':')
    ax.set_xticks(exp)
    ax.set_ylim(51,56)
    ax.set_xlabel(xlabel, fontsize='x-large')
    ax.set_ylabel(r'Strength of Max. 10 hPa Zonal Wind (ms$^{-1}$)', color='#B30000', fontsize='x-large')
    ax.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    ax2 = ax.twinx()
    ax2.errorbar(exp, open_file(name, 'maxlats')[1:], yerr=open_file(name, 'maxlats_sd')[1:], fmt='o', linewidth=1.25, capsize=5, color='#4D0099', linestyle=':')
    ax2.set_yticks(np.arange(69,71.5,0.5))
    ax2.set_ylabel(r'Laitude of Max. 10 hPa Zonal Wind ($\degree$N)', color='#4D0099', fontsize='x-large')
    ax2.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    #plt.title(r'Max. NH SPV Strength and Location at $p \sim 10$ hPa', fontsize='xx-large')
    plt.savefig(name+'_SPVvheat.pdf', bbox_inches = 'tight')

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

def find_SSW(SPV):
    #finding SSWs
    print(datetime.now(), " - finding SSWs")
    SPV_flag = np.select([SPV<0, SPV>0], [True, False], True)
    days = len(SPV)
    count = 0
    for k in range(days):
        if SPV[k] < 0:
            if SPV[k-1] > 0:
                subset = SPV_flag[k-20:k]
                if True not in subset:
                    count += 1
    #winters = (12/4) * (days/360)
    #SSWs_w = (count/winters)
    #SSWs_w_err = calc_error(count, winters)
    SSWs_h = (count/days)*100
    SSWs_h_err = calc_error(count, days)*100
    return SSWs_h, SSWs_h_err

def find_SSWs(exp):
    h_list = []
    h_err_list = []
    for i in range(len(exp)):
        SPV = open_file(exp[i], 'SPV')
        h, h_err = find_SSW(SPV)
        h_list.append(h)
        h_err_list.append(h_err)
    return h_list, h_err_list

def SSWsperrun(exp):
    SPV = open_file(exp, 'SPV')
    years = np.arange(5, int(len(SPV)/360)+5, 5)
    SSWs = []
    errors = []
    for n in years:
        h, h_err = find_SSW(SPV[:int(n*360)])
        SSWs.append(h)
        errors.append(h_err)
    return SSWs, errors, years

def SSWsvexp(exp, x, xlabel, fig_name):
    """
    Plots SSW frequency against (heating) experiment.
    """
    SSWs, errors = find_SSWs(exp)
    og = SSWs[0]
    og_err = errors[0]
    print(SSWs, errors)
    print(datetime.now(), " - plotting SSWs v experiment")
    fig, ax = plt.subplots(figsize=(10,6))
    ax.errorbar(x, SSWs[1:], yerr=errors[1:], fmt='o', linewidth=1.25, capsize=5, color='#B30000', linestyle=':')
    ax.set_xlim(-0.5,6.5)
    ax.set_xticks(x)
    ax.set_xlabel(xlabel, fontsize='x-large')
    ax.set_ylabel(r'SSWs per 100 days', fontsize='x-large')
    ax.axhline(0.42, color='#4D0099', linewidth=0.5)
    ax.text(5.4, 0.425, 'ERA-Interim', color='#4D0099', fontsize='x-large')
    ax.axhline(og, color='#666666', linewidth=0.5)
    ax.fill_between(range(-1,8), (og - og_err), (og + og_err), facecolor ='gainsboro', alpha = 0.4)
    ax.text(5.75, 0.335, 'Control', color='#666666', fontsize='x-large')
    ax.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    #plt.title(r'SSW Frequency', fontsize='xx-large')
    plt.savefig(fig_name+'_SSWsvheat.pdf', bbox_inches = 'tight')

    return plt.close()

def SSWsvexp2(exp1, exp2, x1, xlabel1, x2, xlabel2, fig_name):
    """
    Plots SSW frequency against (heating) experiment for 2 types of experiment.
    """
    SSWs1, errors1 = find_SSWs(exp1)
    SSWs2, errors2 = find_SSWs(exp2)
    og = SSWs1[0]
    og_err = errors1[0]
    
    print(datetime.now(), " - plotting SSWs v experiment")
    fig, ax = plt.subplots(figsize=(10,6))
    ax.errorbar(x1, SSWs1[1:], yerr=errors1[1:], fmt='o', linewidth=1.25, capsize=5, color='#B30000', linestyle=':')
    ax.set_xlim(-0.5,6.5)
    ax.set_xticks(x1)
    ax.set_xlabel(xlabel1, fontsize='x-large', color='#B30000')
    ax.set_ylabel(r'SSWs per 100 days', fontsize='x-large')
    ax.axhline(0.42, color='#0099CC', linewidth=0.5)
    ax.text(5.4, 0.425, 'ERA-Interim', color='#0099CC', fontsize='x-large')
    ax.axhline(og, color='#666666', linewidth=0.5)
    ax.fill_between(range(-1,8), (og - og_err), (og + og_err), facecolor ='gainsboro', alpha = 0.4)
    ax.text(5.75, 0.335, 'Control', color='#666666', fontsize='x-large')
    ax.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    ax2 = ax.twiny()
    ax2.errorbar(x2, SSWs2[1:], yerr=errors2[1:], fmt='o', linewidth=1.25, capsize=5, color='#4D0099', linestyle=':')
    ax2.set_xlim(-0.5,6.5)
    ax2.set_xticks(x2)
    ax2.set_xlabel(xlabel2, fontsize='x-large', color='#4D0099')
    ax2.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    #plt.title(r'SSW Frequency', fontsize='xx-large')
    plt.savefig(fig_name+'_SSWsvheat2.pdf', bbox_inches = 'tight')

    return plt.close()

def SSWsvruntime(exp, colors, labels, fig_name):
    """
    Plots SSW frequency for each (heating) experiment vs. run time.
    """

    print(datetime.now(), " - plotting SSWs v run length")
    fig, ax = plt.subplots(figsize=(10,6))
    for i in range(len(exp)):
        SSWs, errors, years = SSWsperrun(exp[i])
        ax.errorbar(years, SSWs, yerr=errors, fmt='o', linewidth=1.25, capsize=5, color=colors[i], linestyle=':', label=labels[i])
    ax.set_xticks(years)
    ax.set_xlabel('Years Run', fontsize='x-large')
    ax.set_ylabel(r'SSWs per 100 days', fontsize='x-large')
    ax.axhline(0.42, color='k', linewidth=1, label='ERA-Interim')
    ax.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    plt.legend(loc='upper right',  fancybox=False, shadow=True, ncol=3, fontsize='large')
    plt.title(r'SSW Frequency', fontsize='xx-large')
    plt.savefig(fig_name+'_SSWsvrun.pdf', bbox_inches = 'tight')

    return plt.close()

if __name__ == '__main__': 
    #User set
    time = 'daily'
    years = 2 # user sets no. of years worth of data to ignore due to spin-up
    file_suffix = '_interp'

    #Set-up data to be read in
    basis = 'PK_e0v4z13'
    exp = [basis+'_q6m2y45l800u200',\
        #basis+'_w10a4p800f800g50_q6m2y45l800u200',\
        #basis+'_w15a4p800f800g50_q6m2y45l800u200',\
        #basis+'_w20a4p800f800g50_q6m2y45l800u200',\
        #basis+'_w25a4p800f800g50_q6m2y45l800u200',\
        #basis+'_w30a4p800f800g50_q6m2y45l800u200',\
        #basis+'_w35a4p800f800g50_q6m2y45l800u200',\
        #basis+'_w40a4p800f800g50_q6m2y45l800u200']
    #exp2 = [basis+'_q6m2y45l800u200',\
        basis+'_w15a4p900f800g50_q6m2y45l800u200',\
        basis+'_w15a4p800f800g50_q6m2y45l800u200',\
        basis+'_w15a4p700f800g50_q6m2y45l800u200',\
        basis+'_w15a4p600f800g50_q6m2y45l800u200',\
        basis+'_w15a4p500f800g50_q6m2y45l800u200',\
        basis+'_w15a4p400f800g50_q6m2y45l800u200',\
       basis+'_w15a4p300f800g50_q6m2y45l800u200']     
    
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
        labels = ['no asymmetry', 'asymmetry']
        colors = ['#0099CC', '#B30000']
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
        plot_type = input('Plot a) SPV @ 10hPa, 60N over time or b) SPV max. and lat for different experiments, c) SSWs for different experiments?')
        if plot_type == 'a':
            #find_SPV(exp)
            plot_vtx(exp, labels, colors, style, cols, basis)
        elif plot_type == 'b':
            #winds_errs(exp, p, basis+'_extent')
            SPVvexp(exp, ['control', '900', '800', '700', '600', '500', '400', '300'], 'Depth of Heating (hPa)', basis+'_depth')
            #vtxvexp(['10', '15', '20', '25', '30', '35', '40'], r'Extent of Heating ($\degree$)', basis+'_extent')
        elif plot_type == 'c':
            SSWsvexp(exp, ['10', '15', '20', '25', '30', '35', '40'], r'Extent of Heating ($\degree$)', basis)
            #SSWsvexp2(exp, exp2, ['10', '15', '20', '25', '30', '35', '40'], r'Extent of Heating ($\degree$)', ['900', '800', '700', '600', '500', '400', '300'], 'Depth of Heating (hPa)', basis)
            #SSWsvruntime(exp, ['#B30000', '#FF9900', '#FFCC00', '#00B300', '#0099CC', '#4D0099', '#CC0080', '#666666'],\
            #['0 hPa', '900 hPa', '800 hPa', '700 hPa', '600 hPa', '500 hPa', '400 hPa', '300 hPa'], basis)
