"""
Uses jet_locator functions to find location and strength of the tropospheric jet (850 hPa).
Then plots this against heating experiment.
"""

from glob import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from shared_functions import *
from jet_locator import *

def jet_vs_heat(files, heat, p, xlabel, fig_name):    
    lats = []
    lats_sd = []
    maxwinds = []
    maxwinds_sd = []
    for i in range(len(files)):
        ulat, umax = jet_timeseries(files[i], np.arange(0,len(files[i])), p)
        lats.append(np.mean(ulat))
        maxwinds.append(np.mean(umax))
        lats_sd.append(np.std(ulat))
        maxwinds_sd.append(np.std(umax))

    fig, ax = plt.subplots(figsize=(12,8))
    ax.errorbar(heat, maxwinds[0:3], yerr=maxwinds_sd[0:3], fmt='o', linewidth=1.25, capsize=5, color='#C0392B', linestyle=':', label=r'$\gamma = 0$')
    ax.errorbar(heat, maxwinds[3:6], yerr=maxwinds_sd[3:6], fmt='o', linewidth=1.25, capsize=5, color='#C0392B', linestyle='--', label=r'$\gamma = 1$')
    ax.set_xticks(heat)
    #ax.set_xlabel(r'Strength of Heating (K day$^{-1}$)', fontsize='large')
    ax.set_xlabel(xlabel, fontsize='large')
    ax.set_ylabel(r'Max. Jet Speed (ms$^{-1}$)', color='#C0392B', fontsize='large')
    ax.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
    ax2 = ax.twinx()
    ax2.errorbar(heat, lats[0:3], yerr=lats_sd[0:3], fmt='o', linewidth=1.25, capsize=5, color='#2980B9', linestyle=':', label=r'$\gamma = 0$')
    ax2.errorbar(heat, lats[3:6], yerr=lats_sd[3:6], fmt='o', linewidth=1.25, capsize=5, color='#2980B9', linestyle='--', label=r'$\gamma = 1$')
    ax2.set_ylabel(r'Max. Jet Latitude ($\degree$N)', color='#2980B9', fontsize='large')
    ax2.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
    plt.legend(loc='upper right' , fancybox=False, shadow=True, ncol=1, fontsize='large')
    plt.title(r'Max. NH Jet Strength and Location at $p \sim 850$ hPa', fontsize='x-large')
    plt.savefig(fig_name+'_SPVvheat.png', bbox_inches = 'tight')

    return plt.close()

if __name__ == '__main__': 
    #Set-up data to be read in
    basis1 = 'PK_eps0_vtx0_zoz13'
    basis2 = 'PK_eps0_vtx1_zoz13'
    experiments = [basis1+'_7y',\
        basis1+'_w15a2p400f800g50',\
        basis1+'_w15a4p400f800g50',\
        basis2+'_7y',\
        basis2+'_w15a2p400f800g50',\
        basis2+'_w15a4p400f800g50']
    heat = [0, 2, 4]
    xlabel = r'Strength of Heating (K day$^{-1}$)'
    time = 'daily'
    years = 2 # user sets no. of years worth of data to ignore due to spin-up
    file_suffix = '_interp'
    p = 850 #10 # pressure level at which we want to find the SPV (hPa)
    
    files = []
    for i in range(len(experiments)):
        files.append(discard_spinup2(experiments[i], time, file_suffix, years))
        """
        if i == 0:
            files.append(discard_spinup2(experiments[i], time, file_suffix, 0))
        else:
            files.append(discard_spinup2(experiments[i], time, file_suffix, years))
        """

    jet_vs_heat(files, heat, p, xlabel, "ruth_experiments") #basis)  
