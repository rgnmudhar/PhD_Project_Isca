"""
Uses jet_locator functions to find location and strength of maximum stratopsheric vortex (10 hPa).
Then plots this against heating experiment.
"""

from glob import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from shared_functions import *
from jet_locator import *

def vtx_vs_heat(files, heat, p, xlabel, fig_name):    
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
    ax.errorbar(heat, maxwinds, yerr=maxwinds_sd, fmt='o', linewidth=1.25, capsize=5, color='#C0392B', linestyle=':')
    ax.set_xticks(heat)
    #ax.set_xlabel(r'Strength of Heating (K day$^{-1}$)', fontsize='large')
    ax.set_xlabel(xlabel, fontsize='large')
    ax.set_ylabel(r'Max. SPV Speed (ms$^{-1}$)', color='#C0392B', fontsize='large')
    ax.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
    ax2 = ax.twinx()
    ax2.errorbar(heat, lats, yerr=lats_sd, fmt='o', linewidth=1.25, capsize=5, color='#2980B9', linestyle=':')
    ax2.set_ylabel(r'Max. SPV Latitude ($\degree$N)', color='#2980B9', fontsize='large')
    ax2.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
    plt.title(r'Max. NH SPV Strength and Location at $p \sim 10$ hPa', fontsize='x-large')
    plt.savefig(fig_name+'_SPVvheat.png', bbox_inches = 'tight')

    return plt.close()

if __name__ == '__main__': 
    #Set-up data to be read in
    basis = 'PK_eps0_vtx1_zoz13'
    experiments = [basis+'_7y',\
        basis+'_w15a0.5p800f800g50',\
        basis+'_w15a2p800f800g50',\
        basis+'_w15a4p800f800g50',\
        basis+'_w15a6p800f800g50',\
        basis+'_w15a8p800f800g50']
    heat = [0, 0.5, 2, 4, 6, 8]
    xlabel = r'Strength of Heating (K day$^{-1}$)'
    #heat = ['no heat', '900', '800', '700', '600', '500']
    #xlabel = r'p$_{top}$ (hPa)'
    time = 'daily'
    years = 2 # user sets no. of years worth of data to ignore due to spin-up
    file_suffix = '_interp'
    p = 10 # pressure level at which we want to find the SPV (hPa)
    
    files = []
    for i in range(len(experiments)):
        files.append(discard_spinup2(experiments[i], time, file_suffix, years))
        """
        if i == 0:
            files.append(discard_spinup2(experiments[i], time, file_suffix, 0))
        else:
            files.append(discard_spinup2(experiments[i], time, file_suffix, years))
        """

    vtx_vs_heat(files, heat, p, xlabel, basis)  
