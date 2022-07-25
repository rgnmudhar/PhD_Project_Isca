"""
    For creating a plot that shows SPV speed and AM timescale for various experiments
"""

import os
from glob import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from eofs.xarray import Eof
import statsmodels.api as sm
from shared_functions import *
from EOFs import *

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
    lags = 50 
    tau1, tau2 = AM_times(pcs, lags)

    return tau1

if __name__ == '__main__':
    indir = '/disco/share/rm811/processed/'
    exp = ['PK_e0v1z13', 'PK_e0v2z13', 'PK_e0v3z13', 'PK_e0v4z13',\
        'PK_e0v1z18', 'PK_e0v2z18', 'PK_e0v3z18', 'PK_e0v4z18']

    print(datetime.now(), " - finding SPV and tau values")
    tau = []
    vtx = []
    for i in range(len(exp)):
        ds = add_phalf(indir+exp[i], '_uz.nc')
        uz = ds.ucomp
        tau.append(return_tau(uz))
        SPV = open_file(exp[i], 'SPV')
        vtx.append(np.mean(SPV))
    
    print(datetime.now(), " - plotting")
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(tau, vtx)
    ax.set_xlabel(r'EOF1 $\tau$ (days)', fontsize='x-large')
    ax.set_ylabel(r'10 hPa, 60 N Zonal Wind Mean (ms$^{-1}$)', fontsize='x-large')
    ax.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    plt.legend()
    plt.savefig('vtx_vs_tau.pdf', bbox_inches = 'tight')
    plt.close()
