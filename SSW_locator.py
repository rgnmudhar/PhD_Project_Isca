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

def calc_error(nevents, ndays):
    """
    Returns the 95% error interval assuming a binomial distribution:
    e.g. http://www.sigmazone.com/binomial_confidence_interval.htm
    From W. Seviour
    """
    p = nevents / float(ndays)
    e = 1.96 * np.sqrt(p * (1 - p) / ndays)
    return e

def find_SPV(files):
    SPV = []
    SPV_flag = []
    for i in range(len(files)):
        file = files[i]
        ds = xr.open_dataset(file, decode_times = False)
        u = ds.ucomp.mean(dim='lon').sel(pfull=10, method='nearest').sel(lat=60, method='nearest')
        for j in range(len(u)):
            SPV.append(u[j].data)
            if u[j] < 0:
                SPV_flag.append(True)
            elif u[j] >= 0:
                SPV_flag.append(False)
    return SPV, SPV_flag

def find_SSW(files):
    winds, flag = find_SPV(files)

    count = 0

    for k in range(len(winds)):
        if winds[k] < 0:
            if winds[k-1] > 0:
                subset = flag[k-20:k]
                if True not in subset:
                    count += 1

    SSW_freq = count / len(winds)

    return len(winds), count, SSW_freq

if __name__ == '__main__':  
    file = 'PK_e0v4z13_q6m2y45l800u200_52y'
    time = 'daily'
    years = 2
    file_suffix = '_interp'
    files = discard_spinup2(file, time, file_suffix, years)

    days, SSWs, freq = find_SSW(files)
    err = calc_error(SSWs, days)

    print('{0:.0f} SSWs in {1:.0f} days ({2:.3f} ± {3:.3f}% of the time)'.format(SSWs, days, freq*100, err*100))