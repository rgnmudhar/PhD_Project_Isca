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

file = 'PK_eps0_vtx3_zoz13_h4000m2l25u65'
time = 'daily'
years = 2
file_suffix = '_interp'
files = discard_spinup2(file, time, file_suffix, years)

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

count = 0

for k in range(len(SPV)):
    if SPV[k] < 0:
        if SPV[k-1] > 0:
            SPV_subset = SPV_flag[k-20:k]
            if True not in SPV_subset:
                count += 1

SSW_freq = count / len(SPV)

print('{0:.0f} SSWs in {1:.0f} days ({2:.2f}% of the time)'.format(count, len(SPV), SSW_freq*100))