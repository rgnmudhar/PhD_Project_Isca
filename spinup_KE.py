"""
    This script attempts to use TKE for determining spin-up.
    For every 'time'  (i.e. atmos_monthly file) take the zonal mean of u and v along longitude.
    Then at each (lat,p) find 0.5*(u^2 + v^2).
    Then take a weighted average along lat, and finally a mean along pressure.
    Based on script by Penelope Maher and discussions with William Seviour.
"""
import sys
from glob import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

def calc_TKE(u,v):
    upv = u*u + v*v
    return 0.5 * upv

indir = '/disco/share/rm811/processed/'
exp = 'PK_e0v4z13_q6m2y45l800u200'
KE = []
print("finding KE")
uz = xr.open_dataset(indir+exp+'_u.nc', decode_times=False).ucomp[0].mean(dim='lon')
vz = xr.open_dataset(indir+exp+'_v.nc', decode_times=False).vcomp[0].mean(dim='lon')
coslat = np.cos(np.deg2rad(uz.coords['lat'].values)).clip(0., 1.) # need to weight due to different box sizes over grid
lat_wgts = np.sqrt(coslat)
for j in range(len(uz)):
    TKE_box = np.empty_like(uz[j])
    for q in range(len(uz.coords['pfull'].data)):
        for k in range(len(uz.coords['lat'].data)):
            TKE_box[q,k] = calc_TKE(uz[j][q,k], vz[j][q,k])
    TKE_box = np.average(TKE_box, axis=1, weights=lat_wgts)
    TKE_avg = np.nanmean(TKE_box) # should I weight pressures too? How?
    KE.append(TKE_avg)
save_file(exp_name, KE, 'KE')

# option to plot the KE over time
plot = False
if plot == True:
    print("plotting")
    KE = open_file(exp_name, 'KE')
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(len(KE), KE, color='k')
    ax.set_xlim(1,len(KE))
    ax.set_xlabel("Days Simulated")       
    ax.set_ylabel('TKE', color='k')
    plt.title("Total Kinetic Energy for "+exp_name)
    plt.savefig(exp_name+'_spinup.pdf', bbox_inches = 'tight')
    plt.close()
