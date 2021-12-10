"""
    This script attempts to use KE at a certain pressure level for determining spin-up. 
    Based on code by Penelope Maher.
"""

import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

def calc_TKE(u,v):
    u = bar(u)
    v = bar(v)
    upv = u*u + v*v
    upv_zonal = upv.mean(dim='lon')
    return 0.5 * upv_zonal

def bar(u):
    return u.mean(dim='time')

files=sorted(glob.glob('../isca_data/Polvani_Kushner_4.0_eps0_6y/run*/atmos_monthly_interp_new_height_temp.nc'))
iter = np.arange(0,len(files))
KE = []
for i in iter:
    #print(i)
    file  = files[i]
    ds = xr.open_dataset(file, decode_times=False)
    lat = ds.coords['lat'].data
    p = ds.coords['pfull'].data
    u = ds.ucomp[:,len(p)-2,:,:]
    v = ds.vcomp[:,len(p)-2,:,:]
    TKE = calc_TKE(u,v)
    coslat = np.cos(np.deg2rad(u.coords['lat'].values)).clip(0., 1.) # need to weight due to different box sizes over grid
    wgts = np.sqrt(coslat)
    TKE_avg = np.average(TKE, weights=wgts)
    KE.append(TKE_avg)

fig, ax = plt.subplots(figsize=(11,6))
ax.plot(iter+1, KE, color='k')
ax.set_xlim(1,len(files))
ax.set_xlabel("Run no.")       
ax.set_ylabel('TKE', color='k')
plt.title("Total Kinetic Energy at ~{0:.0f}hPa for ~{1:.0f}y worth".format(p[len(p)-2],len(files)/12))
plt.show()