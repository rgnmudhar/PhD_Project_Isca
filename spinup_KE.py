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

path = str(sys.argv[1]) 
exp_name = str(sys.argv[2])
files=sorted(glob(path+'/'+exp_name+'/run*/*.nc'))
KE = []
print("finding KE")
for i in range(len(files)):
    print(i)
    file  = files[i]
    ds = xr.open_dataset(file, decode_times=False)
    u = ds.ucomp.mean(dim='lon')
    v = ds.vcomp.mean(dim='lon')
    coslat = np.cos(np.deg2rad(u.coords['lat'].values)).clip(0., 1.) # need to weight due to different box sizes over grid
    lat_wgts = np.sqrt(coslat)
    for j in range(len(u)):
        print(j)
        TKE_box = np.empty_like(u[j])
        for q in range(len(ds.coords['pfull'].data)):
            for k in range(len(ds.coords['lat'].data)):
                TKE_box[q,k] = calc_TKE(u[j][q,k], v[j][q,k])
        TKE_box = np.average(TKE_box, axis=1, weights=lat_wgts)
        TKE_avg = np.nanmean(TKE_box) # should I weight pressures too? How?
        KE.append(TKE_avg)
save_file(exp_name, KE, 'KE')

# option to plot the KE over time
#print("plotting")
#KE = open_file(exp_name, 'KE')
#fig, ax = plt.subplots(figsize=(10,6))
#ax.plot(len(KE), KE, color='k')
#ax.set_xlim(1,len(KE))
#ax.set_xlabel("Days Simulated")       
#ax.set_ylabel('TKE', color='k')
#plt.title("Total Kinetic Energy for "+exp_name)
#plt.savefig(exp_name+'_spinup.pdf', bbox_inches = 'tight')
#plt.close()
