"""
    This script attempts to use TKE for determining spin-up.
    For every 'time'  (i.e. atmos_monthly file) take the zonal mean of u and v along longitude.
    Then at each (lat,p) find 0.5*(u^2 + v^2).
    Then take a weighted average along lat, and finally a mean along pressure.
    Based on script by Penelope Maher and discussions with William Seviour.
"""

from glob import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

def calc_TKE(u,v):
    upv = u*u + v*v
    return 0.5 * upv

exp_name = 'PK_eps0_vtx3_zoz13_w15a8p800f800g50'
files=sorted(glob('../isca_data/'+exp_name+'/run*/atmos_daily_interp.nc'))
iter = np.arange(0,len(files))
KE = []
for i in iter:
    #print(i)
    file  = files[i]
    ds = xr.open_dataset(file, decode_times=False)
    lat = ds.coords['lat'].data
    lon = ds.coords['lon'].data
    p = ds.coords['pfull'].data
    u = ds.ucomp.mean(dim='lon').mean(dim='time')
    v = ds.vcomp.mean(dim='lon').mean(dim='time')
    coslat = np.cos(np.deg2rad(u.coords['lat'].values)).clip(0., 1.) # need to weight due to different box sizes over grid
    lat_wgts = np.sqrt(coslat)
    TKE_box = np.empty_like(u)
    for q in range(len(p)):
        for j in range(len(lat)):
            TKE_box[q,j] = calc_TKE(u[q,j], v[q,j])
    
    TKE_box = np.average(TKE_box, axis=1, weights=lat_wgts)
    TKE_avg = np.nanmean(TKE_box) # should I weight pressures too? How?
    KE.append(TKE_avg)

fig, ax = plt.subplots(figsize=(11,6))
ax.plot(iter+1, KE, color='k')
ax.set_xlim(1,len(files))
ax.set_xlabel("Run no.")       
ax.set_ylabel('TKE', color='k')
plt.title("Total Kinetic Energy for "+exp_name) #~{0:.0f}y worth".format(len(files)/12))
plt.show()
