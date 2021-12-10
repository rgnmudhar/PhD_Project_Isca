"""
    This script attempts to find EOFs from daily data.
    Re-write this to do separately for each pressure level to get an EOF which is a function of latitude and longitude, so you can plot a map. It should hopefully look annular!
    Start with 500hPa.
"""

from glob import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from eofs.xarray import Eof

def open(files, dim):
    paths = sorted(glob(files))
    datasets = [xr.open_dataset(p, decode_times=False) for p in paths]
    combined = xr.concat(datasets, dim)
    return combined

files = '../isca_data/Polvani_Kushner_4.0_eps0_1y/run*/atmos_daily.nc'
ds = open(files, 'time')

lat = ds.coords['lat'].data
p = ds.coords['pfull'].data
upper_p = ds.coords['pfull'].sel(pfull=1, method='nearest') # in order to cap plots at pressure = 1hPa

u = ds.ucomp # zonal wind
u_anom = u - u.mean(dim='time').mean(dim='lon') # zonal wind anomalies 
""" u’(lat,lon,time) to calculate the EOF, where u’(lat,lon,time) = u(lat,lon,time) - ubar(lon) so ubar is the zonal and time average"""

coslat = np.cos(np.deg2rad(u.coords['lat'].values)).clip(0., 1.) # need to weight due to different box sizes over grid
wgts = np.sqrt(coslat)[...,np.newaxis]

full_solver = Eof(u, weights=wgts)
full_eof1 = full_solver.eofsAsCovariance(neofs=1)

anom_solver = Eof(u_anom, weights=wgts)
anom_eof1 = anom_solver.eofsAsCovariance(neofs=1)

plt.contourf(lat,p,anom_eof1[0,:,:,:].mean(dim='lon'), cmap='RdBu_r', levels=21)
plt.colorbar()
plt.xlabel('Latitude')
plt.ylabel('Pressure (hPa)')
plt.ylim(max(p), upper_p) #goes to 1hPa
plt.yscale("log")
plt.title('Zonal mean EOF1 of Zonal Wind Anomaly')
plt.show()

plt.pause(1)

plt.contourf(lat,p,full_eof1[0,:,:,:].mean(dim='lon'), cmap='RdBu_r', levels=21)
plt.colorbar()
plt.xlabel('Latitude')
plt.ylabel('Pressure (hPa)')
plt.ylim(max(p), upper_p) #goes to 1hPa
plt.yscale("log")
plt.title('Zonal mean EOF1 of Zonal Wind')
plt.show()