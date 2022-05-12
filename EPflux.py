"""
    Computes and plots EP flux vectors and divergence terms.
    Based on Martin Jucker's code at https://github.com/mjucker/aostools/blob/d857987222f45a131963a9d101da0e96474dca63/climate.py
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from shared_functions import *
from aostools import climate

def calc_ep(ds):
    t = ds.temp
    u = ds.ucomp
    v = ds.vcomp
    ep1, ep2, div1, div2 = climate.ComputeEPfluxDivXr(u, v, t, 'lon', 'lat', 'pfull', 'time')
    # take time mean of relevant quantities
    div = div1 + div2
    div = div.mean(dim='time')
    ep1 = ep1.mean(dim='time')
    ep2 = ep2.mean(dim='time')
    uz = u.mean(dim='time').mean(dim='lon')
    return uz, div, ep1, ep2

#Set-up data to be read in
exp = 'PK_e0v4z13' #_q6m2y45l800u200' #_w15a4p800f800g50'
time = 'daily'
years = 2 # user sets no. of years worth of data to ignore due to spin-up
file_suffix = '_interp'
files = discard_spinup2(exp, time, file_suffix, years)
ds = xr.open_mfdataset(files, decode_times=False)
p = ds.coords['pfull'].data
lat = ds.coords['lat'].data
uz, div, ep1, ep2 = calc_ep(ds)

#Filled contour plot of time-mean EP flux divergence plus EP flux arrows and zonal wind contours
divlvls = 21 #np.arange(-2,2.1,0.1)
ulvls = np.arange(-200, 200, 10)
fig, ax = plt.subplots(figsize=(8,6))
uz.plot.contour(colors='k', linewidths=0.5, alpha=0.4, levels=ulvls)
div_scaled = div #(div.transpose()/p).transpose()
cs = div_scaled.plot.contourf(levels=divlvls, cmap='RdBu_r', add_colorbar=False)
cb = plt.colorbar(cs)
#cb.set_label(label=r'Pressure-scaled Divergence (m s$^{-1}$ day$^{-1}$)', size='large')
cb.set_label(label=r'Divergence (m s$^{-1}$ day$^{-1}$)', size='large')
ticklabs = cb.ax.get_yticklabels()
cb.ax.set_yticklabels(ticklabs, fontsize='large')
climate.PlotEPfluxArrows(ds.lat,ds.pfull,ep1,ep2,fig,ax,yscale='log')
plt.xlabel('Latitude', fontsize='x-large')
plt.xlim(-90,90)
plt.xticks([-90, -45, 0, 45, 90], ['90S', '45S', '0', '45N', '90N'])
plt.ylabel('Pressure (hPa)', fontsize='x-large')
plt.yscale('log')
plt.ylim(max(p), 1) #to 1 hPa
plt.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
plt.title('Time and Zonal Mean EP Flux', fontsize='x-large')
plt.savefig(exp+'_EPflux.pdf', bbox_inches = 'tight')
plt.close()