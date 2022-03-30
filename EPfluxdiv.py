"""
    Computes and plots EP flux vectors and divergence terms.
    Based on Martin Jucker's code at https://github.com/mjucker/aostools/blob/d857987222f45a131963a9d101da0e96474dca63/climate.py
"""

from glob import glob
import xarray as xr
import matplotlib.pyplot as plt
from shared_functions import *
from aostools import climate

#Set-up data to be read in
files = 'PK_e0v1z13'
time = 'daily'
years = 2 # user sets no. of years worth of data to ignore due to spin-up
file_suffix = '_interp'
files = discard_spinup2(files, time, file_suffix, years)
ds = xr.open_mfdataset(files, decode_times=False)
lat = ds.lat
lon = ds.lon
p = ds.pfull
t = ds.temp
u = ds.ucomp
v = ds.vcomp
	
ep1, ep2, div1, div2 = climate.ComputeEPfluxDivXr(u, v, t, 'lon', 'lat', 'pfull', 'time')
div = div1.mean(dim='time') + div2.mean(dim='time')

#Filled contour plot of time-mean EP flux divergence plus EP flux arrows
fig, ax = plt.subplots(figsize=(10,8))
cs = div.plot.contourf(levels=25, cmap='RdBu_r', add_colorbar=False)
plt.colorbar(cs, label=r'Divergence (m s$^{-1}$ day$^{-1}$)')
plt.xlabel('Latitude', fontsize='x-large')
#plt.xlim(-90,90)
#plt.xticks([-90, -45, 0, 45, 90], ['90S', '45S', '0', '45N', '90N'])
plt.ylabel('Pressure (hPa)', fontsize='x-large')
plt.yscale('log')
climate.PlotEPfluxArrows(lat,p,ep1.mean(dim='time'),ep2.mean(dim='time'),fig,ax, yscale='log')
plt.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
plt.show()