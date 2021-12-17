"""
    Computes and plots the leading EOF of zonal wind on the 500 hPa pressure surface during winter time.
    Written for individual pressure level to get an EOF which is a function of latitude and longitude to plot on a map.
    u’(lat,lon,time) is used to calculate the EOF, where u’(lat,lon,time) = u(lat,lon,time) - ubar(lon), with ubar = zonal and time average.
    Based on https://ajdawson.github.io/eofs/latest/examples/nao_xarray.html and discussions with William Seviour.
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

files = '../isca_data/Polvani_Kushner_4.0_eps0_1y/run*/atmos_daily_interp_new_height_temp.nc'
ds = open(files, 'time')

t = ds.coords['time']
times=[]
for i in range(len(t)):
    times.append(t[i]-t[0])

lat = ds.coords['lat'].data
lon = ds.coords['lon'].data
p = ds.coords['pfull'].data
p_level = 500 # hPa

u = ds.ucomp # zonal wind
u_level = u.sel(pfull=p_level, method='nearest')
u_anom = u_level - u_level.mean(dim='time').mean(dim='lat') # zonal wind anomalies 

coslat = np.cos(np.deg2rad(u.coords['lat'].values)).clip(0., 1.) # need to weight due to different box sizes over grid
wgts = np.sqrt(coslat)[...,np.newaxis]


# Create an EOF solver to do the EOF analysis.
solver = Eof(u_anom, weights=wgts)

# Retrieve the leading EOF, expressed as the covariance between the leading PC time series and the input anomalies at each grid point
eof1 = solver.eofsAsCovariance(neofs=1)

# Plot the leading EOF expressed as covariance over the globe
fig1 = plt.figure()
plt.contourf(lon,lat,eof1[0], cmap='RdBu_r', levels=np.linspace(-7,7,14))
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('EOF1 of Zonal Wind Anomaly at ~{0:.0f}hPa'.format(p_level))
plt.show()

# Find the leading EOF, expressed as the correlation between the leading PC time series and the u anomalies at each grid point
# and the leading PC time series itself
eof1_corr = solver.eofsAsCorrelation(neofs=1)
pc1 = solver.pcs(npcs=1, pcscaling=1)

"""
pcs = solver.pcs()
eofs = solver.eofs()
"""

fig2 = plt.figure()
plt.plot(times, pc1[:, 0], linewidth=2)
ax = plt.gca()
ax.axhline(0, color='k')
ax.set_xlim(0, max(times))
ax.set_xlabel('Days')
ax.set_ylabel('Normalized Units')
ax.set_title('PC1 Time Series', fontsize='x-large')
plt.show()

fig3 = plt.figure()
plt.contourf(lon,lat,eof1_corr[0], cmap='RdBu_r', levels=np.linspace(-0.75,0.75,15))
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('EOF1 of Zonal Wind Anomaly expressed as correlation at ~{0:.0f}hPa'.format(p_level))
cb = plt.colorbar(orientation='horizontal')
cb.set_label('correlation coefficient', fontsize='large')
plt.show()