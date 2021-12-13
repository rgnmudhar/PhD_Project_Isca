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
import cartopy.crs as ccrs 
from eofs.xarray import Eof

def open(files, dim):
    paths = sorted(glob(files))
    datasets = [xr.open_dataset(p, decode_times=False) for p in paths]
    combined = xr.concat(datasets, dim)
    return combined


files = '../isca_data/Polvani_Kushner_4.0_eps0_1y/run*/atmos_daily_interp_new_height_temp.nc'
ds = open(files, 'time')

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
anom_solver = Eof(u_anom, weights=wgts)

# Retrieve the leading EOF, expressed as the covariance between the leading PC time series and the input anomalies at each grid point
anom_eof1 = anom_solver.eofsAsCovariance(neofs=1)

"""
# Plot the leading EOF expressed as covariance in the European/Atlantic domain
proj = ccrs.Orthographic(central_longitude=-20, central_latitude=60)
ax = plt.axes(projection=proj)
ax.coastlines()
ax.set_global()
anom_eof1[0].plot.contourf(ax=ax, cmap='RdBu_r', levels=np.linspace(-7,7,14), transform=ccrs.PlateCarree(), add_colorbar=False)
ax.set_title('EOF1 of Zonal Wind Anomaly at ~{0:.0f}hPa'.format(p_level))
plt.show()
"""

# Plot the leading EOF expressed as covariance over the globe
ax = plt.axes(projection=ccrs.PlateCarree())
plt.contourf(lon,lat,anom_eof1[0], cmap='RdBu_r', levels=np.linspace(-7,7,14), transform=ccrs.PlateCarree())
ax.coastlines()
#plt.xlabel('Longitude')
#plt.ylabel('Latitude')
plt.title('EOF1 of Zonal Wind Anomaly at ~{0:.0f}hPa'.format(p_level))
plt.show()