"""
    Computes and plots the leading EOF of zonal wind as per Fig. 4 of Sheshadri and Plumb (2017)
    u anomaly is used to calculate the EOF, where u’(lat,p,time) = u(lat,p,time) - ubar(p), with ubar = zonal and time average.
    Based on Sheshadri and Plumb (2017), https://ajdawson.github.io/eofs/latest/examples/nao_xarray.html and discussions with William Seviour.
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

t = ds.coords['time']
times=[]
for i in range(len(t)):
    times.append(t[i]-t[0])

lat = ds.coords['lat']
lon = ds.coords['lon']
p = ds.coords['pfull']
p_half = ds.coords['phalf']

u = ds.ucomp.mean(dim='lon') # zonal-mean zonal wind
p_min = 100 # hPa
lat_min = 20 # degrees
u_subset = u.sel(pfull=slice(p_min,max(p)), lat=slice(lat_min,max(lat))) # restrict analysis to window used in Sheshadri and Plumb (2017)
u_anom = u_subset - u_subset.mean(dim='time').mean(dim='lat') # zonal-mean zonal wind anomalies 

lat_subset = lat.sel(lat=slice(lat_min,max(lat))) 
coslat = np.cos(np.deg2rad(u_subset.coords['lat'].values)).clip(0., 1.) # need to weight due to different box sizes over grid
p_subset = p.sel(pfull=slice(p_min,max(p)))
index_min = len(p_half) - len(p_subset)
p_half_subset = p_half[index_min-1:]
dp = np.empty_like(p_subset)

for i in range(len(dp)):
    dp[i] = p_half_subset[i+1] - p_half_subset[i] # find pressure thickness of each model level

"""
for i in range(len(coslat)):
    for j in range(len(dp)):
        u_anom[:,j,i] = u_anom[:,j,i]*np.sqrt(dp[j]*coslat[i]) # weight u anomalies using sqrt(dp*cos(lat))
# print(u_anom)
"""
# Need to weight due to different box sizes over grid and pressure level thicknesses
da = xr.DataArray(data=u_anom, dims=['time', 'dp', 'coslat'], coords=[t, np.sqrt(dp), np.sqrt(coslat)])
wgts = da[0,:,:].to_numpy()

# Create an EOF solver to do the EOF analysis.
solver = Eof(u_anom, weights=wgts)

# Retrieve the leading EOF, expressed as the covariance between the leading PC time series and the input anomalies at each grid point
# By default PCs used are scaled to unit variance (dvided by square root of the eigenvalue)
eof1 = solver.eofsAsCovariance(neofs=1)

for i in range(len(coslat)):
    for j in range(len(dp)):
        eof1[:,j,i] = eof1[:,j,i]/np.sqrt(dp[j]*coslat[i]) # retrieve EOF1 

# Plot the leading EOF expressed as covariance
fig1 = plt.figure(figsize=(5,6))
plt.contourf(lat_subset,p_subset,eof1[0], cmap='RdBu_r', levels=np.linspace(-2,2,21))
plt.xlabel('Latitude')
plt.ylabel('Pressure (hPa)')
plt.ylim(max(p),p_min)
cb = plt.colorbar(orientation='horizontal')
plt.title('EOF1 of Zonal Wind Anomaly')
plt.show()

# Find the leading PC time series itself
pc1 = solver.pcs(npcs=1, pcscaling=1) # first PC scaled to unit variance

fig2 = plt.figure(figsize=(6,5))
plt.plot(times, pc1[:, 0], linewidth=2)
ax = plt.gca()
ax.axhline(0, color='k')
ax.set_xlim(0, max(times))
ax.set_xlabel('Days')
ax.set_ylabel('Normalized Units')
ax.set_title('PC1 Time Series', fontsize='x-large')
plt.show()

# Fractional EOF mode variances.
# The fraction of the total variance explained by each EOF mode, values between 0 and 1 inclusive.
variance_fractions = solver.varianceFraction()
print(variance_fractions[0].data * 100) # Is this the % of zonal wind variance associated w/ EOF1?