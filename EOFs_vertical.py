"""
    Computes and plots the leading EOF of zonal wind as per Fig. 4 of Sheshadri and Plumb (2017)
    u anomaly is used to calculate the EOF, where u’(lat,p,time) = u(lat,p,time) - ubar(p), with ubar = zonal and time average.
    Based on Sheshadri and Plumb (2017), https://ajdawson.github.io/eofs/latest/examples/nao_xarray.html and code by William Seviour.
"""

from glob import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from eofs.xarray import Eof
import statsmodels.api as sm

def findtau(ac):
    for i in range(len(ac)):
        if ac[i] - 1/np.e < 1e-5:
            tau = i
            break
    return tau

files = sorted(glob('../isca_data/Polvani_Kushner_4.0_eps0_1y/run*/atmos_daily.nc'))
ds = xr.open_mfdataset(files, decode_times=False)

lat = ds.coords['lat']
lon = ds.coords['lon']

# For EOFs follow Sheshadri & Plumb 2017, use p>100hPa, lat>20degN, zonal mean u
# also throw out first 100 days for spin-up
p_min = 100  # hPa
lat_min = 20  # degrees
spin_up = 100  # days
uz = ds.ucomp\
       .isel(time=slice(spin_up,-1))\
       .sel(pfull=slice(p_min,1000))\
       .sel(lat=slice(lat_min,90))\
       .mean(dim='lon')

# Calculate anomalies
uz_anom = uz - uz.mean(dim='time')

# sqrt(cos(lat)) weights due to different box sizes over grid
sqrtcoslat = np.sqrt(np.cos(np.deg2rad(uz_anom.coords['lat'].values))) 

# sqrt(dp) weights, select correct number of levels
nplevs = uz_anom.coords['pfull'].shape[0]
sqrtdp = np.sqrt(np.diff(ds.coords['phalf'].values[-nplevs-2:-1]))

# calculate gridpoint weights
wgts = np.outer(sqrtdp,sqrtcoslat)

# Create an EOF solver to do the EOF analysis.
solver = Eof(uz_anom.compute(), weights=wgts)

# Retrieve the leading 2 EOFs and leading PC time series
# expressed  as the covariance between the leading PC time series and the input anomalies
# By default PCs used are scaled to unit variance (dvided by square root of the eigenvalue)
eofs = solver.eofsAsCovariance(neofs=2)
pcs = solver.pcs(npcs=2, pcscaling=1)

# Fractional EOF mode variances.
# The fraction of the total variance explained by each EOF mode, values between 0 and 1 inclusive.
variance_fractions = solver.varianceFraction()

# Find autocorrelation function in order to determine decorrelation time (i.e. time for correlation reduce to 1/e)
lags = 50 
ac1 = sm.tsa.acf(pcs.sel(mode=0).values, nlags=lags)
ac2 = sm.tsa.acf(pcs.sel(mode=1).values, nlags=lags)

tau1 = findtau(ac1)
tau2 = findtau(ac2)

# Plot equivalent to Fig 4a-c from Sheshadri & Plumb
fig = plt.figure(figsize=(15,5))

ax1 = fig.add_subplot(1,3,1)
eofs.sel(mode=0).plot.contourf(ax=ax1, cmap='RdBu_r', yincrease=False, levels=21)
uz.mean(dim='time').plot.contour(ax=ax1, colors='k', yincrease=False, levels=15)
ax1.set_title(r'EOF1, {0:.0f}% of variance, $\tau$ = {1:.0f} days'\
    .format(100*variance_fractions.values[0],\
    tau1))

ax2 = fig.add_subplot(1,3,2)
eofs.sel(mode=1).plot.contourf(ax=ax2, cmap='RdBu_r', yincrease=False, levels=21)
uz.mean(dim='time').plot.contour(ax=ax2, colors='k', yincrease=False, levels=15)
ax2.set_title(r'EOF2, {0:.0f}% of variance, $\tau$ = {1:.0f} days'\
    .format(100*variance_fractions.values[1],\
    tau2))

ax3 = fig.add_subplot(1,3,3)
ax3.acorr(pcs.sel(mode=0), maxlags=lags, usevlines=False, linestyle="-", marker=None, linewidth=1, label='EOF1')
ax3.acorr(pcs.sel(mode=1), maxlags=lags, usevlines=False, linestyle="-", marker=None, linewidth=1, label='EOF2')
ax3.legend()
ax3.axhline(0, color='k', linewidth=1)
ax3.axhline(1/np.e, color='k', linestyle=":")
ax3.set_xlabel('Lag (days)')
ax3.set_title('PC autocorrelation')

plt.tight_layout()
plt.show()

"""
# Plot the PC timeseries
t = ds.coords['time']
times=[]
for i in range(len(t)):
    times.append(t[i]-t[0])

fig2 = plt.figure(figsize=(6,5))
plt.plot(times, pcs[0, 0], linewidth=2)
ax = plt.gca()
ax.axhline(0, color='k')
ax.set_xlim(0, max(times))
ax.set_xlabel('Days')
ax.set_ylabel('Normalized Units')
ax.set_title('PC1 Time Series', fontsize='x-large')
plt.show()
"""
