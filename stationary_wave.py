import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from shared_functions import *

def anomaly(gph):
    print("finding anomaly")
    anom = gph - gph.mean(dim='lon')
    return anom

def plot(var, ds, x, p, lvls):
    print("plotting")
    asym = xr.open_dataset('/home/links/rm811/Isca/input/asymmetry/'+var+'.nc', decode_times=False)
    heat = asym.sel(pfull=500, method='nearest').variables[var]
    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    cs = ax.contourf(ds.coords['lon'].data, ds.coords['lat'].data, x,\
        cmap='RdBu_r', levels=lvls, transform = ccrs.PlateCarree())
    cb = plt.colorbar(cs, pad=0.1)
    cb.set_label(label='Geopotential Height Anomaly (m)', size='x-large')
    cb.ax.tick_params(labelsize='x-large')
    ln = ax.contour(asym.coords['lon'].data, asym.coords['lat'].data, heat,\
        levels=11, colors='g', linewidths=0.5, alpha=0.4, transform = ccrs.PlateCarree())
    #ax.coastlines()
    ax.set_global()
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    ax.set_extent([-180, 180, 0, 90], crs=ccrs.PlateCarree())
    #plt.title(r'$\sim$' + '{0:.0f} hPa Geopotential Height Anomaly'.format(p), fontsize='xx-large')
    plt.savefig(var+'_'+str(p)+'gph.pdf', bbox_inches = 'tight')
    return plt.close()

var = 'q6m2y45l800u200'
exp = 'PK_e0v4z13_' + var
time = 'daily'
years = 37 # user sets no. of years worth of data to ignore due to spin-up
file_suffix = '_interp'
files = discard_spinup2(exp, time, file_suffix, years)
print("finding gph")
ds = xr.open_mfdataset(files, decode_times=False)
gph_t = ds.height.mean(dim='time')
p = 200

plot(var, ds, anomaly(gph_t.sel(pfull=p, method='nearest')), p, np.arange(-200,220,20)) #np.arange(-250,260,10)) np.arange(-120,130,10)
