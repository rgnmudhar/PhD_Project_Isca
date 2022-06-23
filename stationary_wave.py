import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def anomaly(a, az):
    print("finding anomaly")
    anom = a - az
    return anom

def plot(var, x, p, lvls):
    print("plotting")
    asym = xr.open_dataset('/home/links/rm811/Isca/input/asymmetry/'+var+'.nc', decode_times=False)
    heat = asym.sel(pfull=500, method='nearest').variables[var]
    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    cs = ax.contourf(asym.coords['lon'].data, asym.coords['lat'].data, x,\
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

indir = '/disco/share/rm811/processed/'
var = 'q6m2y45l800u200'
exp = 'PK_e0v4z13_' + var
p = 200
print("finding gph")
gph_t = xr.open_dataset(indir+exp+'_ht.nc', decode_times=False).height.sel(pfull=p, method='nearest')
gph_tz = xr.open_dataset(indir+exp+'_htz.nc', decode_times=False).height.sel(pfull=p, method='nearest')

plot(var, anomaly(gph_t, gph_tz)[0], p, np.arange(-200,220,20)) #np.arange(-250,260,10)) np.arange(-120,130,10)
