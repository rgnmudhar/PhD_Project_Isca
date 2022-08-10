import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import datetime

def anomaly(a, az):
    print(datetime.now(), " - finding anomaly")
    anom = a - az
    return anom

def plot(x, p, lvls, heat, name):
    print(datetime.now(), " - plotting")
    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    cs = ax.contourf(heat.coords['lon'].data, heat.coords['lat'].data, x,\
        cmap='RdBu_r', levels=lvls, transform = ccrs.PlateCarree())
    cb = plt.colorbar(cs, pad=0.1)
    cb.set_label(label='Geopotential Height Anomaly (m)', size='x-large')
    cb.ax.tick_params(labelsize='x-large')
    ln = ax.contour(heat.coords['lon'].data, heat.coords['lat'].data, heat,\
        levels=11, colors='g', linewidths=0.5, alpha=0.4, transform = ccrs.PlateCarree())
    #ax.coastlines()
    ax.set_global()
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    ax.set_extent([-180, 180, 0, 90], crs=ccrs.PlateCarree())
    #plt.title(r'$\sim$' + '{0:.0f} hPa Geopotential Height Anomaly'.format(p), fontsize='xx-large')
    plt.savefig(name, bbox_inches = 'tight')
    return plt.close()

if __name__ == '__main__': 
    #Set-up data to be read in
    indir = '/disco/share/rm811/processed/'
    basis = 'PK_e0v4z13'
    perturb = '_q6m2y45'
    ctrl = basis+'_q6m2y45l800u200'
    exp = [basis+'_a4x75y0w5v30p800'+perturb,\
            basis+'_a4x75y90w5v30p800'+perturb,\
            basis+'_a4x75y180w5v30p800'+perturb,\
            basis+'_a4x75y270w5v30p800'+perturb]
    p = [850, 200, 10]

    plot_type = input("Plot a) single experiment or b) difference between 2 experiments?")

    for i in range(len(exp)):
        print(datetime.now(), " - opening files for ", exp[i])
        #Read in data to plot polar heat contours
        file = '/disco/share/rm811/isca_data/' + exp[i] + '/run0100/atmos_daily_interp.nc'
        ds = xr.open_dataset(file)
        heat = ds.local_heating.sel(pfull=1000, method='nearest').mean(dim='time')
        if plot_type == 'a':
            for j in range(len(p)):
                print(datetime.now(), " - p = {0:.0f} hPa".format(p[j]))
                gph_t = xr.open_dataset(indir+exp[i]+'_ht.nc', decode_times=False).height.sel(pfull=p[j], method='nearest')
                gph_tz = xr.open_dataset(indir+exp[i]+'_htz.nc', decode_times=False).height.sel(pfull=p[j], method='nearest')
                plot(anomaly(gph_t, gph_tz)[0], p[j], np.arange(-225, 250, 25), heat, exp[i]+'_'+str(p[j])+'gph.pdf')
        elif plot_type == 'b':
            for j in range(len(p)):
                print(datetime.now(), " - p = {0:.0f} hPa".format(p[j]))
                gph_t2 = xr.open_dataset(indir+ctrl+'_ht.nc', decode_times=False).height.sel(pfull=p[j], method='nearest')
                gph_t1 = xr.open_dataset(indir+exp[i]+'_ht.nc', decode_times=False).height.sel(pfull=p[j], method='nearest')
                gph_t_diff = (gph_t1 - gph_t2)[0]
                plot(gph_t_diff, p[j], 21, heat, exp[i]+'_'+str(p[j])+'gph_diff.pdf')

