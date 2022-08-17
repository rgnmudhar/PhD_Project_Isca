import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import datetime

def anomaly(a, az):
    print(datetime.now(), " - finding anomaly")
    anom = a - az
    return anom

def plot_v(x, lvls, heat, name, lab):
    print(datetime.now(), " - plotting")
    # Plot of GPH anomaly
    lat = heat.coords['lat'].data
    p = heat.coords['pfull'].data
    fig, ax = plt.subplots(figsize=(6,6))
    cs = ax.contourf(lat, p, x, levels=lvls, cmap='Reds')
    ax.contourf(cs, colors='none')
    cb = plt.colorbar(cs)
    cb.set_label(label=lab, size='x-large')
    cb.ax.tick_params(labelsize='x-large')
    plt.contour(lat, p, heat, colors='g', linewidths=1, alpha=0.4, levels=11)
    plt.xlabel('Latitude', fontsize='x-large')
    plt.xlim(-90,90)
    plt.xticks([-90, -45, 0, 45, 90],\
        ['-90N', '-45N', '0', '45N', '90N'])
    plt.ylabel('Pressure (hPa)', fontsize='x-large')
    plt.ylim(max(p), 1) #goes to ~1hPa
    plt.yscale('log')
    plt.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    plt.savefig(name+'_zonal.pdf', bbox_inches = 'tight')
    return plt.close()

def plot_h(x, lvls, heat, name):
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
    ctrl = basis #+'_q6m2y45l800u200'
    exp = [basis+'_a4x75y0w5v30p800'+perturb,\
            basis+'_a4x75y90w5v30p800'+perturb,\
            basis+'_a4x75y180w5v30p800'+perturb,\
            basis+'_a4x75y270w5v30p800'+perturb]
    lons = [0, 90, 180, 270]
    p = [850, 200, 10]
    p_lvls_diff = [np.arange(-150, 170, 20), np.arange(-200, 220, 20), np.arange(-250, 270, 20)]

    plot_orientation = input("Plot a) horizontal plane or b) vertical cross-section?")
    plot_type = input("Plot a) single experiment or b) difference between 2 experiments?")
    
    if plot_orientation == 'a':
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
                    plot_h(anomaly(gph_t, gph_tz)[0], p[j], np.arange(-225, 250, 25), heat, exp[i]+'_'+str(p[j])+'gph.pdf')
            elif plot_type == 'b':
                for j in range(len(p)):
                    print(datetime.now(), " - p = {0:.0f} hPa".format(p[j]))
                    gph_t1 = xr.open_dataset(indir+exp[i]+'_ht.nc', decode_times=False).height.sel(pfull=p[j], method='nearest')
                    gph_tz1 = xr.open_dataset(indir+exp[i]+'_htz.nc', decode_times=False).height.sel(pfull=p[j], method='nearest')
                    gph_t2 = xr.open_dataset(indir+ctrl+'_ht.nc', decode_times=False).height.sel(pfull=p[j], method='nearest')
                    gph_tz2 = xr.open_dataset(indir+ctrl+'_htz.nc', decode_times=False).height.sel(pfull=p[j], method='nearest')
                    anom1 = anomaly(gph_t1, gph_tz1)[0]
                    anom2 = anomaly(gph_t2, gph_tz2)[0]
                    anom_diff = anom1 - anom2
                    plot_h(anom_diff, p_lvls_diff[j], heat, exp[i]+'_'+str(p[j])+'gph_diff.pdf')
    elif plot_orientation == 'b':
        for i in range(len(exp)):
            print(datetime.now(), " - opening files for ", exp[i])
            #Read in data to plot polar heat contours
            file = '/disco/share/rm811/isca_data/' + exp[i] + '/run0100/atmos_daily_interp.nc'
            ds = xr.open_dataset(file)
            heat = ds.local_heating.sel(lon=lons[i], method='nearest').mean(dim='time')
            if plot_type == 'a':
                gph_t = xr.open_dataset(indir+exp[i]+'_ht.nc', decode_times=False).height
                gph_tz = xr.open_dataset(indir+exp[i]+'_htz.nc', decode_times=False).height
                #a = anomaly(gph_t, gph_tz).mean(dim='time').mean(dim='lon')
                plot_v(gph_tz[0], 21, heat, exp[i]+'_gph_v.pdf', 'GPH (m)')
