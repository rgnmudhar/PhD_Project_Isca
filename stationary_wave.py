import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from aostools import climate
from shared_functions import *
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

def plot_h(x, lvls, heat, ds, name):
    print(datetime.now(), " - plotting")
    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    cs = ax.contourf(ds.coords['lon'].data, ds.coords['lat'].data, x,\
        cmap='RdBu_r', levels=lvls, transform = ccrs.PlateCarree())
    cb = plt.colorbar(cs, pad=0.1)
    cb.set_label(label='Geopotential Height Anomaly (m)', size='x-large')
    cb.ax.tick_params(labelsize='x-large')
    ln = ax.contour(ds.coords['lon'].data, ds.coords['lat'].data, heat,\
        levels=11, colors='g', linewidths=0.5, alpha=0.4, transform = ccrs.PlateCarree())
    #ax.coastlines()
    ax.set_global()
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    ax.set_extent([-180, 180, 0, 90], crs=ccrs.PlateCarree())
    #plt.title(r'$\sim$' + '{0:.0f} hPa Geopotential Height Anomaly'.format(p), fontsize='xx-large')
    plt.savefig(name, bbox_inches = 'tight')
    return plt.close()

def find_heat(exp, p, type):
    if type == 'exp':
        file = '/disco/share/rm811/isca_data/' + exp + '/run0100/atmos_daily_interp.nc'
        ds = xr.open_dataset(file)
        heat = ds.local_heating.sel(pfull=p, method='nearest').mean(dim='time')
    elif type == 'polar':
        name = 'a4x75y180w5v30p800'
        file = '/secamfs/userspace/phd/rm811/Isca/input/polar_heating/'+name+'.nc'
        ds = xr.open_dataset(file)
        heat = ds.sel(pfull=p, method='nearest').variables[name]
    elif type == 'asym':
        name = 'a4x75y180w5v30p800'
        file = '/secamfs/userspace/phd/rm811/Isca/input/asymmetry/'+name+'.nc'
        ds = xr.open_dataset(file)
        heat = ds.sel(pfull=p, method='nearest').variables[name]
    return ds, heat

if __name__ == '__main__': 
    #Set-up data to be read in
    indir = '/disco/share/rm811/processed/'
    basis = 'PK_e0v4z13'
    perturb = '_q6m2y45'
    off_pole = '_a4x75y180w5v30p800'
    polar = '_w15a4p800f800g50'
    var_type = input("Plot a) depth, b) width, c) location, or d) strength experiments?")
    if var_type == 'a':
        extension = '_depth'
    elif var_type == 'b':
        extension = '_width'
    elif var_type == 'c':
        extension = '_loc'
    elif var_type == 'd':
        extension = '_strength'
    exp, labels, xlabel = return_exp(extension)

colors = ['#B30000', '#FF9900', '#FFCC00', '#00B300', '#0099CC', '#4D0099', '#CC0080', '#666666']

# For mean state, plot pressure vs. wave 1/2 magnitudes across experiments
fig, ax = plt.subplots(figsize=(6,6))
for i in range(len(exp)):
    gph = xr.open_dataset(indir+exp[i]+'_ht.nc', decode_times=False).height[0].sel(lat=60, method='nearest')
    waves = climate.GetWavesXr(gph)
    ax.plot(waves.sel(k=1).mean('lon').transpose(), gph.pfull, color=colors[i], linestyle='-', label=labels[i])
ax.set_xlabel('mean wave-1 magnitude ', fontsize='x-large')
ax.set_ylabel('Pressure (hPa)', fontsize='x-large')
ax.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
plt.legend()
plt.ylim(max(gph.pfull), 1)
plt.yscale('log')
plt.title(xlabel, fontsize='x-large')
plt.show()

"""
    lons = [0, 90, 180, 270]
    p_lvls_diff = [np.arange(-150, 170, 20), np.arange(-200, 220, 20), np.arange(-250, 270, 20)]
    p_lvls_diff2 = [np.arange(-20, 22, 2), np.arange(-50, 55, 5), np.arange(-120, 130, 10)]

    plot_orientation = input("Plot a) horizontal plane or b) vertical cross-section?")
    plot_type = input("Plot a) single experiment, b) difference between 2 experiments?")
    
    if plot_orientation == 'a':
        for i in range(len(exp)):
            print(datetime.now(), " - opening files for ", exp[i])
            #Read in data to plot polar heat contours
            ds, heat = find_heat(exp[i], 1000, type='exp')
            if plot_type == 'a':
                for j in range(len(p)):
                    print(datetime.now(), " - p = {0:.0f} hPa".format(p[j]))
                    gph_t = xr.open_dataset(indir+exp[i]+'_ht.nc', decode_times=False).height.sel(pfull=p[j], method='nearest')
                    gph_tz = xr.open_dataset(indir+exp[i]+'_htz.nc', decode_times=False).height.sel(pfull=p[j], method='nearest')
                    plot_h(anomaly(gph_t, gph_tz)[0], np.arange(-250, 275, 25), heat, ds, exp[i]+'_'+str(p[j])+'gph.pdf') 
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
                    plot_h(anom_diff, p_lvls_diff[j], heat, ds, exp[i]+'_'+str(p[j])+'gph_diff.pdf')

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

exp = [basis+off_pole, basis+'_q6m2y45l800u200', basis+off_pole+perturb]
for j in range(len(p)):
    print(datetime.now(), " - p = {0:.0f} hPa".format(p[j]))
    anomalies = []
    for i in range(len(exp)):
        gph_t = xr.open_dataset(indir+exp[i]+'_ht.nc', decode_times=False).height.sel(pfull=p[j], method='nearest')
        gph_tz = xr.open_dataset(indir+exp[i]+'_htz.nc', decode_times=False).height.sel(pfull=p[j], method='nearest')
        anomalies.append(anomaly(gph_t, gph_tz)[0])
    print(datetime.now(), " - addition")
    anom_add = anomalies[0] + anomalies[1]
    print(datetime.now(), " - combo")
    anom_combo = anomalies[2]
    print(datetime.now(), " - difference")
    anom_diff = anom_combo - anom_add
    ds, heat1 = find_heat(exp[0], 1000, type='polar')
    ds, heat2 = find_heat(exp[1], 500, type='exp')
    #anom_diff = anomalies[0] - anomalies[1]
    plotting = [anomalies[0], anomalies[1], anom_add, anomalies[2], anom_diff]
    names = [exp[0], exp[1], exp[0]+'+ctrl', exp[2], 'combo-'+exp[0]+'+ctrl']
    lvls_full = [np.arange(-120, 130, 10), np.arange(-200, 220, 20), np.arange(-250, 270, 20)]
    lvls_diff = [np.arange(-25, 27, 2), np.arange(-60, 70, 10), np.arange(-150, 170, 20)]
    lvls_polar = [np.arange(-10, 11, 1), np.arange(-16, 18, 2), np.arange(-30, 35, 5)] #np.arange(-7, 8, 1)
    lvls = [lvls_polar[j], lvls_full[j], lvls_full[j], lvls_full[j], lvls_diff[j]]

    for k in range(len(plotting)):
        print(datetime.now(), " - plotting ", names[k])
        ax = plt.axes(projection=ccrs.NorthPolarStereo())
        cs = ax.contourf(ds.coords['lon'].data, ds.coords['lat'].data, plotting[k],\
        cmap='RdBu_r', levels=lvls[k], transform = ccrs.PlateCarree())
        cb = plt.colorbar(cs, pad=0.1)
        cb.set_label(label='Geopotential Height Anomaly (m)', size='x-large')
        cb.ax.tick_params(labelsize='x-large')
        ln_ctrl = ax.contour(ds.coords['lon'].data, ds.coords['lat'].data, heat1,\
            levels=11, colors='g', linewidths=0.5, alpha=0.4, transform = ccrs.PlateCarree())
        if k in range(1,5):
            ln_exp = ax.contour(ds.coords['lon'].data, ds.coords['lat'].data, heat2,\
                levels=11, colors='g', linewidths=0.5, alpha=0.4, transform = ccrs.PlateCarree())
        ax.set_global()
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        ax.set_extent([-180, 180, 0, 90], crs=ccrs.PlateCarree())
        #ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        plt.savefig(names[k]+'_'+str(p[j])+'gph.pdf', bbox_inches = 'tight')
        plt.close()
"""