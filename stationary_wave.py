import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cm
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

def plot_waves1(exp, wav, k, name, type):
    uz = xr.open_dataset(indir+exp+'_utz.nc', decode_times=False).ucomp[0]
    p = uz.coords['pfull']
    lat = uz.coords['lat']
    fig, ax = plt.subplots(figsize=(6,6))
    print(datetime.now(), " - plot uz")
    uplt = uz.plot.contour(colors='k', linewidths=0.5, alpha=0.4, levels=ulvls)
    uplt.collections[int(len(ulvls)/2)].set_linewidth(1.5)
    print(datetime.now(), " - plot waves")
    if type == 'single':
        lvls = np.arange(0, 180, 20)
        cs = wav.plot.contourf(levels=lvls, cmap='Reds', add_colorbar=False)
    elif type == 'diff':
        lvls = np.arange(-20, 55, 5)
        norm = cm.TwoSlopeNorm(vmin=-20, vmax=50, vcenter=0)
        cs = wav.plot.contourf(levels=lvls, cmap='RdBu_r', norm=norm, extend='both', add_colorbar=False)
    cb = plt.colorbar(cs, extend='both')
    cb.set_label(label='absolute wave-{0:.0f} magnitude (m)'.format(2), size='large')
    cb.ax.set_yticks(lvls)
    fig.canvas.draw()
    ticklabs = cb.ax.get_yticklabels()
    cb.ax.set_yticklabels(ticklabs, fontsize='large')
    plt.ylim(max(p), 1) #to 1 hPa
    plt.yscale('log')
    plt.ylabel('Pressure (hPa)', fontsize='xx-large')
    plt.xlim(0,max(lat))
    plt.xticks([20, 40, 60, 80], ['20', '40', '60', '80'])
    plt.xlabel(r'Latitude ($\degree$N)', fontsize='xx-large')
    plt.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
    plt.title('')
    plt.savefig(name+'_wavemag1_k{0:.0f}_{1}.pdf'.format(k,type), bbox_inches = 'tight')
    return plt.close()

def find_tilt(indir, exp, chosen_lat, given_lon):
    print(datetime.now(), " - opening heat files")
    h_name = exp[11:]
    h = xr.open_dataset('../Inputs/' + h_name + '.nc')
    heat = h.sel(lat=chosen_lat, method='nearest').variables[h_name]
    h_lvls = 11
    
    print(datetime.now(), " - finding anomaly")
    ds_t = xr.open_dataset(indir+exp+'_ht.nc', decode_times=False).height[0]
    ds_tz = xr.open_dataset(indir+exp+'_htz.nc', decode_times=False).height[0]
    ds_anom = ds_t - ds_tz
    ds_anom_45 = ds_anom.sel(lat=chosen_lat, method='nearest')

    print(datetime.now(), " - plotting")
    lvls=np.arange(-300, 700, 100)
    norm = cm.TwoSlopeNorm(vmin=min(lvls), vmax=max(lvls), vcenter=0)
    fig, ax = plt.subplots(figsize=(10,6))
    cs = plt.contourf(ds_t.lon, ds_t.pfull, ds_anom_45, cmap='RdBu_r', levels=lvls, norm=norm, extend='both')
    cb = plt.colorbar(cs, extend='both')
    cb.set_label(label='GPH Anomaly (m)', size='xx-large')
    cb.ax.tick_params(labelsize='x-large')
    ax.contour(h.lon, h.pfull, heat, alpha=0.5, colors='g', levels=h_lvls)
    ax.set_ylim(max(ds_t.pfull), 1) #goes to ~1hPa
    ax.set_yscale('log')
    ax.set_ylabel('Pressure (hPa)', fontsize='xx-large')
    ax.set_xlabel(r'Longitude ($\degree$E)', fontsize='xx-large')
    ax.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
    ax.text(5, 1.75, r'$\lambda=$'+str(given_lon)+r'$\degree$E', color='k', fontsize='xx-large')
    plt.savefig(exp+'_tilt.pdf', bbox_inches = 'tight')
    return plt.close()

if __name__ == '__main__': 
    #Set-up data to be read in
    indir = '/disco/share/rm811/processed/'
    basis = 'PK_e0v4z13'
    perturb = '_q6m2y45'
    off_pole = '_a4x75y90w5v30p600'
    polar = '_w15a4p800f800g50'

    colors = ['k', '#B30000', '#FF9900', '#FFCC00', '#00B300', '#0099CC', '#4D0099', '#CC0080']
    blues = ['k', '#dbe9f6', '#bbd6eb', '#88bedc', '#549ecd',  '#2a7aba', '#0c56a0', '#08306b']
    ulvls = np.arange(-200, 210, 10)

    plot_type = input('Plot a) gph lat-p, b) gph @ 60N and PDFs @ 10 & 100 hPa, c) linear add, or d) tilt? ')

    if plot_type == 'c':
        exp = [basis+polar, basis+perturb+'l800u200', basis+polar+perturb+'l800u200']
        p = [500, 100, 10]
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
            #ds, heat1 = find_heat(exp[0], 1000, type='exp')
            ds, heat2 = find_heat(exp[1], 500, type='exp')
            #anom_diff = anomalies[0] - anomalies[1]
            plotting = [anomalies[0], anomalies[1], anom_add, anomalies[2], anom_diff]
            names = [exp[0], exp[1], exp[0]+'+ctrl', exp[2], 'combo-'+exp[0]+'+ctrl']
            lvls_full = [np.arange(-100, 110, 10), np.arange(-180, 200, 20), np.arange(-300, 320, 20)]
            lvls_diff = [np.arange(-100, 110, 10), np.arange(-180, 200, 20), np.arange(-300, 320, 20)]
            lvls_polar = [np.arange(-30, 75, 5), np.arange(-40, 50, 10), np.arange(-50, 50, 10)] #np.arange(-7, 8, 1)
            lvls = [lvls_polar[j], lvls_full[j], lvls_full[j], lvls_full[j], lvls_diff[j]]

            for k in range(len(plotting)):
                norm = cm.TwoSlopeNorm(vmin=min(lvls[k]), vmax=max(lvls[k]), vcenter=0)
                print(datetime.now(), " - plotting ", names[k])
                ax = plt.axes(projection=ccrs.NorthPolarStereo())
                cs = ax.contourf(ds.coords['lon'].data, ds.coords['lat'].data, plotting[k],\
                cmap='RdBu_r', levels=lvls[k], norm=norm, extend='both', transform = ccrs.PlateCarree())
                cb = plt.colorbar(cs, pad=0.1, extend='both')
                cb.set_label(label='Geopotential Height Anomaly (m)', size='x-large')
                cb.ax.tick_params(labelsize='x-large')
                #ln_ctrl = ax.contour(ds.coords['lon'].data, ds.coords['lat'].data, heat1,\
                #    levels=11, colors='g', linewidths=0.5, alpha=0.4, transform = ccrs.PlateCarree())
                #if k in range(1,5):
                #    ln_exp = ax.contour(ds.coords['lon'].data, ds.coords['lat'].data, heat2,\
                #        levels=11, colors='g', linewidths=0.5, alpha=0.4, transform = ccrs.PlateCarree())
                ax.set_global()
                ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
                ax.set_extent([-180, 180, 0, 90], crs=ccrs.PlateCarree())
                #ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
                plt.savefig(names[k]+'_'+str(p[j])+'gph.pdf', bbox_inches = 'tight')
                plt.close()

    elif plot_type == 'd':
        exp = 'PK_e0v4z13_a4x45y90w5v15p800_q6m2y45_s'
        lat = int(exp.partition("x")[2][:2])
        lon = int(exp.partition("y")[2][:2])
        find_tilt(indir, exp, lat, lon)
    else:
        var_type = input("Plot a) depth, b) width, c) location, d) strength or e) topography experiments, or f) test?")
        if var_type == 'a':
            extension = '_depth'
        elif var_type == 'b':
            extension = '_width'
        elif var_type == 'c':
            extension = '_loc'
        elif var_type == 'd':
            extension = '_strength'
        elif var_type == 'e':
            extension = '_topo'
        elif var_type == 'f':
            extension = '_test'
        exp, labels, xlabel = return_exp(extension)
        k = int(input('Which wave no.? (i.e. 0 for all, 1, 2, etc.) '))

        if plot_type == 'a':
            for i in range(len(exp)):
                print(datetime.now(), ' - opening files {0:.0f}/{1:.0f} - '.format(i+1, len(exp)), exp[i])
                gph = xr.open_dataset(indir+exp[i]+'_h.nc', decode_times=False).height.mean('time')
                print(datetime.now(), ' - wave decomposition')
                waves = climate.GetWavesXr(gph)
                wav = np.abs(waves.sel(k=k)).mean('lon')
                if i == 0:
                    print("skipping control")
                    wav_og = wav
                elif i != 0:
                    # Read in data to plot polar heat contours
                    plot_waves1(exp[i], wav, k, exp[i], 'single')

                    print(datetime.now(), " - taking differences")
                    wav_diff = wav - wav_og
                    plot_waves1(exp[i], wav_diff, k, exp[i], 'diff')
                
        elif plot_type == 'b':
            # For mean state, plot pressure vs. wave 1/2 magnitudes across experiments
            loc = input("Plot a) at 60N or b) for 45-75N mean?")
            if loc == "a":
                lats = slice(45,75)
                lab = r'$45-75\deg$N'
            elif loc == "b":
                lats = 60
                lab = r'$60\deg$N'
            mags10 = []
            mags100 = []
            fig, ax = plt.subplots(figsize=(6,6))
            for i in range(len(exp)):
                print(datetime.now(), ' - opening files {0:.0f}/{1:.0f} - '.format(i+1, len(exp)), exp[i])
                if type(lats) == int:
                    gph = xr.open_dataset(indir+exp[i]+'_h.nc', decode_times=False).height.sel(lat=lats, method='nearest')
                elif type(lats) == slice:
                    gph = xr.open_dataset(indir+exp[i]+'_h.nc', decode_times=False).height.sel(lat=lats).mean('lat')        
                print(datetime.now(), ' - wave decomposition')
                waves = climate.GetWavesXr(gph)
                wav = np.abs(waves.sel(k=k)).mean('lon')
                print(datetime.now(), ' - plotting')
                ax.plot(wav.mean('time').transpose(), gph.pfull, color=blues[i], linestyle='-', label=labels[i])
                mags10.append(wav.sel(pfull=10, method='nearest'))
                mags100.append(wav.sel(pfull=100, method='nearest'))
            ax.set_xlabel('absolute wave-{0:.0f} magnitude (m)'.format(k), fontsize='xx-large')
            ax.set_ylabel('Pressure (hPa)', fontsize='xx-large')
            ax.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
            plt.legend(fancybox=False, ncol=1, fontsize='x-large', loc='lower right')
            plt.ylim(max(gph.pfull), 1)
            plt.yscale('log')
            #plt.title(xlabel, fontsize='x-large')
            plt.savefig(basis+extension+'_wavemag2_k{0:.0f}.pdf'.format(k), bbox_inches = 'tight')

            print(datetime.now(), ' - plotting PDFs')
            plot_pdf('gph', indir, exp, '', mags10, 10, 0, labels, 'zonal-mean 10 hPa '+lab+' wave-{0:.0f} absolute magnitude'.format(k), blues, exp[0]+extension+'_wavemag_k{0:.0f}.pdf'.format(k))
            plot_pdf('gph', indir, exp, '', mags100, 100, 0, labels, 'zonal-mean 100 hPa '+lab+' wave-{0:.0f} absolute magnitude'.format(k), blues, exp[0]+extension+'_wavemag_k{0:.0f}.pdf'.format(k))
        
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
"""