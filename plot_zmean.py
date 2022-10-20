"""
    This script plots (time and zonal) averages of various output variables averaged over X years'-worth of data from Isca
    Also plots differences between 2 datasets - important that they are of the same resolution (e.g. both T21 or both T42)
    TEST
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from shared_functions import *
from datetime import datetime

def plot_combo(u, T, lvls, perturb, lat, p, exp_name):
    # Plots time and zonal-mean Zonal Wind Speed and Temperature
    vertical = input("Plot vs. a) log pressure or b) pseudo-altitude?")
    print(datetime.now(), " - plotting")
    fig, ax = plt.subplots(figsize=(6,6))

    if vertical == "a":
        csa = ax.contourf(lat, p, T, levels=lvls[0], cmap='Blues_r')
        csb = ax.contour(lat, p, u, colors='k', levels=lvls[1], linewidths=1)
        plt.contour(lat, p, perturb, colors='g', linewidths=1, alpha=0.4, levels=11)
        plt.ylabel('Pressure (hPa)', fontsize='x-large')
        plt.ylim(max(p), upper_p) #goes to ~1hPa
        plt.yscale('log')

    if vertical == "b":
        # Use altitude rather than pressure for vertical
        z = altitude(p)
        upper_z = -7*np.log(upper_p/1000)
        u = use_altitude(u, z, lat, 'pfull', 'lat', r'ms$^{-1}$')
        T = use_altitude(T, z, lat, 'pfull', 'lat', 'K')
        H = use_altitude(heat, z, lat, 'pfull', 'lat', r'Ks$^{-1}$')
        csa = T.plot.contourf(levels=lvls[0], cmap='Blues_r', add_colorbar=False)
        csb = ax.contour(lat, z, u, colors='k', levels=lvls[1], linewidths=1)
        plt.contour(lat, z, H, colors='g', linewidths=1, alpha=0.4, levels=11)
        plt.ylabel('Pseudo-Altitude (km)', fontsize='x-large')
        plt.ylim(min(z), upper_z) #goes to ~1hPa

    ax.contourf(csa, colors='none')
    csb.collections[int(len(lvls[1])/2)].set_linewidth(1.5)
    cb = plt.colorbar(csa, extend='both')
    cb.set_label(label='Temperature (K)', size='x-large')
    cb.ax.tick_params(labelsize='x-large')
    plt.xlabel(r'Latitude ($\degree$N)', fontsize='x-large')
    plt.xlim(0,90)
    plt.xticks([10, 30, 50, 70, 90], ['10', '30', '50', '70', '90'])    
    plt.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    plt.savefig(exp_name+'_zonal.pdf', bbox_inches = 'tight')

    return plt.close()

def plot_diff(vars, units, lvls, perturb, lat, p, exp_name):
    # Plots differences in time and zonal-mean of variables (vars)
    vertical = input("Plot vs. a) log pressure or b) pseudo-altitude?")
    lvls_diff = np.arange(-20, 22.5, 2.5)
    for i in range(len(vars)):
        print(datetime.now(), " - taking differences")
        x = diff_variables(vars[i], lat, p, units[i])
        print(datetime.now(), " - plotting")
        fig, ax = plt.subplots(figsize=(6,6))
        
        if vertical == "a":
            cs1 = ax.contourf(lat, p, x, levels=lvls_diff, cmap='RdBu_r')
            cs2 = ax.contour(lat, p, vars[i][0], colors='k', levels=lvls[i], linewidths=1, alpha=0.4)
            plt.contour(lat, p, perturb, colors='g', linewidths=1, alpha=0.4, levels=11)
            plt.ylabel('Pressure (hPa)', fontsize='x-large')
            plt.ylim(max(p), upper_p) #goes to ~1hPa
            plt.yscale('log')

        if vertical == "b":
            # Use altitude rather than pressure for vertical
            z = altitude(p)
            upper_z = -7*np.log(upper_p/1000)
            var_z = use_altitude(vars[i][0], z, lat, 'pfull', 'lat', units[i])
            H = use_altitude(perturb, z, lat, 'pfull', 'lat', r'Ks$^{-1}$')
            cs1 = x.plot.contourf(levels=lvls_diff, cmap='RdBu_r', add_colorbar=False)
            cs2 = ax.contour(lat, z, var_z, colors='k', levels=lvls[i], linewidths=1)
            plt.contour(lat, z, H, colors='g', linewidths=1, alpha=0.4, levels=11)
            plt.ylabel('Pseudo-Altitude (km)', fontsize='x-large')
            plt.ylim(min(z), upper_z) #goes to ~1hPa

        ax.contourf(cs1, colors='none')        
        cs2.collections[int(len(lvls[i])/2)].set_linewidth(1.5)
        cb = plt.colorbar(cs1, extend='both')
        cb.set_label(label='Difference ('+units[i]+')', size='x-large')
        cb.ax.tick_params(labelsize='x-large')        
        plt.xlabel(r'Latitude ($\degree$N)', fontsize='x-large')
        plt.xlim(0,90)
        plt.xticks([10, 30, 50, 70, 90], ['10', '30', '50', '70', '90'])        
        plt.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
        plt.savefig(exp_name+'_diff{:.0f}.pdf'.format(i), bbox_inches = 'tight')
        plt.close()

def comparison(var, lats):
    print(datetime.now(), " - addition")
    if lats == 60:
        compare = [var[0].sel(lat=lats, method='nearest'), var[1].sel(lat=lats, method='nearest'), var[2].sel(lat=lats, method='nearest')]
    else:
        compare = [var[0].sel(lat=lats).mean('lat'), var[1].sel(lat=lats).mean('lat'), var[2].sel(lat=lats).mean('lat')]
    compare.append(compare[0]+compare[1])
    compare.append(compare[3]-compare[2])
    return compare

def linear_add(indir, exp, label, lats, lats_label):
    vars = ['u', 'T']
    xlabels = [lats_label+r' mean zonal wind (ms$^{-1}$)', lats_label+' mean temperature (K)']
    names = ['mid-lat heat only (a)', 'polar heat only (b)', 'combined simulation (c)', 'linear component (d=a+b)', '-1 x non-linear component -(c-d)']
    colors = ['#B30000', '#0099CC', 'k', '#4D0099', '#CC0080']
    lines = ['--', ':', '-', '-.', ':']

    print(datetime.now(), " - opening files")
    u = []
    T = []
    for i in range(len(exp)):
        u.append(xr.open_dataset(indir+exp[i]+'_utz.nc', decode_times=False).ucomp[0])
        T.append(xr.open_dataset(indir+exp[i]+'_Ttz.nc', decode_times=False).temp[0])
    p = u[0].pfull
    compare = [comparison(u, lats), comparison(T, lats)]
    
    print(datetime.now(), " - plotting")
    for j in range(len(vars)):      
        fig, ax = plt.subplots(figsize=(8,5.5))
        for i in range(len(compare[j])):
            ax.plot(compare[j][i].transpose(), p, color=colors[i], linestyle=lines[i], label=names[i], linewidth=1.75)
        #ax.axvline(0, color='k', linewidth=0.25)
        ax.set_xlabel(xlabels[j], fontsize='x-large')
        ax.set_ylabel('Pressure (hPa)', fontsize='x-large')
        ax.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
        plt.legend(fancybox=False, ncol=1, fontsize='large', labelcolor = colors) #, loc='lower right')
        plt.ylim(max(p), 1)
        plt.yscale('log')
        plt.savefig(vars[j]+'_addvcombo_'+label+'.pdf', bbox_inches = 'tight')
        plt.close()

if __name__ == '__main__': 
    #Set-up data to be read in
    indir = '/disco/share/rm811/processed/'
    plot_type = input("Plot a) individual experiments, b) difference vs. control, or c) linear additions?")
    
    if plot_type == 'a' or plot_type == 'b':
        var_type = input("Plot a) depth, b) width, c) location, d) strength or e) topography experiments?")
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
        exp, labels, xlabel = return_exp(extension)
        upper_p = 1 # hPa
        lvls = [np.arange(160, 330, 10), np.arange(-200, 205, 5)]

        for i in range(len(exp)):
            print(datetime.now(), " - opening files ({0:.0f}/{1:.0f})".format(i+1, len(exp)))
            uz = xr.open_dataset(indir+exp[i]+'_utz.nc', decode_times=False).ucomp[0]
            Tz = xr.open_dataset(indir+exp[i]+'_Ttz.nc', decode_times=False).temp[0]
            lat = uz.coords['lat'].data
            p = uz.coords['pfull'].data
            #MSF = calc_streamfn(xr.open_dataset(indir+exp[0]+'_vtz.nc', decode_times=False).vcomp[0], p, lat)  # Meridional Stream Function
            #MSF_xr = xr.DataArray(MSF, coords=[p,lat], dims=['pfull','lat'])  # Make into an xarray DataArray
            #MSF_xr.attrs['units']=r'kgs$^{-1}$'

            if i == 0:
                print("skipping control")
            elif i != 0:
                #Read in data to plot polar heat contours
                file = '/disco/share/rm811/isca_data/' + exp[i] + '/run0100/atmos_daily_interp.nc'
                ds = xr.open_dataset(file)
                perturb = ds.local_heating.sel(lon=180, method='nearest').mean(dim='time')  
                if plot_type =='a':
                    plot_combo(uz, Tz, lvls, perturb, lat, p, exp[i])
                elif plot_type == 'b':
                        u = [uz, xr.open_dataset(indir+exp[0]+'_utz.nc', decode_times=False).ucomp[0]]
                        T = [Tz, xr.open_dataset(indir+exp[0]+'_Ttz.nc', decode_times=False).temp[0]]
                        plot_diff([T, u], ['K', r'ms$^{-1}$'], [np.arange(160, 330, 10), np.arange(-200, 210, 10)],\
                            perturb, lat, p, exp[i])
   
    elif plot_type == 'c':
        basis = 'PK_e0v4z13'
        heat_type = input('Plot a) zonally symmetric pole-centred or b) off-pole heat?')
        if heat_type == 'a':
            polar_heat = '_w15a4p800f800g50'
            midlat_heat = '_q6m2y45l800u200'
            exp = [basis+midlat_heat, basis+polar_heat, basis+polar_heat+midlat_heat]
            label = 'polar'
        elif heat_type == 'b':
            polar_heat = '_a11x75y180w5v45p800'
            midlat_heat = '_q6m2y45'
            exp = [basis+midlat_heat+'l800u200', basis+polar_heat, basis+polar_heat+midlat_heat]
            label = 'offpole'
        lat_slice = input('Plot a) 60N, b) polar cap, or c) 45-75N average?')
        if lat_slice == 'a':
            lats = 60
            lats_label = r'$60\degree$N'
        elif lat_slice == 'b':
            lats = slice(60, 90) # following Dunn-Sigouin and Shaw (2015) for meridional heat flux
            lats_label = 'polar cap'
        elif lat_slice == 'c':
            lats = slice(45, 75)
            lats_label = r'$45-75\degree$N'
        linear_add(indir, exp, label, lats, lats_label)
