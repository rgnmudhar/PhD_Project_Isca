"""
    This script plots v'T' predominantly
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from shared_functions import *

def find_anom(exp):
    print(datetime.now(), " - opening files")
    v = xr.open_dataset(indir+exp+'_v.nc', decode_times=False).vcomp
    T = xr.open_dataset(indir+exp+'_T.nc', decode_times=False).temp
    vz = xr.open_dataset(indir+exp+'_vz.nc', decode_times=False).vcomp
    Tz = T.mean('lon')
    print(datetime.now(), " - finding anomalies")
    vp = v - vz
    Tp = T - Tz
    print(datetime.now(), " - finding v'T'")
    vpTp_bar = (vp*Tp).mean('lon')
    return vpTp_bar

def plot(vpTp, exp, heat, lvls, colors):
    print(datetime.now(), " - plotting ", exp)
    fig, ax = plt.subplots(figsize=(6,6))
    cs = ax.contourf(lat, p, vpTp.mean('time'), levels=lvls, cmap=colors)
    ax.contourf(cs, colors='none')
    cb = plt.colorbar(cs)
    cb.set_label(label=r"v'T' (m s$^{-1}$ K)", size='x-large')
    cb.ax.tick_params(labelsize='x-large')
    plt.contour(lat, p, heat, colors='g', linewidths=1, alpha=0.4, levels=11)
    plt.xlabel(r'Latitude ($\degree$N)', fontsize='x-large')
    plt.xlim(0,90)
    plt.xticks([10, 30, 50, 70, 90], ['10', '30', '50', '70', '90'])
    plt.ylabel('Pressure (hPa)', fontsize='x-large')
    plt.ylim(max(p), 1) #goes to ~1hPa
    plt.yscale('log')
    plt.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    plt.savefig(exp+'_vpTp.pdf', bbox_inches = 'tight')
    return plt.close()

if __name__ == '__main__': 
    #Set-up data to be read in
    indir = '/disco/share/rm811/processed/'
    basis = 'PK_e0v4z13'
    perturb = '_q6m2y45l800u200'
    filename = '_w15a4p800f800g50'+perturb
    extension = '_depth'
    exp = [basis+perturb,\
        basis+'_w15a4p900f800g50'+perturb,\
        basis+'_w15a4p800f800g50'+perturb,\
        basis+'_w15a4p700f800g50'+perturb,\
        basis+'_w15a4p600f800g50'+perturb,\
        basis+'_w15a4p500f800g50'+perturb,\
        basis+'_w15a4p400f800g50'+perturb,\
        basis+'_w15a4p300f800g50'+perturb]
    labels = ['no heat', '900', '800', '700', '600', '500', '400', '300']

    colors = ['#B30000', '#FF9900', '#FFCC00', '#00B300', '#0099CC', '#4D0099', '#CC0080', '#666666']
    legend = ['no heat', '800 hPa']
    levels = [500, 100, 50, 10]

    vpTp = []
    for i in range(len(exp)):
        vpTp.append(find_anom(exp[i]))
        file = '/disco/share/rm811/isca_data/' + exp[i]+ '/run0100/atmos_daily_interp.nc'
        ds = xr.open_dataset(file)
        heat = ds.local_heating.sel(lon=180, method='nearest').mean(dim='time')
        lat = ds.coords['lat'].data
        p = ds.coords['pfull'].data
        plot(vpTp[i], exp[i], heat, np.arange(-20, 190, 10), 'Blues')

        if i != 0:
            vpTp_diff = vpTp[i] - vpTp[0]
            plot(vpTp_diff, exp[i]+'_diff', heat, np.arange(-40, 42, 2), 'RdBu_r')   

    for j in range(len(levels)):
        print(datetime.now(), " - plotting zonal wind PDFs at {:.0f} hPa".format(levels[j]))
        x_min = x_max = 0 
        fig, ax = plt.subplots(figsize=(8,6))
        for i in range(len(exp)):
            x, f, mode = pdf(vpTp[i].sel(pfull=levels[j], method='nearest').sel(lat=60, method='nearest'))
            if max(x) > x_max:
                x_max = max(x)
            if min(x) < x_min:
                x_min = min(x)
            ax.plot(x, f, linewidth=1.25, color=colors[i], label=legend[i])
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel(r"v'T' (m s$^{-1}$ K)", fontsize='x-large')
        ax.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
        plt.legend(loc='upper right',fancybox=False, shadow=True, ncol=1, fontsize='large')
        plt.savefig(exp[0]+extension+'_vpTp'+'_{:.0f}pdf.pdf'.format(str(levels[j])), bbox_inches = 'tight')
        plt.close()