"""
    This script plots v'T' predominantly
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from shared_functions import *

def find_anom_all(exp):
    print(datetime.now(), " - opening files")
    v = xr.open_dataset(indir+exp+'_v.nc', decode_times=False).vcomp
    T = xr.open_dataset(indir+exp+'_T.nc', decode_times=False).temp
    vz = xr.open_dataset(indir+exp+'_vz.nc', decode_times=False).vcomp
    Tz = xr.open_dataset(indir+exp+'_Tz.nc', decode_times=False).temp
    print(datetime.now(), " - finding anomalies")
    vp = v - vz
    Tp = T - Tz
    print(datetime.now(), " - finding v'T'")
    vpTp_bar = (vp*Tp).mean('lon')
    return vpTp_bar

def find_anom_p(exp, p):
    # Meridional heat flux weighted by cos(lat) and meridionally averaged from 60 to 90 N at p hPa
    # Based on Dunn-Sigouin & Shaw (2015)
    print(datetime.now(), " - opening files for ", exp)
    v = xr.open_dataset(indir+exp+'_v.nc', decode_times=False).vcomp.sel(pfull=p, method='nearest')
    T = xr.open_dataset(indir+exp+'_T.nc', decode_times=False).temp.sel(pfull=p, method='nearest')
    vz = xr.open_dataset(indir+exp+'_vz.nc', decode_times=False).vcomp.sel(pfull=p, method='nearest')
    Tz = xr.open_dataset(indir+exp+'_Tz.nc', decode_times=False).temp.sel(pfull=p, method='nearest')
    print(datetime.now(), " - finding anomalies")
    vp = v - vz
    Tp = T - Tz
    print(datetime.now(), " - finding v'T'")
    vpTp_w = (vp*Tp) / np.cos(np.deg2rad(vz.lat))
    vpTp_sub = vpTp_w.sel(lat=slice(60,90)).mean('lat')
    vpTp_bar = vpTp_sub.mean('lon')
    return vpTp_bar

def plot(vpTp, exp, heat, lvls, colors):
    print(datetime.now(), " - plotting ", exp)
    fig, ax = plt.subplots(figsize=(6,6))
    cs = ax.contourf(lat, p, vpTp.mean('time'), levels=lvls, cmap=colors)
    ax.contourf(cs, colors='none')
    cb = plt.colorbar(cs)
    cb.set_label(label=r"v'T' (K m s$^{-1}$)", size='x-large')
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
    extension = '_width'
    if extension == '_depth':
        exp = [basis+perturb,\
        basis+'_w15a4p900f800g50'+perturb,\
        basis+'_w15a4p800f800g50'+perturb,\
        basis+'_w15a4p700f800g50'+perturb,\
        basis+'_w15a4p600f800g50'+perturb,\
        basis+'_w15a4p500f800g50'+perturb,\
        basis+'_w15a4p400f800g50'+perturb,\
        basis+'_w15a4p300f800g50'+perturb]
        labels = ['no heat', '900', '800', '700', '600', '500', '400', '300']
        xlabel = 'Depth of Heating (hPa)'
    elif extension == '_width':
        exp = [basis+perturb,\
        basis+'_w10a4p800f800g50'+perturb,\
        basis+'_w15a4p800f800g50'+perturb,\
        basis+'_w20a4p800f800g50'+perturb,\
        basis+'_w25a4p800f800g50'+perturb,\
        basis+'_w30a4p800f800g50'+perturb,\
        basis+'_w35a4p800f800g50'+perturb,\
        basis+'_w40a4p800f800g50'+perturb]
        labels = ['no heat', '10', '15', '20', '25', '30', '35', '40']
        xlabel = r'Extent of Heating ($\degree$)'
    elif extension == '_strengthp800':
        exp = [basis+perturb,\
        basis+'_w15a2p800f800g50'+perturb,\
        basis+'_w15a4p800f800g50'+perturb,\
        basis+'_w15a6p800f800g50'+perturb,\
        basis+'_w15a8p800f800g50'+perturb]
        labels = ['no heat', '2', '4', '6', '8'] #['no heat', '800', '600', '400']
        xlabel = r'Strength of Heating (K day$^{-1}$)'
    elif extension == '_loc':   
        perturb = '_q6m2y45'
        exp = [basis+'_q6m2y45l800u200',\
            basis+'_a4x75y0w5v30p800'+perturb,\
            basis+'_a4x75y90w5v30p800'+perturb,\
            basis+'_a4x75y180w5v30p800'+perturb,\
            basis+'_a4x75y270w5v30p800'+perturb]
        labels = [r'no heat', '0', '90', '180', '270']
        xlabel = r'Longitude of Heating ($\degree$E)'
    
    colors = ['#B30000', '#FF9900', '#FFCC00', '#00B300', '#0099CC', '#4D0099', '#CC0080', '#666666']

    plot_option = input("Plot a) lat-p of v'T' or b) PDFs of polar cap average v'T'?")

    if plot_option == 'a':
        vpTp = []
        for i in range(len(exp)):
            vpTp.append(find_anom_all(exp[i]))
            file = '/disco/share/rm811/isca_data/' + exp[i]+ '/run0100/atmos_daily_interp.nc'
            ds = xr.open_dataset(file)
            heat = ds.local_heating.sel(lon=180, method='nearest').mean(dim='time')
            lat = ds.coords['lat'].data
            p = ds.coords['pfull'].data
            plot(vpTp[i], exp[i], heat, np.arange(-20, 190, 10), 'Blues')

            if i != 0:
                vpTp_diff = vpTp[i] - vpTp[0]
                plot(vpTp_diff, exp[i]+'_diff', heat, np.arange(-60, 65, 5), 'RdBu_r')   
    
    elif plot_option == 'b':
        p = [500, 100, 50, 10]
        mode = []
        sd = []
        for j in range(len(p)):
            print(datetime.now(), ' - {:.0f} hPa'.format(p[j]))
            x_min = x_max = 0
            sub_mode = []
            sub_sd = []
            fig, ax = plt.subplots(figsize=(6,6))
            for i in range(len(exp)):
                x, f, m = pdf(find_anom_p(exp[i], p[j]))
                sub_mode.append(m)
                sub_sd.append(np.std(x))
                if max(x) > x_max:
                    x_max = max(x)
                if min(x) < x_min:
                    x_min = min(x)
                print(datetime.now(), ' - plotting')
                ax.plot(x, f, linewidth=1.25, color=colors[i], label=labels[i])
            mode.append(sub_mode)
            sd.append(sub_sd)
            ax.set_xlim(x_min, x_max)
            ax.set_xlabel(r"Polar Cap Average v'T' (K m s$^{-1}$)", fontsize='x-large')
            ax.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
            plt.legend(labels=labels, loc='upper right',fancybox=False, shadow=True, ncol=1, fontsize='large')
            plt.savefig(exp[0]+extension+'_vpTp'+'_{:.0f}pdf.pdf'.format(p[j]), bbox_inches = 'tight')
            plt.close()
        
        print(datetime.now(), " - plotting v'T' mode")
        colors = ['#B30000', '#00B300', '#0099CC', 'k']
        fig, ax = plt.subplots(figsize=(8,6))
        for j in range(len(p)):
            ax.plot(labels, mode[j], marker='o', linewidth=1.25, color=colors[j], linestyle=':', label='{:.0f} hPa'.format(p[j]))
        ax.set_xticks(labels)
        ax.set_xlabel(xlabel, fontsize='x-large')
        ax.set_ylabel(r"Polar Cap Average v'T' mode (K m s$^{-1}$)", fontsize='x-large')
        ax.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
        plt.legend(loc='right',fancybox=False, shadow=True, ncol=1, fontsize='large')
        plt.savefig(exp[0]+extension+'_vpTp'+'_mode.pdf', bbox_inches = 'tight')
        plt.close()

        print(datetime.now(), " - plotting v'T' s.d.")
        fig, ax = plt.subplots(figsize=(8,6))
        for j in range(len(p)):
            ax.plot(labels, sd[j], marker='o', linewidth=1.25, color=colors[j], linestyle=':', label='{:.0f} hPa'.format(p[j]))
        ax.set_xticks(labels)
        ax.set_xlabel(xlabel, fontsize='x-large')
        ax.set_ylabel(r"Polar Cap Average v'T' S.D. (K m s$^{-1}$)", fontsize='x-large')
        ax.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
        plt.legend(loc='right',fancybox=False, shadow=True, ncol=1, fontsize='large')
        plt.savefig(exp[0]+extension+'_vpTp'+'_sd.pdf', bbox_inches = 'tight')
        plt.close()