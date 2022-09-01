"""
    Computes and plots EP flux vectors and divergence terms, based on Martin Jucker's code at https://github.com/mjucker/aostools/blob/d857987222f45a131963a9d101da0e96474dca63/climate.py
    Computes and plots meridional heat flux 
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from aostools import climate
from shared_functions import *
from datetime import datetime

def calc_ep(u, v, w, t):
    print(datetime.now(), ' - finding EP Fluxes')
    ep1, ep2, div1, div2 = climate.ComputeEPfluxDivXr(u, v, t, 'lon', 'lat', 'pfull', 'time', w=w, do_ubar=True) # default w=None and do_ubar=False for QG approx.
    # take time mean of relevant quantities
    print(datetime.now(), ' - taking time mean')
    div = div1 + div2
    div = div.mean(dim='time')
    ep1 = ep1.mean(dim='time')
    ep2 = ep2.mean(dim='time')
    return div, ep1, ep2

def plot_ep(uz, u, v, w, t, exp_name, heat, type, vertical):
    print(datetime.now(), " - calculating variables")
    p = uz.coords['pfull']
    lat = uz.coords['lat']
    ulvls = np.arange(-200, 200, 10)
    if type == 'a':
        div, ep1, ep2 = calc_ep(u, v, w, t)
        divlvls = np.arange(-12,13,1)

    elif type == 'b':
        div1, ep1a, ep2a = calc_ep(u[0], v[0], w[0], t[0])
        div2, ep1b, ep2b = calc_ep(u[1], v[1], w[1], t[1])
        print(datetime.now(), " - taking differences")
        div = div1 - div2
        ep1 = ep1a - ep1b
        ep2 = ep2a - ep2b
        divlvls = np.arange(-5,6,1)
        exp_name = exp_name+'_diff'

    #Filled contour plot of time-mean EP flux divergence plus EP flux arrows and zonal wind contours
    fig, ax = plt.subplots(figsize=(6,6), constrained_layout=True)
    print(datetime.now(), " - plot uz")
    uz.plot.contour(colors='k', linewidths=0.5, alpha=0.4, levels=ulvls)
    print(datetime.now(), " - plot polar heat")
    plt.contour(lat, p, heat, colors='g', linewidths=0.25, alpha=0.4, levels=11)
    print(datetime.now(), " - plot EP flux divergence")
    cs = div.plot.contourf(levels=divlvls, cmap='RdBu_r', add_colorbar=False)
    cb = plt.colorbar(cs)
    cb.set_label(label=r'Divergence (m s$^{-1}$ day$^{-1}$)', size='large')
    cb.ax.set_yticks(divlvls)
    fig.canvas.draw()
    ticklabs = cb.ax.get_yticklabels()
    cb.ax.set_yticklabels(ticklabs, fontsize='large')
    print(datetime.now(), " - plot EP flux arrows")
    ax = climate.PlotEPfluxArrows(lat, p, ep1, ep2, fig, ax, yscale='log')
    plt.yscale('log')
    plt.ylim(max(p), 1) #to 1 hPa
    plt.xlabel(r'Latitude ($\degree$N)', fontsize='xx-large')
    plt.xlim(0,90)
    plt.xticks([10, 30, 50, 70, 90], ['10', '30', '50', '70', '90'])
    plt.ylabel('Pressure (hPa)', fontsize='xx-large')
    plt.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
    plt.savefig(exp_name+'_EPflux.pdf', bbox_inches = 'tight')
    return plt.close()

def vT_full(exp):
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

def vT_level(exp, p):
    # Meridional heat flux weighted by cos(lat) and meridionally averaged from 75 to 90 N at p hPa
    # Based on Dunn-Sigouin & Shaw (2015) but their polar cap was 60 to 90 N
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
    vpTp_sub = vpTp_w.sel(lat=slice(75,90)).mean('lat')
    vpTp_bar = vpTp_sub.mean('lon')
    return vpTp_bar

def plot_vT(u, vpTp, exp, heat, lvls, colors):
    print(datetime.now(), " - plotting ", exp)
    fig, ax = plt.subplots(figsize=(6,6))
    cs1 = ax.contourf(lat, p, vpTp.mean('time'), levels=lvls, cmap=colors)
    ax.contourf(cs1, colors='none')
    cb = plt.colorbar(cs1)
    cb.set_label(label=r"v'T' (K m s$^{-1}$)", size='x-large')
    cb.ax.tick_params(labelsize='x-large')
    cs2 = ax.contour(lat, p, u[0], colors='k', levels=ulvls, linewidths=1, alpha=0.4)
    cs2.collections[int(len(ulvls)/2)].set_linewidth(1.5)
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

def plot_stats(stat, p, lab):
        print(datetime.now(), " - plotting v'T' ", lab)
        colors = ['#B30000', '#00B300', '#0099CC', 'k']
        fig, ax = plt.subplots(figsize=(8,6))
        for j in range(len(stat)):
            ax.plot(labels, stat[j], marker='o', linewidth=1.25, color=colors[j], linestyle=':', label='{:.0f} hPa'.format(p[j]))
        ax.set_xticks(labels)
        ax.set_xlabel(xlabel, fontsize='x-large')
        ax.set_ylabel("Polar Cap Average v'T' "+lab+r" (K m s$^{-1}$)", fontsize='x-large')
        ax.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
        plt.legend(loc='right',fancybox=False, shadow=True, ncol=1, fontsize='large')
        plt.savefig(exp+extension+'_vpTp_'+lab+'.pdf', bbox_inches = 'tight')
        return plt.close()

if __name__ == '__main__': 
    #Set-up data to be read in
    indir = '/disco/share/rm811/processed/'
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

    flux = input("Plot a) EP flux or b) heat flux?")
    ulvls = np.arange(-200, 200, 10)

    if flux == 'a':
        plot_type = input("Plot a) individuals experiment or b) difference vs. control?")

        for i in range(len(exp)):
            print(datetime.now(), " - opening files ({0:.0f}/{1:.0f})".format(i, len(exp)))
            if i == 0:
                print("skipping control")
            elif i != 0:
                # Read in data to plot polar heat contours
                file = '/disco/share/rm811/isca_data/' + exp[i] + '/run0100/atmos_daily_interp.nc'
                ds = xr.open_dataset(file)
                heat = ds.local_heating.sel(lon=180, method='nearest').mean(dim='time')    
                u = xr.open_dataset(indir+exp[i]+'_u.nc', decode_times=False).ucomp
                v = xr.open_dataset(indir+exp[i]+'_v.nc', decode_times=False).vcomp
                w = xr.open_dataset(indir+exp[i]+'_w.nc', decode_times=False).omega/100 # Pa --> hPa
                T = xr.open_dataset(indir+exp[i]+'_T.nc', decode_times=False).temp
                utz = xr.open_dataset(indir+exp[i]+'_utz.nc', decode_times=False).ucomp[0]
        
                if plot_type =='a':
                    plot_ep(utz, u, v, w, T, exp[i], heat, plot_type)
                elif plot_type == 'b':
                    u = [u, xr.open_dataset(indir+exp[0]+'_u.nc', decode_times=False).ucomp]
                    v = [v, xr.open_dataset(indir+exp[0]+'_v.nc', decode_times=False).vcomp]
                    w = [w, xr.open_dataset(indir+exp[0]+'_w.nc', decode_times=False).omega/100] # Pa --> hPa
                    T = [T, xr.open_dataset(indir+exp[0]+'_T.nc', decode_times=False).temp]
                    plot_ep(utz, u, v, w, T, exp[i], heat, plot_type)
    
    elif flux == 'b':
        colors = ['#B30000', '#FF9900', '#FFCC00', '#00B300', '#0099CC', '#4D0099', '#CC0080', '#666666']
        plot_option = input("Plot a) lat-p of v'T' and its s.d., \
            or b) PDFs of polar cap average v'T'?")

        if plot_option == 'a':
            vpTp = []
            for i in range(len(exp)):
                utz = xr.open_dataset(indir+exp[i]+'_utz.nc', decode_times=False).ucomp[0]
                vpTp_i = vT_full(exp[i])
                vpTp.append(vpTp_i)
                file = '/disco/share/rm811/isca_data/' + exp[i]+ '/run0100/atmos_daily_interp.nc'
                ds = xr.open_dataset(file)
                heat = ds.local_heating.sel(lon=180, method='nearest').mean(dim='time')
                lat = ds.coords['lat'].data
                p = ds.coords['pfull'].data
                plot_vT(utz, vpTp[i], exp[i], heat, np.arange(-20, 190, 10), 'Blues')
                plot_sd(lat, p, find_sd(vpTp_i), utz, 11, ulvls, 'RdBu_r', r"v'T' SD (K m s$^{-1}$)", exp[i]+'_vTsd.pdf')

                if i != 0:
                    vpTp_diff = vpTp[i] - vpTp[0]
                    vpTp_sd_diff = find_sd(vpTp[i]) - find_sd(vpTp[0])
                    plot_vT(utz, vpTp_diff, exp[i]+'_diff', heat, np.arange(-60, 65, 5), 'RdBu_r')
                    NH_zonal(lat, p, vpTp_sd_diff, utz, 11, ulvls, 'RdBu_r', r"v'T' SD (K m s$^{-1}$)", exp[i]+'_vTsd_diff.pdf') 
        
        elif plot_option == 'b':
            p = [500, 100, 50, 10]
            mean, mode, sd = plot_pdf('vT', indir, exp, '', p, labels, r"Polar Cap Average v'T' (K m s$^{-1}$)", colors, exp[0]+extension+'_vpTp')            
            plot_stats(mode, p, 'mode')
            plot_stats(sd, p, 'SD')
"""
# Following commented functions/code is for checking against Neil Lewis' code
def get_pt(t, p, Rd=287., cp=1005., p0=1000.): 
    #Neil's code
    return t * (p0/p)**(Rd/cp)

def TEM(ds, om=7.29e-5, a=6.371e6): 
    #Neil's code
    u = ds.ucomp 
    v = ds.vcomp 
    w = ds.omega 
    t = ds.temp 
    p = ds.pfull*100.
    latr = np.deg2rad(ds.lat)
    pt = get_pt(t, p, p0=1.e5)
    
    coslat = np.cos(latr)
    f = 2 * om * np.sin(latr)
    
    ub = u.mean('lon')
    vb = v.mean('lon')
    wb = w.mean('lon')
    ptb = pt.mean('lon')
    
    dub_dp = ub.differentiate('pfull', edge_order=2) / 100. # hPa -> Pa
    dptb_dp = ptb.differentiate('pfull', edge_order=2) / 100.
    
    up = u - ub 
    vp = v - vb
    wp = w - wb
    ptp = pt - ptb
    
    psi = (vp*ptp).mean('lon') / dptb_dp
    dpsi_dp = psi.differentiate('pfull', edge_order=2) / 100.
    
    F_lat =  (-(up*vp).mean('lon') + psi*dub_dp) * a * coslat
    F_p =  (-(up*wp).mean('lon') - psi * ((ub*coslat).differentiate('lat',edge_order=2)*180/np.pi / (a*coslat) - f)) * a * coslat
    
    v_star = vb - dpsi_dp 
    w_star = wb + (psi*coslat).differentiate('lat',edge_order=2)*180/np.pi / (a*coslat)
    
    return F_lat, F_p, v_star, w_star

def divF(F_lat, F_p, lat):
    a = 6.371e6 #earth radius m
    coslat = np.cos(lat)
    dF_lat = (coslat * F_lat).differentiate('lat',edge_order=2) * 1/(a*coslat) * (180./np.pi)
    dF_p = F_p.differentiate('pfull',edge_order=2) / 100.
    divF = dF_lat + dF_p
    return divF * 1/(a*coslat) * 1e5

def neil_plot(ds, p, lat):
    F_lat, F_p, v_star, w_star = TEM(ds)
    div_F = divF(F_lat, F_p, np.deg2rad(lat))
    div_F_mean = div_F.mean('time').transpose()

    fig, ax = plt.subplots(figsize=(6,6), constrained_layout=True)
    cs = plt.contourf(lat, p, div_F_mean, levels=np.arange(-15, 16, 1), cmap='RdBu_r', add_colorbar=False)
    cb = plt.colorbar(cs)
    plt.yscale('log')
    plt.ylim(max(p), 10) #to 10 hPa
    plt.xlim(0, 90)
    plt.xlabel(r'Latitude ($\degree$N)', fontsize='xx-large')
    plt.ylabel('Pressure (hPa)', fontsize='xx-large')
    plt.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
    plt.savefig('Neils_EPflux.pdf', bbox_inches = 'tight')
    return plt.close()
    
def aostools_plot(ds, p, lat):
    u = ds.ucomp
    v = ds.vcomp
    t = ds.temp
    w = ds.omega/100
    div, ep1, ep2 = calc_ep(u, v, w, t)

    fig, ax = plt.subplots(figsize=(6,6), constrained_layout=True)
    cs = plt.contourf(lat, p, div, levels=np.arange(-15, 16, 1), cmap='RdBu_r', add_colorbar=False)
    cb = plt.colorbar(cs)
    plt.yscale('log')
    plt.ylim(max(p), 10) #to 10 hPa
    plt.xlim(0, 90)
    plt.xlabel(r'Latitude ($\degree$N)', fontsize='xx-large')
    plt.ylabel('Pressure (hPa)', fontsize='xx-large')
    plt.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
    plt.savefig('aostools_EPflux.pdf', bbox_inches = 'tight')
    return plt.close()

#Compare Neil and aostools' code
file = '/scratch/rm811/isca_data/PK_e0v4z13_q6m2y45l800u200/run0001/atmos_daily.nc'
ds = xr.open_dataset(file, decode_times = False)
p = ds.coords['pfull']
lat = ds.coords['lat']
neil_plot(ds, p, lat)
aostools_plot(ds, p, lat)
"""
