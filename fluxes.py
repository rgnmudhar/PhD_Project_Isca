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

def plot_ep(uz, div, ep1, ep2, exp_name, heat, type):
    print(datetime.now(), " - plotting EP Fluxes")
    p = uz.coords['pfull']
    lat = uz.coords['lat']
    ulvls = np.arange(-200, 200, 10)
    if type == 'single':
        divlvls = np.arange(-12,13,1)

    elif type == 'diff':
        divlvls = np.arange(-5,5.5,0.5)
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
    plt.close()
    return div, ep1

def vT_calc(exp):
    print(datetime.now(), " - opening files for ", exp)
    v = xr.open_dataset(indir+exp+'_v.nc', decode_times=False).vcomp
    T = xr.open_dataset(indir+exp+'_T.nc', decode_times=False).temp
    vz = xr.open_dataset(indir+exp+'_vz.nc', decode_times=False).vcomp
    Tz = xr.open_dataset(indir+exp+'_Tz.nc', decode_times=False).temp
    print(datetime.now(), " - finding anomalies")
    vp = v - vz
    Tp = T - Tz
    print(datetime.now(), " - finding v'T'")
    vpTp = (vp*Tp)        
    return vpTp

def vT_level(vpTp, p):
    # Meridional heat flux weighted by cos(lat) and meridionally averaged from 75 to 90 N at p hPa
    # Based on Dunn-Sigouin & Shaw (2015) but their polar cap was 60 to 90 N
    vpTp_w = vpTp.sel(pfull=p, method='nearest') / np.cos(np.deg2rad(vpTp.lat))
    vpTp_sub = vpTp_w.sel(lat=slice(75,90)).mean('lat')
    vpTp_bar = vpTp_sub.mean('lon')
    return vpTp_bar

def plot_vT(u, vT, exp, heat, lvls, colors):
    fig, ax = plt.subplots(figsize=(6,6))
    cs1 = ax.contourf(lat, p, vT, levels=lvls, cmap=colors)
    ax.contourf(cs1, colors='none')
    cb = plt.colorbar(cs1)
    cb.set_label(label=r"v'T' (K m s$^{-1}$)", size='x-large')
    cb.ax.tick_params(labelsize='x-large')
    cs2 = ax.contour(lat, p, u, colors='k', levels=ulvls, linewidths=0.5, alpha=0.2)
    cs2.collections[int(len(ulvls)/2)].set_linewidth(1)
    plt.contour(lat, p, heat, colors='g', linewidths=1, alpha=0.4, levels=11)
    plt.xlabel(r'Latitude ($\degree$N)', fontsize='x-large')
    plt.xlim(0,90)
    plt.xticks([10, 30, 50, 70, 90], ['10', '30', '50', '70', '90'])
    plt.ylabel('Pressure (hPa)', fontsize='x-large')
    plt.ylim(max(p), 1) #goes to ~1hPa
    plt.yscale('log')
    plt.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    plt.savefig(exp+'_vT.pdf', bbox_inches = 'tight')
    return plt.close()

def plot_stats(stat, p, exp, ext, lab):
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
        plt.savefig(exp+ext+'_vT_'+lab+'.pdf', bbox_inches = 'tight')
        return plt.close()

if __name__ == '__main__': 
    #Set-up data to be read in
    indir = '/disco/share/rm811/processed/'
    basis = 'PK_e0v4z13'
    var_type = input("Plot a) depth, b) width, c) location, d) strength experiments, or e) test?")
    if var_type == 'a':
        extension = '_depth'
    elif var_type == 'b':
        extension = '_width'
    elif var_type == 'c':
        extension = '_loc'
    elif var_type == 'd':
        extension = '_strength'
    elif var_type == 'e':
        extension = 'ctrl'
    exp, labels, xlabel = return_exp(extension)

    flux = input("Plot a) EP flux or b) heat flux?")
    ulvls = np.arange(-200, 200, 10)

    if flux == 'a':
        for i in range(len(exp)):
            print(datetime.now(), " - opening files ({0:.0f}/{1:.0f})".format(i+1, len(exp)))
            # Read in data to plot polar heat contours
            file = '/disco/share/rm811/isca_data/' + exp[i] + '/run0100/atmos_daily_interp.nc'
            ds = xr.open_dataset(file)
            heat = ds.local_heating.sel(lon=180, method='nearest').mean(dim='time')    
            u = xr.open_dataset(indir+exp[i]+'_u.nc', decode_times=False).ucomp
            v = xr.open_dataset(indir+exp[i]+'_v.nc', decode_times=False).vcomp
            w = xr.open_dataset(indir+exp[i]+'_w.nc', decode_times=False).omega/100 # Pa --> hPa
            T = xr.open_dataset(indir+exp[i]+'_T.nc', decode_times=False).temp
            utz = xr.open_dataset(indir+exp[i]+'_utz.nc', decode_times=False).ucomp[0]
            div, ep1, ep2 = calc_ep(u, v, w, T)

            print(datetime.now(), " - plotting")
            plot_ep(utz, div, ep1, ep2, exp[i], heat, 'single')

            if i == 0:
                print("skipping control")
                div_og = div
                ep1_og = ep1
                ep2_og = ep2
            elif i != 0:
                print(datetime.now(), " - taking differences")
                div_diff = div - div_og
                ep1_diff = ep1 - ep1_og
                ep2_diff = ep2 - ep2_og
                plot_ep(utz, div_diff, ep1_diff, ep2_diff, exp[i], heat, 'diff')
        
    elif flux == 'b':
        colors = ['#B30000', '#FF9900', '#FFCC00', '#00B300', '#0099CC', '#4D0099', '#CC0080', '#666666']
        vpTp = []
        for i in range(len(exp)):
            utz = xr.open_dataset(indir+exp[i]+'_utz.nc', decode_times=False).ucomp[0]
            vT = vT_calc(exp[i])
            vpTp.append(vT)
            vT_iz = vpTp[i].mean('lon')
            vT_itz = vT_iz.mean('time')
            lat, p, sd = find_sd(vT_iz)
            if i == 0:
                sd_og = sd
                vT_itz_og = vT_itz
            #Read in data to plot polar heat contours
            file = '/disco/share/rm811/isca_data/' + exp[i]+ '/run0100/atmos_daily_interp.nc'
            ds = xr.open_dataset(file)
            heat = ds.local_heating.sel(lon=180, method='nearest').mean(dim='time')
            
            print(datetime.now(), " - plotting vT")
            plot_vT(utz, vT_itz, exp[i], heat, np.arange(-20, 190, 10), 'Blues')

            print(datetime.now(), " - plotting s.d.")
            NH_zonal(lat, p, sd, utz, np.arange(0, 300, 20), ulvls, 'Blues', r"v'T' SD (K m s$^{-1}$)", exp[i]+'_vTsd.pdf')

            if i != 0:
                vT_diff = vT_itz - vT_itz_og
                vT_sd_diff = sd - sd_og
                plot_vT(utz, vT_diff, exp[i]+'_diff', heat, np.arange(-40, 42, 2), 'RdBu_r')
                NH_zonal(lat, p, vT_sd_diff, utz, np.arange(-45, 50, 5), ulvls, 'RdBu_r', r"v'T' SD (K m s$^{-1}$)", exp[i]+'_vTsd_diff.pdf') 

        p = [500, 100, 50, 10]
        me, mo, sd, e, sk, k = plot_pdf('vT', indir, exp, '', vpTp, p, labels, r"75-90N average v'T' (K m s$^{-1}$)", colors, exp[0]+extension+'_vT')    
        plot_stats(mo, p, exp[0], extension, 'mode')
        plot_stats(sd, p, exp[0], extension, 'SD')

"""
exp = [basis+'_a4x75y180w5v30p800', basis+'_q6m2y45l800u200', basis+'_a4x75y180w5v30p800'+'_q6m2y45l800u200']
for i in range(len(exp)):
    vT = vT_calc(exp[i])
    
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
