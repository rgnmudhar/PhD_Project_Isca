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

def open_data(dir, exp):
    u = xr.open_dataset(dir+exp+'_u.nc', decode_times=False).ucomp
    v = xr.open_dataset(dir+exp+'_v.nc', decode_times=False).vcomp
    w = xr.open_dataset(dir+exp+'_w.nc', decode_times=False).omega/100 # Pa --> hPa
    T = xr.open_dataset(dir+exp+'_T.nc', decode_times=False).temp
    utz = xr.open_dataset(dir+exp+'_utz.nc', decode_times=False).ucomp[0]
    return utz, u, v, w, T

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

def uv_calc(exp):
    print(datetime.now(), " - opening files for ", exp)
    u = xr.open_dataset(indir+exp+'_u.nc', decode_times=False).ucomp
    v = xr.open_dataset(indir+exp+'_v.nc', decode_times=False).vcomp
    uz = xr.open_dataset(indir+exp+'_uz.nc', decode_times=False).ucomp
    vz = xr.open_dataset(indir+exp+'_vz.nc', decode_times=False).vcomp
    print(datetime.now(), " - finding anomalies")
    up = u - uz
    vp = v - vz
    print(datetime.now(), " - finding v'T'")
    upvp = (up*vp)        
    return upvp

def vT_level(vpTp, p):
    # Meridional heat flux weighted by cos(lat) and meridionally averaged from 75 to 90 N at p hPa
    # Based on Dunn-Sigouin & Shaw (2015) but their polar cap was 60 to 90 N
    vpTp_w = vpTp.sel(pfull=p, method='nearest') / np.cos(np.deg2rad(vpTp.lat))
    vpTp_sub = vpTp_w.sel(lat=slice(75,90)).mean('lat')
    vpTp_bar = vpTp_sub.mean('lon')
    return vpTp_bar

def comparison(var, lats):
    print(datetime.now(), " - addition")
    if lats == 60:
        compare = [var[0].sel(lat=lats, method='nearest').mean(('lon', 'time')),\
            var[1].sel(lat=lats, method='nearest').mean(('lon', 'time')),\
            var[2].sel(lat=lats, method='nearest').mean(('lon', 'time'))]
    else:
        compare = [var[0].sel(lat=lats).mean(('lat', 'lon', 'time')),\
            var[1].sel(lat=lats).mean(('lat', 'lon', 'time')),\
            var[2].sel(lat=lats).mean(('lat', 'lon', 'time'))]
    compare.append(compare[0]+compare[1]) # addition after taking means
    compare.append(compare[3]-compare[2])
    return compare

def linear_add(compare, p, label, lats_label):
    xlabel = lats_label+r" mean v'T' magnitude (K m s$^{-1}$)"
    names = ['mid-lat heat only (a)', 'polar heat only (b)', 'combined simulation (c)', 'linear component (d=a+b)', '-1 x non-linear component -(c-d)']
    colors = ['#B30000', '#0099CC', 'k', '#4D0099', '#CC0080']
    lines = ['--', ':', '-', '-.', ':']   
    print(datetime.now(), " - plotting")
    fig, ax = plt.subplots(figsize=(8,5.5))
    for i in range(len(compare)):
        ax.plot(compare[i].transpose(), p, color=colors[i], linestyle=lines[i], label=names[i], linewidth=1.75)
    ax.set_xlim(-5, 125)
    ax.axvline(0, color='k', linewidth=0.25)
    ax.set_xlabel(xlabel, fontsize='large')
    ax.set_ylabel('Pressure (hPa)', fontsize='large')
    ax.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
    plt.legend(fancybox=False, ncol=1, loc='lower right', fontsize='large', labelcolor = colors)
    plt.ylim(max(p), 1)
    plt.yscale('log')
    plt.savefig('vT_addvcombo_'+label+lats_label+'.pdf', bbox_inches = 'tight')
    return plt.close()

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

def check_vs_MERRA(exp):
    print(datetime.now(), " - setting up MERRA2 data")
    vars = ['uv', 'vT', 'z1']
    merra2_p = [100, 70, 50, 30, 10]
    merra2_uv_4575mean = [-3.64865774378585, -2.9196845124283, 1.50874952198854, 17.1705812619504, 92.7788451242831]
    merra2_vT_4575mean = [17.2502638623327, 21.3728413001912, 26.7985621414914, 38.4797055449332, 75.6868871892927]
    merra1_z1_60mean = [198.617317399618, 251.389999999999, 319.728965583174, 451.971359464627, 781.923244741875]
    merra2_uv_4575sd = [18.622067598426, 20.5531656863889, 25.1494305485741, 37.5029171205106, 86.9587997758016]
    merra2_vT_4575sd = [11.756538354396, 15.1474702772449, 19.7676915762039, 30.2586902501783, 70.0534319324615]
    merra1_z1_60sd = [96.9030853954761, 125.266337610638, 158.865054753343, 218.504235655414, 357.295161771025]
    merra2_data = [merra2_uv_4575mean, merra2_vT_4575mean, merra1_z1_60mean]
    merra2_sd = [merra2_uv_4575sd, merra2_vT_4575sd, merra1_z1_60sd]

    print(datetime.now(), ' - opening Isca data')
    uv = uv_calc(exp)
    vT = vT_calc(exp)
    print(datetime.now(), ' - wave decomposition')
    z_60 = xr.open_dataset(indir+exp+'_h.nc', decode_times=False).height.sel(lat=60, method='nearest')
    waves = climate.GetWavesXr(z_60)
    z1_60mean = np.abs(waves.sel(k=1)).mean(('lon', 'time'))
    p = uv.pfull
    uv_4575mean = uv.sel(lat=slice(45,75)).mean(('lat', 'lon', 'time'))
    vT_4575mean = vT.sel(lat=slice(45,75)).mean(('lat', 'lon', 'time'))
    isca_data = [uv_4575mean, vT_4575mean, z1_60mean]

    print(datetime.now(), " - plotting")
    colors = ['k', '#B30000']
    lines = ['--', '-']
    labels = ['MERRA2 data', 'control simulation']
    units = [r"v'T' (K m s$^{-1}$)", r"u'v' (m$^{2}$ s$^{-2}$)", 'Wave-1 GPH (m)']
    for i in range(len(vars)):
        fig, ax = plt.subplots(figsize=(6,6))
        ax.errorbar(merra2_data[i], merra2_p, xerr=merra2_sd[i], fmt='.', linewidth=1.25, capsize=5, color=colors[0], linestyle=lines[0], label=labels[0])
        ax.plot(isca_data[i], p, linewidth=1.25, color=colors[1], linestyle=lines[1], label=labels[1])
        plt.xlabel(r"$45-75\degree$N mean "+units[i], fontsize='x-large')
        plt.ylabel('Pressure (hPa)', fontsize='x-large')
        plt.ylim(max(p), 1)
        plt.yscale('log')
        plt.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
        plt.legend(fancybox=False, ncol=1, fontsize='x-large')
        plt.savefig(exp+'_'+vars[i]+'_vsMERRA2.pdf', bbox_inches = 'tight')
        plt.show()

if __name__ == '__main__': 
    #Set-up data to be read in
    indir = '/disco/share/rm811/processed/'
    basis = 'PK_e0v4z13'
    flux = input("Plot a) EP flux divergence, b) upward EP Flux or c) v'T'?")
    var_type = input("Plot a) depth, b) width, c) location, d) strength, or e) topography experiments?")
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
    colors = ['k', '#B30000', '#FF9900', '#FFCC00', '#00B300', '#0099CC', '#4D0099', '#CC0080']
    blues = ['k', '#dbe9f6', '#bbd6eb', '#88bedc', '#549ecd',  '#2a7aba', '#0c56a0', '#08306b']

    ulvls = np.arange(-200, 200, 10)

    if flux == 'a':
        for i in range(len(exp)):
            print(datetime.now(), " - opening files ({0:.0f}/{1:.0f})".format(i+1, len(exp)))
            utz, u, v, w, T = open_data(indir, exp[i])
            div, ep1, ep2 = calc_ep(u, v, w, T)

            if i == 0:
                print("skipping control")
                div_og = div
                ep1_og = ep1
                ep2_og = ep2
            elif i != 0:
                # Read in data to plot polar heat contours
                file = '/disco/share/rm811/isca_data/' + exp[i] + '/run0100/atmos_daily_interp.nc'
                ds = xr.open_dataset(file)
                heat = ds.local_heating.sel(lon=180, method='nearest').mean(dim='time')  
                print(datetime.now(), " - plotting")
                plot_ep(utz, div, ep1, ep2, exp[i], heat, 'single')

                print(datetime.now(), " - taking differences")
                div_diff = div - div_og
                ep1_diff = ep1 - ep1_og
                ep2_diff = ep2 - ep2_og
                plot_ep(utz, div_diff, ep1_diff, ep2_diff, exp[i], heat, 'diff')

    elif flux == 'b':
        p = int(input('At which pressure level? (i.e. 10 or 100 hPa) '))
        print(datetime.now(), " - plotting PDFs at {:.0f} hPa".format(p))
        x_min = x_max = 0
        fig, ax = plt.subplots(figsize=(6,6))
        for i in range(len(exp)):
            print(datetime.now(), " - opening files ({0:.0f}/{1:.0f})".format(i+1, len(exp)))
            u, v, w, t = open_data(indir, exp[i])[1:]
            print(datetime.now(), " - finding EP flux")
            ep1, ep2, div1, div2 = climate.ComputeEPfluxDivXr(u, v, t, 'lon', 'lat', 'pfull', 'time', w=w, do_ubar=True)
            x = ep2.sel(pfull=p,method='nearest').sel(lat=slice(45,75)).mean('lat')
            x_sort, f, m = pdf(x)
            if max(x) > x_max:
                x_max = max(x)
            if min(x) < x_min:
                x_min = min(x)
            print(datetime.now(), ' - plotting')
            ax.plot(x_sort, f, linewidth=1.25, color=blues[i], label=labels[i])
        ax.axvline(0, color='k', linewidth=0.25)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r'45-75$\degree$N,'+str(p)+r'hPa EP$_z$ (hPa m s$^{-2}$)', fontsize='xx-large')
        ax.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
        plt.legend(fancybox=False, ncol=1, fontsize='x-large')
        plt.savefig(basis+'_{:.0f}pdf.pdf'.format(p), bbox_inches = 'tight')
        plt.show()
        plt.close()
            
    elif flux == 'c':
        plot_type = input("Plot a) lat-p climatology and variability or b) linear addition?")
        if plot_type == 'a':
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
                    print("skipping control")

                elif i != 0:
                    #Read in data to plot polar heat contours
                    file = '/disco/share/rm811/isca_data/' + exp[i]+ '/run0100/atmos_daily_interp.nc'
                    ds = xr.open_dataset(file)
                    heat = ds.local_heating.sel(lon=180, method='nearest').mean(dim='time')

                    print(datetime.now(), " - plotting vT")
                    plot_vT(utz, vT_itz, exp[i], heat, np.arange(-20, 190, 10), 'Blues')

                    print(datetime.now(), " - plotting s.d.")
                    NH_zonal(lat, p, sd, utz, np.arange(0, 300, 20), ulvls, 'Blues', r"v'T' SD (K m s$^{-1}$)", exp[i]+'_vTsd.pdf')

                    vT_diff = vT_itz - vT_itz_og
                    vT_sd_diff = sd - sd_og
                    plot_vT(utz, vT_diff, exp[i]+'_diff', heat, np.arange(-40, 42, 2), 'RdBu_r')
                    NH_zonal(lat, p, vT_sd_diff, utz, np.arange(-45, 50, 5), ulvls, 'RdBu_r', r"v'T' SD (K m s$^{-1}$)", exp[i]+'_vTsd_diff.pdf') 

            p = [500, 100, 50, 10]
            me, mo, sd, e, sk, k = plot_pdf('vT', indir, exp, '', vpTp, p, labels, r"75-90N average v'T' (K m s$^{-1}$)", colors, exp[0]+extension+'_vT')    
            plot_stats(mo, p, exp[0], extension, 'mode')
            plot_stats(sd, p, exp[0], extension, 'SD')
        
        elif plot_type == 'b':
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
           
            print(datetime.now(), " - opening files")
            vT_exp = []
            for i in range(len(exp)):
                vT = vT_calc(exp[i])
                vT_exp.append(vT)
            p = vT.pfull

            # polar cap average following Dunn-Sigouin and Shaw (2015) for meridional heat flux
            # mid-latitude average following NASA Ozone watch vT 
            lats = [60, slice(60, 90), slice(45, 75)]
            lats_labels = [r'$60\degree$N', 'polar cap', r'$45-75\degree$N']
            for i in range(len(lats)):
                compare = comparison(vT_exp, lats[i])
                linear_add(compare, p, label, lats_labels[i]) 

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
