"""
    This script plots (time and zonal) averages of various output variables averaged over X years'-worth of data from Isca
    Also plots differences between 2 datasets - important that they are of the same resolution (e.g. both T21 or both T42)
    TEST
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cm
from shared_functions import *
from datetime import datetime

def zero_wind(exp, labels, name):
    # Shows difference in zero wind line between no polar heat and with polar heat
    print(datetime.now(), " - opening files")
    X0 = []
    X1 = []
    n = len(exp[0])
    for i in range(n):
        Xtz0 = xr.open_dataset(indir+exp[0][i]+'_utz.nc', decode_times=False).ucomp[0]
        Xtz1 = xr.open_dataset(indir+exp[1][i]+'_utz.nc', decode_times=False).ucomp[0]
        X0.append(Xtz0)
        X1.append(Xtz1)

    p = Xtz0.pfull
    lat = Xtz0.lat
    lvls = np.arange(-60, 170, 10)

    h_name = exp[1][0][11:27]
    h = xr.open_dataset('../Inputs/' + h_name + '.nc')
    heat = h.mean('lon').variables[h_name]
    h_p = h.pfull
    h_lat = h.lat
    h_lvls = np.arange(2.5e-6, 1e-4, 5e-6)

    print(datetime.now(), " - plotting")
    fig, axes = plt.subplots(1, n, figsize=(n*5,7))
    norm = cm.TwoSlopeNorm(vmin=min(lvls), vmax=max(lvls), vcenter=0)
    axes[0].set_ylabel('Pressure (hPa)', fontsize='xx-large')
    for i in range(n):
        csa = axes[i].contourf(lat, p, X0[i], levels=lvls, norm=norm, cmap='RdBu_r', extend='both')
        csa_l = axes[i].contour(lat, p, X0[i], levels=lvls, norm=norm, colors='r', alpha=0)
        csb = axes[i].contour(lat, p, X1[i], colors='k', levels=lvls, linewidths=1.5, alpha=0.25)
        csa_l.collections[list(lvls).index(0)].set_alpha(0.5)
        csb.collections[list(lvls).index(0)].set_linewidth(3)
        axes[i].contour(h_lat, h_p, heat, alpha=0.5, colors='g', levels=h_lvls)
        axes[i].text(2, 1.75, labels[i], color='k', fontsize='xx-large')
        axes[i].set_ylim(max(p), 1) #goes to ~1hPa
        axes[i].set_yscale('log')
        axes[i].set_xlabel(r'Latitude ($\degree$N)', fontsize='xx-large')
        axes[i].set_xlim(0, max(lat))
        axes[i].set_xticks([0, 20, 40, 60, 80], ['0', '20', '40', '60', '80'])
        axes[i].tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
        if i > 0:
            axes[i].tick_params(axis='y',label1On=False)
    cb  = fig.colorbar(csa, ax=axes[:], shrink=0.2, orientation='horizontal', extend='both', pad=0.15)
    cb.set_label(label=r'Zonal Wind (m s$^{-1}$)', size='xx-large')
    cb.ax.tick_params(labelsize='x-large')        
    plt.savefig(name+'_0wind.pdf', bbox_inches = 'tight')
    return plt.close()

def report_plot1(exp, lvls, variable, unit, labels, name):
    # Plots difference between no polar heat and with polar heat
    print(datetime.now(), " - opening files")
    X = []
    X_response = []
    n = len(exp[0])
    for i in range(n):
        if variable == 'Temperature':
            Xtz0 = xr.open_dataset(indir+exp[0][i]+'_Ttz.nc', decode_times=False).temp[0]
            Xtz1 = xr.open_dataset(indir+exp[1][i]+'_Ttz.nc', decode_times=False).temp[0]
            X_response.append(Xtz1 - Xtz0)
            X.append(Xtz1)
        elif variable == 'Zonal Wind':
            Xtz0 = xr.open_dataset(indir+exp[0][i]+'_utz.nc', decode_times=False).ucomp[0]
            Xtz1 = xr.open_dataset(indir+exp[1][i]+'_utz.nc', decode_times=False).ucomp[0]
            X_response.append(Xtz1 - Xtz0)
            X.append(Xtz1)

    p = Xtz0.pfull
    lat = Xtz0.lat

    h_name = exp[1][0][11:27]
    h = xr.open_dataset('../Inputs/' + h_name + '.nc')
    heat = h.mean('lon').variables[h_name]
    h_p = h.pfull
    h_lat = h.lat
    h_lvls = np.arange(2.5e-6, 1e-4, 5e-6)

    print(datetime.now(), " - plotting")
    fig, axes = plt.subplots(1, n, figsize=(n*4.5,5), layout="constrained")
    norm = cm.TwoSlopeNorm(vmin=min(lvls[1]), vmax=max(lvls[1]), vcenter=0)
    axes[0].set_ylabel('Pressure (hPa)', fontsize='xx-large')
    for i in range(n):
        csa = axes[i].contourf(lat, p, X_response[i], levels=lvls[1], norm=norm, cmap='RdBu_r', extend='both')
        csb = axes[i].contour(lat, p, X[i], colors='k', levels=lvls[0], linewidths=1.5, alpha=0.25)
        if variable == 'Zonal Wind':
            csb.collections[list(lvls[0]).index(0)].set_linewidth(3)
        axes[i].contour(h_lat, h_p, heat, alpha=0.5, colors='g', levels=h_lvls)
        axes[i].scatter(60, 10, marker='x', color='k')
        axes[i].text(2, 1.75, labels[i], color='k', fontsize='xx-large')
        axes[i].set_ylim(max(p), 1) #goes to ~1hPa
        axes[i].set_yscale('log')
        axes[i].set_xlabel(r'Latitude ($\degree$N)', fontsize='xx-large')
        axes[i].set_xlim(0, max(lat))
        axes[i].set_xticks([0, 20, 40, 60, 80], ['0', '20', '40', '60', '80'])
        axes[i].tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
        if i > 0:
            axes[i].tick_params(axis='y',label1On=False)
    #cb  = fig.colorbar(csa, ax=axes[:], shrink=0.2, orientation='horizontal', extend='both', pad=0.15)
    cb  = fig.colorbar(csa, orientation='vertical', extend='both', pad=0.1)
    cb.set_label(label=variable+' Response'+unit, size='xx-large')
    cb.ax.tick_params(labelsize='x-large')
    #fig.get_layout_engine()
    plt.savefig(name+'.pdf', bbox_inches = 'tight')
    return plt.close()

def report_plot2(exp, lvls, variable, unit, labels, name):
    # Plots control, then 3 experiments of choice
    print(datetime.now(), " - opening files")
    X = []
    X_response = []
    heat = []
    for i in range(len(exp)):
        if variable == 'Temperature':
            Xtz = xr.open_dataset(indir+exp[i]+'_Ttz.nc', decode_times=False).temp[0]
        elif variable == 'Zonal Wind':
                Xtz = xr.open_dataset(indir+exp[i] +'_utz.nc', decode_times=False).ucomp[0]
        X.append(Xtz)
        if i == 0:
            ctrl = Xtz
        else:
            X_response.append(Xtz - ctrl)
            h_name = exp[i][11:27]
            h = xr.open_dataset('../Inputs/' + h_name + '.nc')
            heat.append(h.mean('lon').variables[h_name])

    p = ctrl.pfull
    lat = ctrl.lat
    h_p = h.pfull
    h_lat = h.lat
    h_lvls = np.arange(2.5e-6, 1e-4, 5e-6)

    print(datetime.now(), " - plotting")
    fig, axes = plt.subplots(1, 4, figsize=(4*4,6), layout="constrained")
    if variable == 'Temperature':
        csa_ctrl = axes[0].contourf(lat, p, ctrl, levels=lvls[0], cmap='Blues_r')
    elif variable == 'Zonal Wind':
        norm = cm.TwoSlopeNorm(vmin=min(lvls[0]), vmax=max(lvls[0]), vcenter=0)
        csa_ctrl = axes[0].contourf(lat, p, ctrl, levels=lvls[0], norm=norm, cmap='RdBu_r')
    cb_ctrl  = fig.colorbar(csa_ctrl, ax=axes[0], orientation='horizontal', extend='both')
    cb_ctrl.set_label(label=variable+unit, size='xx-large')
    cb_ctrl.ax.tick_params(labelsize='x-large')
    axes[0].set_ylabel('Pressure (hPa)', fontsize='xx-large')
    
    norm = cm.TwoSlopeNorm(vmin=min(lvls[1]), vmax=max(lvls[1]), vcenter=0)
    for i in range(len(exp)):
        if i > 0:
            csa = axes[i].contourf(lat, p, X_response[i-1], levels=lvls[1], norm=norm, cmap='RdBu_r')
            csb = axes[i].contour(lat, p, X[i], colors='k', levels=lvls[2], linewidths=1.5, alpha=0.25)
            if variable == 'Zonal Wind':
                csb.collections[list(lvls[2]).index(0)].set_linewidth(3)
            axes[i].contour(h_lat, h_p, heat[i-1], alpha=0.5, colors='g', levels=h_lvls)

    cb  = fig.colorbar(csa, ax=axes[1:], shrink=0.3, orientation='horizontal', extend='both')
    cb.set_label(label='Response'+unit, size='xx-large')
    cb.ax.tick_params(labelsize='x-large')

    for i in range(len(axes)):
        axes[i].scatter(60, 10, marker='x', color='k')
        axes[i].text(2, 1.75, labels[i], color='k', fontsize='xx-large')
        axes[i].set_ylim(max(p), 1) #goes to ~1hPa
        axes[i].set_yscale('log')
        axes[i].set_xlabel(r'Latitude ($\degree$N)', fontsize='xx-large')
        axes[i].set_xlim(0, max(lat))
        axes[i].set_xticks([0, 20, 40, 60, 80], ['0', '20', '40', '60', '80'])
        axes[i].tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
        if i > 0:
            axes[i].tick_params(axis='y',label1On=False)
    plt.savefig(name+'.pdf', bbox_inches = 'tight')
    return plt.close()

def plot_combo(u, T, lvls, perturb, lat, p, exp_name, vertical):
    # Plots time and zonal-mean Zonal Wind Speed and Temperature
    print(datetime.now(), " - plotting")
    fig, ax = plt.subplots(figsize=(6,6))

    if vertical == "a":
        csa = ax.contourf(lat, p, T, levels=lvls[0], cmap='Blues_r')
        csb = ax.contour(lat, p, u, colors='k', levels=lvls[1], linewidths=1)
        #plt.contour(lat, p, perturb, colors='g', linewidths=1, alpha=0.4, levels=11)
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
    plt.xlim(0, max(lat))
    plt.xticks([0, 20, 40, 60, 80], ['0', '20', '40', '60', '80'])
    plt.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    plt.savefig(exp_name+'_zonal.pdf', bbox_inches = 'tight')

    return plt.close()

def plot_diff(vars, units, lvls, perturb, lat, p, exp_name, vertical):
    # Plots differences in time and zonal-mean of variables (vars)
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
        plt.xlim(0, max(lat))
        plt.xticks([0, 20, 40, 60, 80], ['0', '20', '40', '60', '80'])
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

def min_lapse(T, z):
    # Finds where lapse rate reaches < 2 K/km (in hPa)
    dtdz = []
    for i in range(len(T)-1):
        dtdz.append( -(T[i+1] - T[i]) / (z[i+1] - z[i]) )
    for j in range(len(dtdz)):
        if dtdz[j] < 2:
            idx = (np.abs(z - (z[j]+2)).argmin())
            if dtdz[idx] < 2:
                h_tropo = z[j]
                break
    return h_tropo

def tropopause(exp):
    # Finds tropopause
    T = xr.open_dataset(indir+exp+'_Ttz.nc', decode_times=False).temp[0]
    T_sort = T.transpose().reindex(pfull=list(reversed(T.pfull)))
    p = T.pfull
    z = list(reversed(altitude(p).data))
    tropo = []
    for i in range(len(T_sort)):
        tropo.append(min_lapse(T_sort[i], z))
    return z, T.lat.data, tropo

def merid_Tgrad(exp, lat_min):
    lat_max = 90
    T = xr.open_dataset(indir+exp+'_Ttz.nc', decode_times=False).temp[0].sel(pfull=slice(1,1000))
    p = T.pfull
    T1 = T.sel(lat=lat_min, method='nearest')
    T2 = T.sel(lat=lat_max, method='nearest')
    grads = []
    for i in range(len(p)):
        grads.append(T1[i] - T2[i])
    return p, grads

if __name__ == '__main__': 
    #Set-up data to be read in
    indir = '/disco/share/rm811/processed/'
    basis = 'PK_e0v4z13'
    plot_type = input("Plot a) individual experiments, b) difference vs. control, c) linear additions, d) meridional T gradients, e) tropopause, f) zero wind check, or g) paper plot?")
   
    if plot_type == 'a' or plot_type == 'b' or plot_type == 'd' or plot_type == 'e' or plot_type == 'f' or plot_type == 'g':
        var_type = input("Plot a) depth, b) width, c) location, d) strength, e) vortex experiments or f) test?")
        if var_type == 'a':
            extension = '_depth'
        elif var_type == 'b':
            extension = '_width'
        elif var_type == 'c':
            extension = '_loc2'
        elif var_type == 'd':
            extension = '_strength'
        elif var_type == 'e':
            basis = 'PK_e0vXz13'
            extension = '_vtx'
        elif var_type == 'f':
            extension = '_test'
        exp, labels, xlabel = return_exp(extension)
        upper_p = 1 # hPa
        lvls = [np.arange(160, 330, 10), np.arange(-200, 205, 5)]
        blues = ['k', '#dbe9f6', '#bbd6eb', '#88bedc', '#549ecd',  '#2a7aba', '#0c56a0', '#08306b']

        if plot_type =='d':
            lat_min = 0
            fig, ax = plt.subplots(figsize=(6,6))
            for i in range(len(exp)):
                p, grads = merid_Tgrad(exp[i], lat_min)
                ax.plot(grads, p, linewidth=1.25, color=blues[i], label=labels[i])
            ax.axvline(0, color='k', linewidth=0.25)
            ax.fill_between(range(-30,100), 200, 800, facecolor ='gainsboro', alpha = 0.2)
            ax.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
            plt.xlim(-15,35)
            plt.xlabel(str(lat_min)+r'$-90\degree$N $\Delta T_{y}$ (K)', fontsize='xx-large')
            plt.ylim(max(p), 50)
            plt.yscale('log')
            plt.ylabel('pressure (hPa)', fontsize='xx-large')
            plt.legend(fancybox=False, ncol=1, fontsize='x-large')
            plt.savefig('Tgradv{:.0f}N'.format(lat_min)+extension+'.pdf', bbox_inches = 'tight')
            plt.show()
            plt.close()
        
        elif plot_type =='e':
            # Below is experiment before zonal asymmetry was added for reference
            exp.append('PK_e0v4z13')
            labels.append('original')
            blues.append('#B30000')

            fig, ax = plt.subplots(figsize=(6,6))
            for i in range(len(exp)):
                print(datetime.now(), " - finding tropopause ({0:.0f}/{1:.0f})".format(i+1, len(exp)))
                z, lats, trop = tropopause(exp[i])
                ax.plot(lats, trop, linewidth=1.25, color=blues[i], label=labels[i])
            ax.set_ylim(min(z), 25)
            ax.set_ylabel('psuedo-altitude (km)', fontsize='xx-large')
            ax.set_xlabel(r'latitude ($\degree$N)', fontsize='xx-large')
            ax.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
            plt.legend(fancybox=False, ncol=1, fontsize='x-large', facecolor='white')

            ax2 = ax.twinx()
            file = '/disco/share/rm811/isca_data/' + exp[0] + '/run0025/atmos_daily_interp.nc'
            ds = xr.open_dataset(file)
            heat = ds.local_heating.sel(lon=180, method='nearest').mean('time')
            ax2.contour(lats, heat.pfull, heat, colors='g', linewidths=1, alpha=0.25, levels=7)
            ax2.set_ylim(inv_altitude(min(z)), inv_altitude(30))
            ax2.set_ylabel('pressure (hPa)', fontsize='xx-large')
            ax2.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')

            plt.xlim(0,90)
            plt.xticks([10, 30, 50, 70, 90], ['10', '30', '50', '70', '90'])
            plt.savefig('tropopause'+extension+'.pdf', bbox_inches = 'tight')
            plt.show()
            plt.close()

        elif plot_type == 'f':
            if extension == '_vtx' or '_loc2':
                zero_wind(exp, labels, basis+extension)
            else:
                print("Not set up to plot these experiments yet!")

        elif plot_type == 'g':
            if var_type == 'e':
                # For polar vortex experiments:
                T_lvls = [np.arange(160, 330, 10), np.arange(-20, 30, 2.5)]
                u_lvls = [np.arange(-70, 100, 10), np.arange(-40, 30, 5)]
                #exp = [[exp[0][1], exp[0][3], exp[0][-1]], [exp[1][1], exp[1][3], exp[1][-1]]]
                #labels = [labels[1], labels[3], labels[-1]]
                report_plot1(exp, T_lvls, 'Temperature', ' (K)', labels, basis+extension+'_T')
                report_plot1(exp, u_lvls, 'Zonal Wind', r' (m s$^{-1}$)', labels, basis+extension+'_u')
            else:
                # For polar heat experiments:
                exp = [exp[0], exp[1], exp[4], exp[-1]]
                labels = [labels[0], labels[1], labels[4], labels[-1]]
                T_lvls = [np.arange(160, 330, 10), np.arange(-10, 25, 2.5), np.arange(160, 340, 20)]
                u_lvls = [np.arange(-70, 100, 10), np.arange(-22.5, 17.5, 2.5), np.arange(-70, 100, 10)] #prev min u_lvls_response = -20
                report_plot2(exp, T_lvls, 'Temperature', ' (K)', labels, basis+extension+'_T')
                report_plot2(exp, u_lvls, 'Zonal Wind', r' (m s$^{-1}$)', labels, basis+extension+'_u')

        else:
            vertical = input("Plot vs. a) log pressure or b) pseudo-altitude?")
            for i in range(len(exp)):
                print(datetime.now(), " - opening files ({0:.0f}/{1:.0f})".format(i+1, len(exp)))
                uz = xr.open_dataset(indir+exp[i]+'_utz.nc', decode_times=False).ucomp[0]
                Tz = xr.open_dataset(indir+exp[i]+'_Ttz.nc', decode_times=False).temp[0]
                lat = uz.coords['lat'].data
                p = uz.coords['pfull'].data
                #MSF = calc_streamfn(xr.open_dataset(indir+exp[0]+'_vtz.nc', decode_times=False).vcomp[0], p, lat)  # Meridional Stream Function
                #MSF_xr = xr.DataArray(MSF, coords=[p,lat], dims=['pfull','lat'])  # Make into an xarray DataArray
                #MSF_xr.attrs['units']=r'kgs$^{-1}$'

                #if i == 0:
                #    print("skipping control")
                #elif i != 0:
                #Read in data to plot polar heat contours
                file = '/disco/share/rm811/isca_data/' + exp[i] + '/run0025/atmos_daily_interp.nc'
                ds = xr.open_dataset(file)
                perturb = 0 #ds.local_heating.sel(lon=180, method='nearest').mean(dim='time')  
                if plot_type =='a':
                    plot_combo(uz, Tz, lvls, perturb, lat, p, exp[i], vertical)
                elif plot_type == 'b':
                        u = [uz, xr.open_dataset(indir+exp[0]+'_utz.nc', decode_times=False).ucomp[0]]
                        T = [Tz, xr.open_dataset(indir+exp[0]+'_Ttz.nc', decode_times=False).temp[0]]
                        plot_diff([T, u], ['K', r'ms$^{-1}$'], [np.arange(160, 330, 10), np.arange(-200, 210, 10)],\
                             perturb, lat, p, exp[i], vertical)
   
    elif plot_type == 'c':
        basis = 'PK_e0v4z13'
        heat_type = input('Plot a) zonally symmetric pole-centred or b) off-pole heat?')
        if heat_type == 'a':
            polar_heat = '_w15a4p800f800g50'
            midlat_heat = '_q6m2y45l800u200'
            exp = [basis+midlat_heat, basis+polar_heat, basis+polar_heat+midlat_heat]
            label = 'polar'
        elif heat_type == 'b':
            polar_heat = '_a4x75y90w5v30p600'
            midlat_heat = '_q6m2y45'
            exp = [basis+midlat_heat+'l800u200', basis+polar_heat+'_s', basis+polar_heat+midlat_heat+'_s']
            label = 'offpole90'
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
