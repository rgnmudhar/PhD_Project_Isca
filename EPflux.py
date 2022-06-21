"""
    Computes and plots EP flux vectors and divergence terms.
    Based on Martin Jucker's code at https://github.com/mjucker/aostools/blob/d857987222f45a131963a9d101da0e96474dca63/climate.py
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from aostools import climate
from datetime import datetime

def calc_ep(u, v, t):
    ep1, ep2, div1, div2 = climate.ComputeEPfluxDivXr(u, v, t, 'lon', 'lat', 'pfull', 'time')
    # take time mean of relevant quantities
    div = div1 + div2
    div = div.mean(dim='time')
    ep1 = ep1.mean(dim='time')
    ep2 = ep2.mean(dim='time')
    return div, ep1, ep2

def plot_single(uz, u, v, t, exp_name, heat):
    print(datetime.now(), " - calculating variables")
    p = uz.coords['pfull']
    lat = uz.coords['lat']
    div, ep1, ep2 = calc_ep(u, v, t)

    #Filled contour plot of time-mean EP flux divergence plus EP flux arrows and zonal wind contours
    divlvls = np.arange(-12,13,1)
    ulvls = np.arange(-200, 200, 10)
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
    plt.xlabel(r'Latitude ($\degree$N)', fontsize='xx-large')
    plt.xlim(0,90)
    plt.xticks([10, 30, 50, 70, 90], ['10', '30', '50', '70', '90'])
    plt.ylabel('Pressure (hPa)', fontsize='xx-large')
    plt.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
    print(datetime.now(), " - plot EP flux arrows")
    ax = climate.PlotEPfluxArrows(lat, p, ep1, ep2, fig, ax, yscale='log')
    plt.yscale('log')
    plt.ylim(max(p), 1) #to 1 hPa
    #plt.title('Time and Zonal Mean EP Flux', fontsize='x-large')
    plt.savefig(exp_name+'_EPflux.pdf', bbox_inches = 'tight')
    return plt.close()

def plot_diff(uz, u, v, t, exp_name, heat):
    p = uz.coords['pfull']
    lat = uz.coords['lat']
    div1, ep1a, ep2a = calc_ep(u[0], v[0], t[0])
    div2, ep1b, ep2b = calc_ep(u[1], v[1], t[1])
    div_diff = div1 - div2
    ep1_diff = ep1a - ep1b
    ep2_diff = ep2a - ep2b

    #Filled contour plot of the difference in time-mean EP flux divergence
    difflvls = np.arange(-2,2.2,0.2)
    ulvls = np.arange(-200, 200, 10)
    fig, ax = plt.subplots(figsize=(6,6), constrained_layout=True)
    print(datetime.now(), " - plot uz")
    uz.plot.contour(colors='k', linewidths=0.5, alpha=0.4, levels=ulvls)
    print(datetime.now(), " - plot polar heat")
    plt.contour(lat, p, heat, colors='g', linewidths=0.25, alpha=0.4, levels=11)
    print(datetime.now(), " - plot EP flux divergence")
    cs = div_diff.plot.contourf(levels=difflvls, cmap='RdBu_r', add_colorbar=False)
    cb = plt.colorbar(cs)
    cb.set_label(label=r'Difference in Divergence (m s$^{-1}$ day$^{-1}$)', size='large')
    cb.ax.set_yticks(difflvls)
    fig.canvas.draw()
    ticklabs = cb.ax.get_yticklabels()
    cb.ax.set_yticklabels(ticklabs, fontsize='large')
    plt.xlabel(r'Latitude ($\degree$N)', fontsize='xx-large')
    plt.xlim(0,90)
    plt.xticks([10, 30, 50, 70, 90], ['10', '30', '50', '70', '90'])
    plt.ylabel('Pressure (hPa)', fontsize='xx-large')
    print(datetime.now(), " - plot EP flux arrows")
    ax = climate.PlotEPfluxArrows(lat, p, ep1_diff, ep2_diff, fig, ax, yscale='log')
    plt.yscale('log')
    plt.ylim(max(p), 1) #to 1 hPa
    plt.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
    plt.savefig(exp_name+'_EPfluxdiff.pdf', bbox_inches = 'tight')
    return plt.close()

if __name__ == '__main__': 
    #Set-up data to be read in
    indir = '/disco/share/rm811/processed/'
    basis = 'PK_e0v4z13'
    filename = 'w15a4p300f800g50_q6m2y45l800u200'
    exp = [basis+'_'+filename, basis+'_q6m2y45l800u200']

    #Read in data to plot polar heat contours
    file = '/disco/share/rm811/isca_data/' + basis + '_' + filename + '/run0100/atmos_daily_interp.nc'
    ds = xr.open_dataset(file)
    heat = ds.local_heating.sel(lon=180, method='nearest').mean(dim='time')
   
    plot_type = input("Plot a) single experiment or b) difference between 2 experiments?")

    print(datetime.now(), " - opening files")
    u = xr.open_dataset(indir+exp[0]+'_u.nc', decode_times=False).ucomp
    v = xr.open_dataset(indir+exp[0]+'_v.nc', decode_times=False).vcomp
    t = xr.open_dataset(indir+exp[0]+'_t.nc', decode_times=False).temp
    uz = xr.open_dataset(indir+exp[0]+'_tzmean.nc', decode_times=False).ucomp[0]
    
    if plot_type =='a':
        plot_single(uz, u, v, t, exp[0], heat)
    elif plot_type == 'b':
        u = [u, xr.open_dataset(indir+exp[1]+'_u.nc', decode_times=False).ucomp+2]
        v = [v, xr.open_dataset(indir+exp[1]+'_v.nc', decode_times=False).vcomp+2]
        t = [t, xr.open_dataset(indir+exp[1]+'_t.nc', decode_times=False).temp+2]
        plot_diff(uz, u, v, t, exp[0], heat)
