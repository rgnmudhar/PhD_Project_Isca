"""
    Computes and plots EP flux vectors and divergence terms.
    Based on Martin Jucker's code at https://github.com/mjucker/aostools/blob/d857987222f45a131963a9d101da0e96474dca63/climate.py
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from shared_functions import *
from aostools import climate
from datetime import datetime

def calc_ep(ds):
    t = ds.temp
    u = ds.ucomp
    v = ds.vcomp
    ep1, ep2, div1, div2 = climate.ComputeEPfluxDivXr(u, v, t, 'lon', 'lat', 'pfull', 'time')
    # take time mean of relevant quantities
    div = div1 + div2
    div = div.mean(dim='time')
    ep1 = ep1.mean(dim='time')
    ep2 = ep2.mean(dim='time')
    uz = u.mean(dim='time').mean(dim='lon')
    return uz, div, ep1, ep2

def plot_single(ds, exp_name):
    print(datetime.now(), " - calculating variables")
    p = ds.coords['pfull'].data
    lat = ds.coords['lat'].data
    uz, div, ep1, ep2 = calc_ep(ds)

    #Filled contour plot of time-mean EP flux divergence plus EP flux arrows and zonal wind contours
    divlvls = np.arange(-12,13,1)
    ulvls = np.arange(-200, 200, 10)
    fig, ax = plt.subplots(figsize=(6,6), constrained_layout=True)
    print(datetime.now(), " - plot uz")
    uz.plot.contour(colors='k', linewidths=0.5, alpha=0.4, levels=ulvls)
    print(datetime.now(), " - plot EP flux divergence")
    cs = div.plot.contourf(levels=divlvls, cmap='RdBu_r', add_colorbar=False)
    cb = plt.colorbar(cs)
    cb.set_label(label=r'Divergence (m s$^{-1}$ day$^{-1}$)', size='large')
    cb.ax.set_yticks(divlvls)
    plt.draw()
    ticklabs = cb.ax.get_yticklabels()
    cb.ax.set_yticklabels(ticklabs, fontsize='large')
    plt.show()
    print(datetime.now(), " - plot EP flux arrows")
    ax = climate.PlotEPfluxArrows(ds.lat,ds.pfull,ep1,ep2,fig,ax,yscale='log')
    plt.xlabel(r'Latitude ($\degree$N)', fontsize='xx-large')
    plt.xlim(0,90)
    plt.xticks([10, 30, 50, 70, 90], ['10', '30', '50', '70', '90'])
    plt.ylabel('Pressure (hPa)', fontsize='xx-large')
    plt.yscale('log')
    plt.ylim(max(p), 1) #to 1 hPa
    plt.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
    #plt.title('Time and Zonal Mean EP Flux', fontsize='x-large')
    plt.savefig(exp_name+'_EPflux.pdf', bbox_inches = 'tight')
    return plt.close()

def plot_diff(ds1, ds2, exp):
    p = ds1.coords['pfull'].data
    lat = ds1.coords['lat'].data
    uz, div1, ep1, ep2 = calc_ep(ds1)
    uz, div2, ep1, ep2 = calc_ep(ds2)
    div_diff = div2 - div1

    #Filled contour plot of the difference in time-mean EP flux divergence
    divlvls = np.arange(-12,13,1)
    difflvls = np.arange(-2,2.1,0.1)
    fig, ax = plt.subplots(figsize=(8,6))
    cs1 = ax.contourf(lat, p, div_diff, levels=difflvls, cmap='RdBu_r')
    ax.contourf(cs1, colors='none')
    cs2 = ax.contour(lat, p, div1, colors='k', levels=divlvls, linewidths=1)
    cb = plt.colorbar(cs1)
    cb.set_label(label=r'Difference in Divergence (m s$^{-1}$ day$^{-1}$)', size='x-large')
    cb.ax.set_yticks(difflvls)
    plt.draw()
    ticklabs = cb.ax.get_yticklabels()
    cb.ax.set_yticklabels(ticklabs, fontsize='large')
    plt.xlabel(r'Latitude ($\degree$N)', fontsize='x-large')
    plt.xlim(0,90)
    plt.xticks([0, 45, 90], ['0', '45', '90'])
    plt.ylabel('Pressure (hPa)', fontsize='x-large')
    plt.yscale('log')
    plt.ylim(max(p), 1) #to 1 hPa
    plt.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
    #plt.title('Difference in Time and Zonal Mean EP Flux Divergence', fontsize='x-large')
    plt.savefig(exp+'_EPfluxdiff.pdf', bbox_inches = 'tight')
    return plt.close()

if __name__ == '__main__': 
    #Set-up data to be read in
    basis = 'PK_e0v4z13'
    exp_name = basis+'_w25a4p800f800g50_q6m2y45l800u200'
    exp = [exp_name, basis]
    time = 'daily'
    file_suffix = '_interp'
    years = 2 # user sets no. of years worth of data to ignore due to spin-up
    
    plot_type = input("Plot a) single experiment or b) difference between 2 experiments?")
    if plot_type =='a':
        print(datetime.now(), " - opening files")
        ds = discard_spinup1(exp[0], time, file_suffix, years)
        plot_single(ds, exp_name)
    elif plot_type == 'b':
        ds1 = discard_spinup1(exp[0], time, file_suffix, years)
        ds2 = discard_spinup1(exp[1], time, file_suffix, years)
        plot_diff(ds1, ds2, exp_name)
