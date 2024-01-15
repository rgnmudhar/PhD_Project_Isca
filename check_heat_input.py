"""
    This script checks the column integrated heat input of prescribed heating perturbation and finds the proportion above the tropopause.
"""

from hmac import digest_size
import xarray as xr
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from shared_functions import *
from datetime import datetime

def open_heat_static(filename):
    # select a slice of the wave-2 heating perturbation centred on 180 degreesE, ±45 degrees
    file = '../Inputs/' + filename + '.nc'
    ds = xr.open_dataset(file)
    lat = ds.lat
    lon = ds.lon.sel(lon=slice(180-45, 180+45))
    p = ds.pfull
    heat = ds.sel(lon=slice(180-45, 180+45)).variables[filename]
    return  p, lat, lon, heat

def integral(heat, ticks, ax):
    return integrate.trapezoid(y = heat, x = ticks, axis = ax)

def integrate_lat(input):
    int_final = 0
    for j in range(len(input)): #in latitude
        int_final += (input[j]*coslat[j])
    int_final = int_final/np.sum(coslat)
    return int_final

if __name__ == '__main__': 
    #Set-up data to be read in
    indir = '/disco/share/rm811/processed/'
    basis = 'PK_e0v4z13'
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
    
    """
    p_heat, lat_heat, lon_heat, heat = open_heat_static('q6m2y45l800u400')
    coslat = np.cos(np.deg2rad(lat_heat)).data

    for e in exp:

        trop = tropopause(indir, e)[-1]

        #Now integrate...
        int_lon = integral(heat, lon_heat, 2) #in longitude
        int_p = integral(int_lon, p_heat, 0) #in pressure
        int_full = integrate_lat(int_p)
        
        #And for that which is only above the tropopause
        int_p_tropo = []
        for m in range(len(lat_heat)):
            sub = int_lon[:,m]
            trop_lvl = trop[m]
            sub_above_trop = np.where(p_heat<=trop_lvl, sub, 0)
            int_p_tropo.append(integral(sub_above_trop, p_heat, 0))
        int_tropo = integrate_lat(int_p_tropo)
        percent_above_tropo = (int_tropo / int_full) * 100

        print(e, ': ', percent_above_tropo, ' %')

    """
    
    exp = [exp[0], 'PK_e0v4z13']
    labels = ['control', 'no asymmetry']
    c = ['k', 'r']
    
    fig, ax = plt.subplots(figsize=(6,6))
    for i in range(len(exp)):
        print(datetime.now(), " - finding tropopause ({0:.0f}/{1:.0f})".format(i+1, len(exp)))
        p, lats, trop = tropopause(indir, exp[i])
        ax.plot(lats, trop, linewidth=1.25, color=c[i], label=labels[i])
    ax.set_ylabel('Pressure (hPa)', fontsize='xx-large')
    ax.set_xlabel(r'Latitude ($\degree$N)', fontsize='xx-large')
    ax.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
    plt.legend(loc='upper center' , bbox_to_anchor=(0.5, -0.15), fancybox=False, ncol=2, fontsize='x-large', facecolor='white')

    h = 6/86400
    inc2 = 1e-5
    h_range2 = np.arange(inc2/2, h+inc2, inc2)
    filenames = ['q6m2y45l800u200', 'q6m2y45l800u300', 'q6m2y45l800u400']
    colors = ['#bbd6eb', '#88bedc', '#549ecd',  '#2a7aba', '#0c56a0', '#08306b']
    for i in range(len(filenames)):
        file = '../Inputs/' + filenames[i] + '.nc'
        ds = xr.open_dataset(file)
        heat = ds.sel(lon=180, method='nearest').variables[filenames[i]]
        ax.contour(ds.lat, ds.pfull, heat, colors=colors[i], levels=h_range2, linewidths=1.5, alpha=0.25)

    #plt.xlim(0,90)
    plt.xticks([-80, -60, -40, -20, 0, 20, 40, 60, 80], ['-80', '-60', '-40', '-20', '0', '20', '40', '60', '80'])
    plt.ylim(max(p), 100)
    plt.yscale('log')
    plt.yticks([900, 800, 700, 600, 500, 400, 300, 200, 100], ['900', '800', '700', '600', '500', '400', '300', '200', '100'])
    plt.savefig('tropopause_vs_heatperturb.pdf', bbox_inches = 'tight')
    plt.show()
    plt.close()