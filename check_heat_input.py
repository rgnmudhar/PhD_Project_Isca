"""
    This script checks the column integrated heat input of prescribed heating perturbation and finds the proportion above the tropopause.
"""

import xarray as xr
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from shared_functions import *
from datetime import datetime

def open_heat_static(folder, filename, selection):
    file = folder + filename + '.nc'
    ds = xr.open_dataset(file)
    lat = ds.lat
    p = ds.pfull
    if selection == 'lon180':
        lon = ds.lon
        heat = ds.sel(lon=180,method='nearest').variables[filename]
    elif selection == 'slice':
        # select a slice of the wave-2 heating perturbation centred on 180 degreesE, ±45 degrees
        lon = ds.lon.sel(lon=slice(180-45, 180+45))
        heat = ds.sel(lon=slice(180-45, 180+45)).variables[filename]   
    return  p, lat, lon, heat

def integral(heat, ticks, ax):
    return integrate.trapezoid(y = heat, x = ticks, axis = ax)

def integrate_lat(input, coslat):
    int_final = 0
    for j in range(len(input)): #in latitude
        int_final += (input[j]*coslat[j])
    int_final = int_final/np.sum(coslat)
    return int_final

if __name__ == '__main__': 
    #Set-up folders for data to be read in
    indir = '/disco/share/rm811/processed/'
    basis = 'PK_e0v4z13'
    heat_folder = '/home/links/rm811/Isca/input/asymmetry/' #'../Inputs/'
    
    plot = input('Plot for report? y/n ')
    if plot == 'y':
        exp1 = 'q6m2y45l800u200'
        exp2 = 'q6m2y45l800u300_s'
        p_heat, lat_heat, lon_heat, heat1 = open_heat_static(heat_folder, exp1, 'lon180')
        heat2 = open_heat_static(heat_folder, exp2, 'lon180')[-1]
        p, lat, trop = tropopause(indir, basis)

        h = 6/86400
        inc1 = 1e-5
        h_range1 = np.arange(inc1/2, h+inc1, inc1)

        fig, ax = plt.subplots(figsize=(7.5,7))
        cs = plt.contourf(lat_heat, p_heat, heat1, cmap='Purples', levels=h_range1, extend='both')
        plt.contour(lat_heat, p_heat, heat1, colors='gainsboro', levels=h_range1, linewidths=0.5)
        plt.contour(lat_heat, p_heat, heat2, cmap='Purples', levels=h_range1, linewidths=3)
        plt.plot(lat, trop, linewidth=1.25, color='k', linestyle='--')
        cb = plt.colorbar(cs, location='right', extend='both')
        cb.set_label(label=r'Heating (K s$^{-1}$)', size='xx-large')
        cb.ax.tick_params(labelsize='xx-large')
        plt.xlim(0, max(lat))
        plt.xticks([0, 20, 40, 60, 80], ['0', '20', '40', '60', '80'])
        plt.xlabel(r'Latitude ($\degree$N)', fontsize='xx-large')
        plt.ylim(max(p), 100)
        plt.yscale('log')
        ax.set_yticks([1000, 800, 600, 400, 300, 200, 100])
        ax.set_yticklabels(['1000', '800', '600', '400', '300', '200', '100'])
        plt.ylabel('Pressure (hPa)', fontsize='xx-large')
        plt.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
        plt.savefig('heating_vs_tropopause.pdf', bbox_inches = 'tight')
        plt.show()
    else:
        exp_heat = 'q6m2y45l800u200'
        print('Finding heat input for: ', exp_heat)
        p_heat, lat_heat, lon_heat, heat = open_heat_static(heat_folder, exp_heat, 'slice')
        coslat = np.cos(np.deg2rad(lat_heat)).data
        
        #Now integrate...
        int_lon = integral(heat, lon_heat, 2) #in longitude
        int_p = integral(int_lon, p_heat, 0) #in pressure
        int_full = integrate_lat(int_p, coslat)
        print(exp_heat, ' total heat input: ', int_full)

        #And for that which is only above the tropopause
        trop = tropopause(indir, basis)[-1]
        int_p_tropo = []
        for m in range(len(lat_heat)):
            sub = int_lon[:,m]
            trop_lvl = trop[m]
            sub_above_trop = np.where(p_heat<=trop_lvl, sub, 0)
            int_p_tropo.append(integral(sub_above_trop, p_heat, 0))
        int_tropo = integrate_lat(int_p_tropo, coslat)
        percent_above_tropo = (int_tropo / int_full) * 100
        print(exp_heat, ' percent above {} tropopause: '.format(basis), percent_above_tropo, ' %')