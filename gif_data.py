"""
    This script plots (zonal) averages of various output variables into gifs over time.
"""

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from glob import glob
import imageio
import os
from shared_functions import *  

def plots(ds, i), var:
    ''' Plots '''    
    lon = ds.coords['lon'].data
    lat = ds.coords['lat'].data
    p = ds.coords['pfull'].data
    upper_p = ds.coords['pfull'].sel(pfull=1, method='nearest') # in order to cap plots at pressure = 1hPa
    

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    if var=='a':
        u = ds.ucomp
        uz = u.mean(dim='time').mean(dim='lon')
        lvls = np.linspace(-50, 100, 25)
        cs1 = uz.plot.contourf(levels=lvls, cmap='RdBu_r')
    elif var=='b':
        T = ds.temp
        Tz = T.mean(dim='time').mean(dim='lon')
        lvls = np.linspace(160, 320, 25)
        cs1 = Tz.plot.contourf(levels=lvls, cmap='RdBu_r')
    elif var=='c':
        heat = ds.local_heating
        heatz = heat.mean(dim='time').mean(dim='lon')
        lvls = np.linspace(0, 4.5, 25)
        cs1 = heatz.plot.contourf(levels=lvls, cmap='RdBu_r')
    
    ax1.contour(cs1, colors='gainsboro', linewidths=0.5)
    plt.xlabel('Latitude')
    plt.ylabel('Pressure (hPa)')
    plt.ylim(max(p), upper_p) #goes to 1hPa
    plt.yscale("log")
    plt.title('Month %i'%(i+1))

    if var=='a':
        plt.savefig("uwind_%i.png"%(i), bbox_inches = 'tight')
    elif var=='b':
        plt.savefig("temp_%i.png"%(i), bbox_inches = 'tight')
    elif var=='c':
        plt.savefig("heat_%i.png"%(i), bbox_inches = 'tight')   
    
    return plt.close()


if __name__ == '__main__': 
    files = sorted(glob('../isca_data/PK_eps0_vtx3_zoz13_7y/run*/atmos_daily.nc'))
    var = input('a) uz, b) Tz or c) local heating?')

    for i in np.arange(0, len(files)):
        file = files[i]
        ds = xr.open_dataset(file, decode_times=False)
        plots(ds, i, var)
    
    # Merge all plots into a GIF for visualisation
    if var=='a':
        images = glob('uwind*.png')
    elif var=='b':
        images = glob('temp*.png')
    elif var=='c':
        images = glob('heat*.png')

    list.sort(images, key = lambda x: int(x.split('_')[1].split('.png')[0]))
    IMAGES = []
    for i in range(0,len(images)):
        IMAGES.append(imageio.imread(images[i]))

    if var='a':
        imageio.mimsave("uwind.gif", IMAGES, 'GIF', duration = 1/3)
    elif var=='b':
        imageio.mimsave("temp.gif", IMAGES, 'GIF', duration = 1/3)
    elif var=='c':
        imageio.mimsave("local_heat.gif", IMAGES, 'GIF', duration = 1/3)
        
    # Delete all temporary plots from working directory
    for i in range(0,len(images)):
        os.remove(images[i])

