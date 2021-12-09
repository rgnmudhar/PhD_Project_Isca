"""
    This script plots temperature and zonal wind as a function of time.
    Plots the equatorial averages for temperature and wind by taking the mean of the two latitudes either side of 0.
"""

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import glob

def data(ds, time):
    t1 = (ds.temp.mean(dim='lon')[:,24,31]+ds.temp.mean(dim='lon')[:,24,32])/2
    t2 = (ds.temp.mean(dim='lon')[:,12,31]+ds.temp.mean(dim='lon')[:,12,32])/2
    t3 = (ds.temp.mean(dim='lon')[:,6,31]+ds.temp.mean(dim='lon')[:,6,32])/2

    u1 = (ds.ucomp.mean(dim='lon')[:, 24, 31] + ds.ucomp.mean(dim='lon')[:, 24, 32])/2
    u2 = (ds.ucomp.mean(dim='lon')[:, 12, 31] + ds.ucomp.mean(dim='lon')[:, 12, 32])/2
    u3 = (ds.ucomp.mean(dim='lon')[:, 6, 31] + ds.ucomp.mean(dim='lon')[:, 6, 32])/2
        
    plot_x(time, t1, t2, t3, 't')
    plot_x(time, u1, u2, u3, 'u')
    return plt.show()

def plot_x(time, x1, x2, x3, x_label):
    """Plots either temperature or zonal wind over time"""
    
    if x_label == 't':
        x = 'Equatorial Mean Temperature (K)'
    elif x_label == 'u':
        x = 'Equatorial Mean Zonal Wind (m/s)'
    
    fig1 = plt.figure(figsize=(7.5,6.5))
    ax1 = fig1.add_subplot(111)
    
    ax1.plot(time, x1, color='#2980B9', linewidth=3)
    ax1.plot(time, x2, color='#27AE60', linewidth=3)
    ax1.plot(time, x3, color='#C0392B', linewidth=3)
    ax1.set_xlabel('Time (Years)', fontsize='xx-large')
    ax1.set_xlim(0, max(time))
    ax1.set_ylabel(x, fontsize='xx-large')
    
    leg = ax1.legend(labels=('Surface', '~220 hPa', '~80 hPa'), loc='upper center' , bbox_to_anchor=(0.5, -0.07),
                     fancybox=False, shadow=True, ncol=5, fontsize='xx-large')
    
    ax1.tick_params(axis='both', labelsize = 'xx-large', direction='in', which='both')

    return plt.tight_layout()

if __name__ == '__main__':
    files = sorted(glob.glob('run*/*.nc'))
    ds = xr.open_mfdataset(files, decode_times=False)

    tm = ds.coords['time'].data
    Tyears = np.zeros_like(tm)
    for i in range(len(Tyears)):
        Tyears[i] = Tyears[i-1] + 1/12

    data(ds, Tyears)