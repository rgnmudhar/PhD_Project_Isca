# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
from matplotlib import ticker
import xarray as xr
import numpy as np
import glob
import imageio
import os

def calc_streamfn(v, p, lat):
    '''Calculates the meridional streamfunction from v wind.
    Parameters
    ----------
        vz_xr: an xarray DataArray of form [pressure levs, latitudes]
    Returns
    -------
        psi_xr: xarray DataArray of meridional mass streamfunction, units of kg/s
    '''
    radius = 6371000
    g = 9.807
    coeff = (2*np.pi*radius)/g

    psi = np.empty_like(v)
    # Do the integration
    for ilat in range(lat.shape[0]):
        psi[0,ilat] = coeff*np.cos(np.deg2rad(lat[ilat])) *  v[0,ilat] * p[0]
        for ilev in range(p.shape[0])[1:]:
            psi[ilev,ilat] = psi[ilev-1,ilat] + coeff*np.cos(np.deg2rad(lat[ilat])) \
                             * v[ilev,ilat] * (p[ilev]-p[ilev-1])
    # Make into an xarray DataArray
    
    return psi

def v(ds, p, lat):
    ''' Take annual mean of meridional wind speed by taking a mean along time and longitude dimensions 
        Use this to calculate the streamfunction the dedicated function
    '''
    v_anm = ds.vcomp.mean(dim='time').mean(dim='lon').data
    psi = calc_streamfn(v_anm, p, lat)
    
    return psi       

def plots(ds, i):
    ''' Plots '''    
    lon = ds.coords['lon'].data
    lat = ds.coords['lat'].data
    p = ds.coords['pfull'].data
    upper_p = ds.coords['pfull'].sel(pfull=1, method='nearest') # in order to cap plots at pressure = 1hPa
    u = ds.ucomp
    T = ds.temp
    
    
    #Zonal Average Zonal Wind Speed
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    #uz = u.mean(dim='time').mean(dim='lon')
    #lvls = np.linspace(-60, 80, 25)
    #cs1 = uz.plot.contourf(levels=lvls, cmap='RdBu_r')
    Tz = T.mean(dim='time').mean(dim='lon')
    lvls = np.linspace(220, 320, 25)
    cs1 = Tz.plot.contourf(levels=lvls, cmap='RdBu_r')
    ax1.contour(cs1, colors='gainsboro', linewidths=0.5)
    plt.xlabel('Latitude')
    plt.ylabel('Pressure (hPa)')
    plt.ylim(max(p), upper_p) #goes to 1hPa
    plt.yscale("log")
    plt.title('Month %i'%(i+1))
    #plt.savefig("uwind_%i.png"%(i), bbox_inches = 'tight')
    plt.savefig("temp_%i.png"%(i), bbox_inches = 'tight')
    plt.close()
    
    """
    #Surface Temperature
    Ts = T.mean(dim='time')[-1]
    lvls = np.linspace(220, 320, 25)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    cs1 = Ts.plot.contourf(levels=lvls, cmap='RdBu_r')
    ax1.contour(cs1, colors='gainsboro', linewidths=0.5)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Month %i'%(i+1))
    plt.savefig("temp_%i.png"%(i), bbox_inches = 'tight')
    plt.close()
    
    #Zonal Avergae Meridional Stream Function
    msf = v(ds, p, lat)
    msfz_xr = xr.DataArray(msf, coords=[p,lat], dims=['pfull','lat'])
    msfz_xr.attrs['units'] = 'kg/s'
    lvls = np.linspace(-1e9, 1e9, 15)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    cs1 = msfz_xr.plot.contourf(cmap = 'RdBu_r', levels=lvls) #contour plot
    plt.ylabel('Pressure (hPa)')
    plt.ylim(max(p),min(p))
    plt.xlabel('Latitude')
    plt.title('Month %i'%(i+1))
    plt.savefig("msf_%i.png"%(i), bbox_inches = 'tight')
    plt.close()
    """

    return plt.show()


if __name__ == '__main__': 
    files = sorted(glob.glob('../isca_data/PK_eps0_vtx3_zoz13_7y/run*/atmos_daily.nc'))
        
    for i in np.arange(0, len(files)):
        file = files[i]
        ds = xr.open_dataset(file, decode_times=False)
        plots(ds, i)
    
    #Merge all plots into a GIF for visualisation
    #images = glob.glob('uwind*.png')
    images = glob.glob('temp*.png')
    list.sort(images, key = lambda x: int(x.split('_')[1].split('.png')[0]))
    IMAGES = []
    for i in range(0,len(images)):
        IMAGES.append(imageio.imread(images[i]))
    #imageio.mimsave("uwind.gif", IMAGES, 'GIF', duration = 1/3)
    imageio.mimsave("temp.gif", IMAGES, 'GIF', duration = 1/3)
        
    #Delete all temporary plots from working directory
    for i in range(0,len(images)):
        os.remove(images[i])

