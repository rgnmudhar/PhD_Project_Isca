"""
    This script plots (time and zonal) averages of various output variables averaged over X years'-worth of data from Isca
"""

from glob import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from shared_functions import *

def plots(ds, exp_name):
    # Set-up variables from the dataset
    tm = ds.coords['time'].data
    lat = ds.coords['lat'].data
    lon = ds.coords['lon'].data
    p = ds.coords['pfull'].data
    upper_p = ds.coords['pfull'].sel(pfull=1, method='nearest') # in order to cap plots at pressure = 1hPa
    z = altitude(p)
    H = 8 #scale height km
    p0 = 1000 #surface pressure hPa    
    upper_z = -H*np.log(upper_p/p0)

    u = uz(ds)
    T = Tz(ds)
    Teq = Teqz(ds)
    #Vor = ds.vor
    #heat = ds.local_heating

    # Use altitude rather than pressure for vertical
    u = use_altitude(u, z, lat, 'pfull', 'lat', r'ms$^{-1}$')
    T = use_altitude(T, z, lat, 'pfull', 'lat', 'K')
    Teq = use_altitude(Teq, z, lat, 'pfull', 'lat', 'K')

    # Plots of means from across the time period
    """
    #Average Surface Pressure 
    P_surf = ds.ps.mean(dim='time')
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    cs1 = P_surf_mean.plot.contourf(levels=25, cmap='RdBu_r')
    ax1.contour(cs1, colors='gainsboro', linewidths=0.5)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Average Surface Pressure (Pa)')
    """

    # Zonal Average Zonal Wind Speed and Temperature
    lvls2a = np.arange(160, 330, 5)
    lvls2b = np.arange(-200, 200, 5)
    fig2, ax = plt.subplots(figsize=(10,8))
    cs2a = T.plot.contourf(levels=lvls2a, cmap='RdBu_r', add_colorbar=False)
    ax.contourf(cs2a, colors='none')
    cs2b = ax.contour(lat, z, u, colors='k', levels=lvls2b, linewidths=1)
    cs2b.collections[int(len(lvls2b)/2)].set_linewidth(1.25)
    #plt.clabel(cs2b, levels = lvls2b[::4], inline=1, fontsize='x-small')
    plt.colorbar(cs2a, label='Temperature (K)')
    plt.xlabel('Latitude', fontsize='large')
    plt.xlim(-90,90)
    plt.xticks([-90, -45, 0, 45, 90], ['90S', '45S', '0', '45N', '90N'])
    plt.ylabel('Pseudo-Altitude (km)', fontsize='large')
    plt.ylim(min(z), upper_z) #goes to ~1hPa
    plt.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
    plt.title('Mean Zonal Wind and Temperature', fontsize='x-large')
    plt.savefig(exp_name+'_zonal.png', bbox_inches = 'tight')
    plt.close()

    """
    # Zonal Average Teq and Temperature
    fig3, ax = plt.subplots()
    cs3a = Tz.plot.contourf(levels=25, cmap='RdBu_r')
    ax.contourf(cs3a, colors='none')
    cs3b = ax.contour(lat, p, Teqz, cmap = 'bwr', levels=17, linewidths=1.25)
    plt.colorbar(cs3b)
    #ax.clabel(cs3b, inline=1, fontsize='x-small')
    plt.xlabel('Latitude')
    plt.ylabel('Pressure (hPa)')
    plt.ylim(max(p), upper_p) #goes to 1hPa
    plt.yscale("log")
    plt.title('Zonal Average Teq and Temperature')

    # Zonal Average Teq
    lvls4 = np.arange(100, 320, 10)
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    cs4 = Teq.plot.contourf(levels=lvls4, cmap='RdBu_r')
    ax4.contour(cs4, colors='gainsboro', linewidths=0.5)
    plt.xlabel('Latitude')
    plt.xticks([-90, -45, 0, 45, 90], ['90S', '45S', '0', '45N', '90N'])
    plt.ylabel('Pseudo-Altitude (km)')
    plt.ylim(min(z), upper_z) #goes to ~1hPa
    plt.title('Zonal Equilibrium Temperature (K)')

    # Zonal Average T from imposed local heating - sanity check
    heatz = heat.mean(dim='time').mean(dim='lon')
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111)
    cs5 = heatz.plot.contourf(levels=25, cmap='RdBu_r')
    ax5.contour(cs5, colors='gainsboro', linewidths=0.5)
    plt.xlabel('Latitude')
    plt.ylabel('Pressure (hPa)')
    plt.ylim(max(p),upper_p) #goes to 1hPa
    plt.yscale("log")
    plt.title('Average Zonal Local Heating (K)')

    # Average Surface T from imposed local heating - sanity check
    heat_map = heat.mean(dim='time').sel(pfull=850, method='nearest')
    fig6 = plt.figure()
    ax6 = fig6.add_subplot(111)
    cs6 = heat_map.plot.contourf(levels=25, cmap='RdBu_r')
    ax6.contour(cs6, colors='gainsboro', linewidths=0.5)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Average Local Heating at 850hPa Level')

    # Zonal Average Teq at certain latitudes
    fig7 = plt.figure()
    ax7 = fig7.add_subplot(111)
    cs7a = plt.plot(Teqz[:,53],p, 'k', label='60N') #plots Teq at ~60N, index based on locator script
    cs7b = plt.plot(Teqz[:,9],p, '--k', label='60S') #plots Teq at ~60S, index based on locator script
    cs7c = plt.plot(Teqz[:,42],p, 'm', label='30N') #plots Teq at ~30N, index based on locator script
    cs7d = plt.plot(Teqz[:,20],p, '--m', label='30S') #plots Teq at ~30S, index based on locator script
    plt.xlabel('Temperature (K)')
    plt.ylabel('Pressure (hPa)')
    plt.ylim(max(p), upper_p) #goes to 1hPa
    plt.yscale("log")
    plt.legend()
    plt.title('T_eq Profile (K)')

    # Zonal Average Temperature
    Tz = T.mean(dim='time').mean(dim='lon')
    fig8 = plt.figure()
    ax8 = fig8.add_subplot(111)
    cs8 = Tz.plot.contourf(levels=25, cmap='RdBu_r')
    ax8.contour(cs8, colors='gainsboro', linewidths=0.5)
    plt.xlabel('Latitude')
    plt.ylabel('Pressure (hPa)')
    plt.ylim(max(p), upper_p) #goes to 1hPa
    plt.yscale("log")
    plt.title('Average Zonal Temperature (K)')

    # Zonal Average Potential Temperature
    T_anm = T.mean(dim='time').mean(dim='lon')
    P_surf_anm = P_surf.mean(dim='time').mean(dim='lon')
    theta = T_potential(p,P_surf_anm,T_anm,lat)
    theta_xr = xr.DataArray(theta, coords=[p,lat], dims=['pfull','lat'])  # Make into an xarray DataArray
    theta_xr.attrs['units']='K'
    fig9 = plt.figure()
    ax9 = fig9.add_subplot(111)
    cs9 = theta_xr.plot.contourf(levels=25, cmap='RdBu_r')
    ax9.contour(cs9, colors='gainsboro', linewidths=0.5)
    plt.xlabel('Latitude')
    plt.ylabel('Pressure (hPa)')
    plt.ylim(max(p), upper_p) #goes to 1hPa
    plt.yscale("log")
    plt.title('Average Zonal Potential Temperature (K)')

    # Zonal Average Zonal Wind Speed
    fig10 = plt.figure()
    ax10 = fig10.add_subplot(111)
    cs10 = plt.contourf(lat, p, u, levels=25, cmap='RdBu_r')
    ax10.contour(cs10, colors='gainsboro', linewidths=0.5)
    plt.xlabel('Latitude')
    plt.ylabel('Pressure (hPa)')
    plt.ylim(max(p), upper_p) #goes to 1hPa
    plt.title('Average Zonal Wind')

    # Zonal Average Meridional Stream Function
    MSF = v(ds, p, lat)
    MSF_xr = xr.DataArray(MSF, coords=[p,lat], dims=['pfull','lat'])  # Make into an xarray DataArray
    MSF_xr.attrs['units']='kg/s'
    fig11 = plt.figure()
    ax11 = fig11.add_subplot(111)
    cs11 = MSF_xr.plot.contourf(levels=25, cmap='RdBu_r')
    ax11.contour(cs11, colors='gainsboro', linewidths=0.5)
    plt.xlabel('Latitude')
    plt.ylabel('Pressure (hPa)')
    plt.ylim(max(p), upper_p) #goes to 1hPa
    plt.yscale("log")
    plt.title('Streamfunction (kg/s)')
    """
    return plt.show()

if __name__ == '__main__': 
    #Set-up data to be read in
    exp_name = 'PK_eps0_vtx3_zoz13_7y'
    time = 'daily'
    years = 0 # user sets no. of years worth of data to ignore due to spin-up
    ds = discard_spinup1(exp_name, time, '_interp', years)

    plots(ds, exp_name)
