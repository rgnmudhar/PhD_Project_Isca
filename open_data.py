"""
    This script plots (zonal) averages of various output variables averaged over X years'-worth of data from Isca
"""

import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

"""
#set-up data
#files = sorted(glob.glob('../isca_data/Polvani_Kushner_2.0/run*/atmos_monthly.nc')) #non-plevel_interpolated version of dataset
files = sorted(glob.glob('../isca_data/Polvani_Kushner_2.0_6y/run*/atmos_monthly_interp_new_height_temp.nc')) #plevel_interpolated version of dataset
ds = xr.open_mfdataset(files, decode_times = False)
print(ds)
"""

#version of data set-up using subset of the run i.e. excluding "spin-up"
files = sorted(glob.glob('../isca_data/Polvani_Kushner_4.0_6y/run*/atmos_monthly_interp_new_height_temp.nc')) 
max_months = len(files)-1
years = 5 # user sets no. of years worth of data to use
months = years*12
files = files[(max_months-months):max_months]
ds = xr.open_mfdataset(files, decode_times = False)

#set-up variables from the dataset
u = ds.ucomp
v = ds.vcomp
P_surf = ds.ps
T = ds.temp
Teq = ds.teq
Vor = ds.vor
#heat = ds.local_heating
tm = ds.coords['time'].data
lat = ds.coords['lat'].data
lon = ds.coords['lon'].data
p = ds.coords['pfull'].data
upper_p = ds.coords['pfull'].sel(pfull=1, method='nearest') # in order to cap plots at pressure = 1hPa


#function to calculate potential temperature
Kappa = 2./7. #taken from constants script

def T_potential(p, P_surf, T, lat):
    
    theta = np.empty_like(T)
    
    for i in range(len(p)):
        for j in range(len(lat)):
            theta[i,j] = T[i,j] * ((P_surf[j]/100)/p[i])**Kappa #potential temperature calculation with P_surf converted to hPa
    
    return theta
    
#functions to get stream function
def calc_streamfn(v, p, lat):
    #Calculates the meridional streamfunction from v wind
    radius = 6371000.
    g = 9.807
    coeff = (2*np.pi*radius)/g

    psi = np.empty_like(v)
    
    # Do the integration
    for ilat in range(lat.shape[0]):
        psi[0,ilat] = coeff*np.cos(np.deg2rad(lat[ilat])) *  v[0,ilat] * p[0]
        for ilev in range(p.shape[0])[1:]:
            psi[ilev,ilat] = psi[ilev-1,ilat] + coeff*np.cos(np.deg2rad(lat[ilat])) \
                             * v[ilev,ilat] * (p[ilev]-p[ilev-1])
    return psi

def vwind(v, p, lat):
    #Take annual mean of meridional wind speed by taking a mean along time and longitude dimensions 
    #Use this to calculate the streamfunction using the dedicated function    
    v_anm = v.mean(dim='time').mean(dim='lon')
    psi = calc_streamfn(v_anm, p, lat)
    return psi

def altitude(p):
    """Finds altitude from pressure using z = -H*log10(p/p0) """
    
    H = 8 #scale height km
    p0 = 1000 #surface pressure in hPa
    
    z = np.empty_like(p)
    
    for i in range(p.shape[0]):
        z[i] = -H*np.log((p[i])/p0)
        
    # Make into an xarray DataArray
    z_xr = xr.DataArray(z, coords=[z], dims=['pfull'])
    z_xr.attrs['units'] = 'km'
    
    #below is the inverse of the calculation
    #p[i] = p0*np.exp((-1)*z[i]*(10**3)/((R*T/g)))
    
    return z_xr


#Plots of means from across the time period
#Average Surface Pressure 
P_surf_mean = P_surf.mean(dim='time')
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
cs1 = P_surf_mean.plot.contourf(levels=25, cmap='RdBu_r')
ax1.contour(cs1, colors='gainsboro', linewidths=0.5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Average Surface Pressure (Pa)')

#Zonal Average Zonal Wind Speed and Temperature
uz = u.mean(dim='time').mean(dim='lon')
Tz = T.mean(dim='time').mean(dim='lon')
fig2, ax = plt.subplots()
cs2a = Tz.plot.contourf(levels=25, cmap='RdBu_r')
ax.contourf(cs2a, colors='none')
cs2b = ax.contour(lat, p, uz, colors='k', levels=15, linewidths=1.25)
#ax.clabel(cs2b, inline=1, fontsize='x-small')
plt.xlabel('Latitude')
plt.ylabel('Pressure (hPa)')
plt.ylim(max(p), upper_p) #goes to 1hPa
plt.yscale("log")
plt.title('Zonal Average Zonal Wind Speed and Temperature')

"""
#Zonal Average Teq and Temperature
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
"""

#Zonal Average Teq
Teqz = Teq.mean(dim='time').mean(dim='lon')
fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
cs4 = Teqz.plot.contourf(levels=25, cmap='RdBu_r')
ax4.contour(cs4, colors='gainsboro', linewidths=0.5)
plt.xlabel('Latitude')
plt.ylabel('Pressure (hPa)')
plt.ylim(max(p),upper_p) #goes to 1hPa
plt.yscale("log")
plt.title('Average Zonal T_eq (K)')

"""
#Zonal Average T from imposed local heating - sanity check
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

#Average Surface T from imposed local heating - sanity check
heat_map = heat.mean(dim='time').sel(pfull=850, method='nearest')
fig6 = plt.figure()
ax6 = fig6.add_subplot(111)
cs6 = heat_map.plot.contourf(levels=25, cmap='RdBu_r')
ax6.contour(cs6, colors='gainsboro', linewidths=0.5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Average Local Heating at 850hPa Level')

#Zonal Average Teq at certain latitudes
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

#Zonal Average Temperature
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

#Zonal Average Potential Temperature
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

#Zonal Average Zonal Wind Speed
uz = u.mean(dim='time').mean(dim='lon')
fig10 = plt.figure()
ax10 = fig10.add_subplot(111)
cs10 = uz.plot.contourf(levels=25, cmap='RdBu_r')
ax10.contour(cs10, colors='gainsboro', linewidths=0.5)
plt.xlabel('Latitude')
plt.ylabel('Pressure (hPa)')
plt.ylim(max(p), upper_p) #goes to 1hPa
plt.yscale("log")
plt.title('Average Zonal Wind')
"""

#Zonal Average Meridional Stream Function
MSF = vwind(v, p, lat)
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

plt.show()
