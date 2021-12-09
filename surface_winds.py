"""
    This script plots time and zonally averaged zonal surface wind for multiple gamma (averaged over a year's-worth of data from Isca)
"""

import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import cftime

"""
#set-up data
ds0 = xr.open_mfdataset(sorted(glob.glob('Polvani_Kushner_novtx_6y/run*/atmos_monthly_interp_new_height_temp.nc')), decode_times = False)
ds1 = xr.open_mfdataset(sorted(glob.glob('Polvani_Kushner_1.0_6y/run*/atmos_monthly_interp_new_height_temp.nc')), decode_times = False)
ds2 = xr.open_mfdataset(sorted(glob.glob('Polvani_Kushner_2.0_6y/run*/atmos_monthly_interp_new_height_temp.nc')), decode_times = False)
ds3 = xr.open_mfdataset(sorted(glob.glob('Polvani_Kushner_3.0_6y/run*/atmos_monthly_interp_new_height_temp.nc')), decode_times = False)
ds4 = xr.open_mfdataset(sorted(glob.glob('Polvani_Kushner_4.0_6y/run*/atmos_monthly_interp_new_height_temp.nc')), decode_times = False)
"""

#version of data set-up using subset of the run i.e. excluding "spin-up"
years = 5 # user sets no. of years worth of data to use
months = years*12

files0 = sorted(glob.glob('Polvani_Kushner_novtx_6y/run*/atmos_monthly_interp_new_height_temp.nc')) 
max_months = len(files0)-1
files0 = files0[(max_months-months):max_months]
ds0 = xr.open_mfdataset(files0, decode_times = False)
ds1 = xr.open_mfdataset(sorted(glob.glob('Polvani_Kushner_1.0_6y/run*/atmos_monthly_interp_new_height_temp.nc'))[(max_months-months):max_months], decode_times = False)
ds2 = xr.open_mfdataset(sorted(glob.glob('Polvani_Kushner_2.0_6y/run*/atmos_monthly_interp_new_height_temp.nc'))[(max_months-months):max_months], decode_times = False)
ds3 = xr.open_mfdataset(sorted(glob.glob('Polvani_Kushner_3.0_6y/run*/atmos_monthly_interp_new_height_temp.nc'))[(max_months-months):max_months], decode_times = False)
ds4 = xr.open_mfdataset(sorted(glob.glob('Polvani_Kushner_4.0_6y/run*/atmos_monthly_interp_new_height_temp.nc'))[(max_months-months):max_months], decode_times = False)
ds_hs = xr.open_mfdataset(sorted(glob.glob('held_suarez_default/run*/atmos_monthly.nc'))[13:72], decode_times = False)

#following opens ERA5 re-analysis data
file_ra = '/disca/share/pm366/ERA-5/era5_var131_masked_zm.nc'
ds_ra = nc.Dataset(file_ra)
t_ra = ds_ra.variables['time']
lev = ds_ra.variables['lev'][:].data
p_ra = lev/100 # convert to hPa
lat_ra = ds_ra.variables['lat'][:].data
u_ra = ds_ra.variables['ucomp']
#times = []
#times.append(cftime.num2pydate(t_ra, t_ra.units, t_ra.calendar)) # convert to actual dates
#times = np.array(times)

#set-up variables from the datasets
u0 = ds0.ucomp
u1 = ds1.ucomp
u2 = ds2.ucomp
u3 = ds3.ucomp
u4 = ds4.ucomp
u_hs = ds_hs.ucomp

lat = ds1.coords['lat'].data
lon = ds1.coords['lon'].data
p = ds1.coords['pfull'].data
upper_p = ds1.coords['pfull'].sel(pfull=1, method='nearest') # in order to cap plots at pressure = 1hPa

#Time and Zonal Average Zonal Wind Speed at the pressure level closest to XhPa
X = 100
uz0 = u0.mean(dim='time').mean(dim='lon').sel(pfull=X, method='nearest')
uz1 = u1.mean(dim='time').mean(dim='lon').sel(pfull=X, method='nearest')
uz2 = u2.mean(dim='time').mean(dim='lon').sel(pfull=X, method='nearest')
uz3 = u3.mean(dim='time').mean(dim='lon').sel(pfull=X, method='nearest')
uz4 = u4.mean(dim='time').mean(dim='lon').sel(pfull=X, method='nearest')
uz_hs = u_hs.mean(dim='time').mean(dim='lon').sel(pfull=X, method='nearest')

fig = plt.subplots(1,1)
plt.plot(lat, uz0, 'k', label='no vortex', linewidth='3.0')
plt.plot(lat, uz1, ':k', label=r'$\gamma$ = 1.0')
plt.plot(lat, uz2, '-.k', label=r'$\gamma$  = 2.0')
plt.plot(lat, uz3, '--k', label=r'$\gamma$  = 3.0')
plt.plot(lat, uz4, 'k', label=r'$\gamma$  = 4.0')
plt.plot(lat, uz_hs, 'b', label='Held-Suarez')
plt.plot(lat_ra, np.mean(u_ra[491:494,np.where(p_ra == X)[0],:], axis=0)[0], 'r', label='ERA5 (DJF)') # plot re-analysis uwind against latitude average over Nov-19 to Feb-20 (NH winter)
plt.xlabel('Latitude')
plt.xlim(-90,90)
plt.ylabel('Wind Speed (m/s)')
plt.legend()
plt.title('Time and Zonally Averaged Zonal Wind at ~{:.0f}hPa'.format(X))

plt.show()