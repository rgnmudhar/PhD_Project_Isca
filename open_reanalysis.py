from distutils import file_util
import netCDF4 as nc
import cftime
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from aostools import climate

def calc_ep(t, u, v):
    ep1, ep2, div1, div2 = climate.ComputeEPfluxDivXr(u, v, t, 'longitude', 'latitude', 'level', 'time')
    # take time mean of relevant quantities
    div = div1 + div2
    div = div.mean(dim='time')
    ep1 = ep1.mean(dim='time')
    ep2 = ep2.mean(dim='time')
    uz = u.mean(dim='time').mean(dim='longitude')
    return uz, div, ep1, ep2

folder = '/disco/share/rg419/ERA_5/daily_means_1979_to_2020/'
dst = xr.open_mfdataset(folder + 'temperature*.nc', decode_times=False)
dsu = xr.open_mfdataset(folder + 'u_component_of_wind*.nc', decode_times=False)
dsv = xr.open_mfdataset(folder + 'v_component_of_wind*.nc', decode_times=False)
p = dst.coords['level'].data

uz, div, ep1, ep2 = calc_ep(dst.t, dsu.u, dsv.v)

divlvls = np.arange(-20,21,1)
ulvls = np.arange(-200, 200, 10)
fig, ax = plt.subplots(figsize=(8,6))
uz.plot.contour(colors='k', linewidths=0.5, alpha=0.4, levels=ulvls)
div_scaled = div #(div.transpose()/p).transpose()
cs = div_scaled.plot.contourf(levels=divlvls, cmap='RdBu_r', add_colorbar=False)
cb = plt.colorbar(cs)
cb.set_label(label=r'Pressure-scaled Divergence (m s$^{-1}$ day$^{-1}$)', size='large')
#ticklabs = cb.ax.get_yticklabels()
#cb.ax.set_yticklabels(ticklabs, fontsize='large')
climate.PlotEPfluxArrows(dst.latitude,dst.level,ep1,ep2,fig,ax,yscale='log')
plt.xlabel('Latitude', fontsize='x-large')
plt.xlim(-90,90)
plt.xticks([-90, -45, 0, 45, 90], ['90S', '45S', '0', '45N', '90N'])
plt.ylabel('Pressure (hPa)', fontsize='x-large')
plt.yscale('log')
plt.ylim(max(p), min(p)) #to 1 hPa
plt.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
plt.title('Time and Zonal Mean EP Flux', fontsize='x-large')
plt.show()

"""
file = '/disca/share/pm366/ERA-5/era5_var131_masked_zm.nc'
ds = nc.Dataset(file)

for dim in ds.dimensions.values():
    print(dim)

for var in ds.variables.values():
    print(var)

t = ds.variables['time']
lev = ds.variables['lev'][:].data
p = lev/100 # convert to hPa
lat = ds.variables['lat'][:].data
u = ds.variables['ucomp']
H = 8 # km
z = -H*np.log(p/max(p))

#times = []
#times = np.array(times.append(cftime.num2date(t, t.units, t.calendar)))
#print(times)

times2 = []
times2.append(cftime.num2pydate(t, t.units, t.calendar))
times2 = np.array(times2)
#print(times2)
#print(times2[:,503])

plt.plot(times2[0,:], u[:,np.where(p == 900)[0],np.where(lat == 50)[0]][:,0,0]) # plot uwind at 900hPa and 50N over time
plt.title("u (m/s) over time @ 900 hPa and 50N")
plt.show()

plt.plot(lat, np.mean(u[491:494,np.where(p == 900)[0],:], axis=0)[0]) # plot NH winter uwind against latitude
plt.title("mean u (m/s) over Dec-2019 to Feb-2020 @ 900 hPa")
plt.show()

#print(t[:])
#print(p[:])
#print(lat[:])
#print(u[:])

# pull together ucomp data for months Dec, Jan and Feb only
u_DJF = []
for i in range(len(times2[0,:])):
    if times2[0,i].month == 1:
        u_DJF.append(u[i,:,:])
    elif times2[0,i].month == 2:
        u_DJF.append(u[i,:,:])
    elif times2[0,i].month == 12:
        u_DJF.append(u[i,:,:])
u_DJF = np.array(u_DJF)

lvls = np.arange(-10, 47.5, 2.5)
fig, ax = plt.subplots(figsize=(10,8))
plt.contourf(lat, z, np.mean(u_DJF, axis=0), levels=lvls, cmap='RdBu_r')
plt.colorbar()
plt.xlabel('Latitude', fontsize='x-large')
plt.xlim(-90,90)
plt.xticks([-90, -45, 0, 45, 90], ['90S', '45S', '0', '45N', '90N'])
plt.ylabel('Pseudo-Altitude (km)', fontsize='x-large')
plt.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
plt.title('ERA5 DJF Zonal Mean Zonal Wind', fontsize='xx-large')
plt.show()
"""
