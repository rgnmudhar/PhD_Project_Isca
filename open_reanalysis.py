import netCDF4 as nc
import cftime
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

file = '/disca/share/pm366/ERA-5/era5_var131_masked_zm.nc'
ds = nc.Dataset(file)

print(ds)

print(ds.variables.keys())

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

"""
print(t[:])
print(p[:])
print(lat[:])
print(u[:])
"""

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

plt.contourf(lat, z, np.mean(u_DJF, axis=0), cmap='RdBu_r', levels=25)
plt.xlabel('Latidude')
plt.ylabel('Approx. Altitude (km)')
plt.colorbar()
plt.title('ERA5 DJF Zonal Mean Zonal Wind (m/s)')
plt.show()