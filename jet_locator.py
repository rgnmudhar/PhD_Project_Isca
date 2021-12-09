"""
    Function for finding location and strength of maximum given zonal wind u(lat) field - based on WSeviour code
    Amended to only look at NH tropospheric jet.
"""

import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

def calc_jet_lat_quad(u, lats, p, plot=False):
    # Restrict to 3 points around maximum
    u_new = u.mean(dim='time').mean(dim='lon').sel(pfull=p, method='nearest')
    u_max = np.where(u_new == np.ma.max(u_new))[0][0]
    u_near = u_new[u_max-1:u_max+2]
    lats_near = lats[u_max-1:u_max+2]
    # Quadratic fit, with smaller lat spacing
    coefs = np.ma.polyfit(lats_near,u_near,2)
    fine_lats = np.linspace(lats_near[0], lats_near[-1],200)
    quad = coefs[2]+coefs[1]*fine_lats+coefs[0]*fine_lats**2
    # Find jet lat and max
    jet_lat = fine_lats[np.where(quad == max(quad))[0][0]]
    jet_max = coefs[2]+coefs[1]*jet_lat+coefs[0]*jet_lat**2
    # Plot fit?
    if plot:
        print(jet_max)
        print(jet_lat)
        plt.plot(lats_near, u_near)
        plt.plot(fine_lats, quad)
        plt.xlabel("Latitude")
        plt.ylabel("Wind Speed")
        plt.title("Jet Latitude at p={:.0f}hPa".format(p))
        plt.text(jet_lat-2, jet_max-1, "Jet max = {0:.2f}m/s at {1:.2f} deg latitude".format(jet_max, jet_lat))
        plt.show()
 
    return jet_lat, jet_max

p = 850 # pressure level at which we want to find the jet (hPa)

"""
#For a single dataset
ds = xr.open_dataset(''Polvani_Kushner_novtx_6y/run0001/atmos_monthly_interp_new_height_temp.nc', decode_times=False) 
lat = ds.coords['lat'].data
u = ds.ucomp
calc_jet_lat_quad(u, lat, p, plot=True)
"""

#For a time series
files=sorted(glob.glob('Polvani_Kushner_4.0_6y/run*/atmos_monthly_interp_new_height_temp.nc'))
iter = np.arange(0,len(files))
jet_maxima = []
jet_lats = []
for i in iter:
    print(i)
    file  = files[i]
    ds = xr.open_dataset(file, decode_times=False)
    lat = ds.coords['lat'].data
    u = ds.ucomp
    # restrict to NH:
    u = u[:,:,int(len(lat)/2):len(lat),:]
    lat = lat[int(len(lat)/2):len(lat)] 
    # find and store jet maxima and latitudes for each month
    jet_lat, jet_max = calc_jet_lat_quad(u, lat, p)
    jet_maxima.append(jet_max)
    jet_lats.append(jet_lat)

fig, ax = plt.subplots(figsize=(11,6))
ax.plot(iter+1, jet_maxima, color='k')
ax.set_xlim(1,len(files))
ax.set_xlabel("Run no.")       
ax.set_ylabel('Jet Max (m/s)', color='k')
ax2 = ax.twinx()
ax2.plot(iter+1, jet_lats, color='r')
ax2.set_ylabel('Jet Latitude', color='r')
plt.title("NH Tropospheric (p={0:.0f}hPa) Jet Max and Latitude for ~{1:.0f}y worth".format(p, len(files)/12))
plt.show()

