from glob import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from open_winds import *
from shared_functions import *

"""
def EOF_finder():
    from EOFs import *
    folder = '/disco/share/rg419/ERA_5/daily_means_1979_to_2020/'
    var = 'u_component_of_wind'
    files = sorted(glob(folder+var+'*.nc'))
    ds = xr.open_mfdataset(files, decode_times=True)
    ds = ds.assign_coords({'month': ds.time.dt.month})
    ds_DJF = ds.where((ds.month <= 2)|(ds.month == 12), drop=True)
    uz = ds_DJF.u.mean('lon')
    #EOFs
    # For EOFs follow Sheshadri & Plumb 2017, use p>100hPa, lat>20degN
    p_min = 100  # hPa
    lat_min = 20  # degrees
    u_new = uz.sel(level=slice(p_min,1000)).sel(lat=slice(lat_min,90))
    # Calculate anomalies
    u_anom = u_new - u_new.mean(dim='time')
    # sqrt(cos(lat)) weights due to different box sizes over grid
    sqrtcoslat = np.sqrt(np.cos(np.deg2rad(u_anom.coords['lat'].values))) 
    # sqrt(dp) weights, select correct number of levels
    nplevs = u_anom.coords['pfull'].shape[0]
    sqrtdp = np.sqrt(np.diff(ds.coords['phalf'].values[-nplevs-2:-1]))
    # Calculate gridpoint weights
    wgts = np.outer(sqrtdp,sqrtcoslat)
    # Create an EOF solver to do the EOF analysis.
    solver = Eof(u_anom.compute(), weights=wgts)
    pcs = leading_pcs(solver)
    variance_fractions = variance(solver)
    lags = 50 
    tau1, tau2 = AM_times(pcs, lags)
    return print(tau1+' days', variance_fractions[0]+' %', tau2+' days', variance_fractions[1]+' %')
"""

def SPV_finder():
    folder = "../Files/"
    file = 'obs_u1060'
    months = ['NDJF', 'NDJFM', 'DJF', 'DJFM']
    for i in range(len(months)):
        print(months[i])
        SPV = open_file(folder, file, months[i])
        SPV_mean = np.mean(SPV)
        SPV_sd = np.std(SPV)
        SSWs, err = find_SSW(SPV)
        print('{0:.2f} ± {1:.2f} m/s'.format(SPV_mean, SPV_sd))
        print('{0:.2f} ± {1:.2f} / 100 days'.format(SSWs, err))

SPV_finder()