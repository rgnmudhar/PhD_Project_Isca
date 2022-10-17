from glob import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from open_winds import *
from shared_functions import *
from eofs.xarray import Eof
import statsmodels.api as sm

def leading_pcs(solver):
    """
    Retrieve the leading PC time series.
    By default PCs used are scaled to unit variance (divided by square root of the eigenvalue).
    """
    pcs = solver.pcs(npcs=2, pcscaling=1)

    return pcs

def variance(solver):
    """
    Fractional EOF mode variances.
    The fraction of the total variance explained by each EOF mode, values between 0 and 1 inclusive.
    """

    return solver.varianceFraction()

def findtau(ac):
    """
    Finds the time for correlation reduce to 1/e.
    """
    for i in range(len(ac)):
        if ac[i] - 1/np.e < 0:
            tau = i
            break

    return tau

def AM_times(pcs, lags):
    """
    Finds autocorrelation function in order to determine decorrelation time (tau)
    """
    ac1 = sm.tsa.acf(pcs.sel(mode=0).values, nlags=lags)
    ac2 = sm.tsa.acf(pcs.sel(mode=1).values, nlags=lags)

    tau1 = findtau(ac1)
    tau2 = findtau(ac2)

    return tau1, tau2

def EOF_finder():
    folder = '/disco/share/rg419/ERA_5/daily_means_1979_to_2020/'
    var = 'u_component_of_wind'
    files = sorted(glob(folder+var+'*.nc'))
    ds = xr.open_mfdataset(files, decode_times=True)
    ds = ds.assign_coords({'month': ds.time.dt.month})
    ds_NDJF = ds.where((ds.month <= 2)|(ds.month == 11), drop=True)
    uz = ds_NDJF.u.drop_vars('month').mean('longitude')
    #EOFs
    # For EOFs follow Sheshadri & Plumb 2017, use p>100hPa, lat>20degN
    p_min = 100  # hPa
    lat_min = 20  # degrees
    u_new = uz.sel(level=slice(p_min,1000)).sel(latitude=slice(90, lat_min))
    # Calculate anomalies
    u_anom = u_new - u_new.mean(dim='time')
    # sqrt(cos(lat)) weights due to different box sizes over grid
    sqrtcoslat = np.sqrt(np.cos(np.deg2rad(u_anom.coords['latitude'].values))) 
    # sqrt(dp) weights, select correct number of levels
    nplevs = u_anom.coords['level'].shape[0]
    sqrtdp = np.sqrt(np.diff(ds.coords['level'].values[-nplevs-2:]))
    # Calculate gridpoint weights
    wgts = np.outer(sqrtdp,sqrtcoslat)
    # Create an EOF solver to do the EOF analysis.
    solver = Eof(u_anom.compute(), weights=wgts)
    pcs = leading_pcs(solver)
    variance_fractions = variance(solver)
    lags = 50 
    tau1, tau2 = AM_times(pcs, lags)
    return tau1, tau2, variance_fractions[0]*100, variance_fractions[1]*100

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
#t1, t2, v1, v2 = EOF_finder()