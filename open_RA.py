from glob import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cm
from datetime import datetime
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

def uz_plot():
    # Plots re-analysis NDJF mean zonal wind
    print(datetime.now(), " - opening files")
    folder = '/disco/share/rg419/ERA_5/daily_means_1979_to_2020/'
    var = 'u_component_of_wind'
    files = sorted(glob(folder+var+'*.nc'))
    ds = xr.open_mfdataset(files, decode_times=True)
    ds = ds.assign_coords({'month': ds.time.dt.month})
    ds_NDJF = ds.where((ds.month <= 2)|(ds.month == 11), drop=True)
    uz1 = ds_NDJF.u.drop_vars('month').mean('longitude').mean('time')
    u_lvls = np.arange(-70, 100, 10)
    p1 = ds.level
    lat1 = ds.latitude

    indir = '/disco/share/rm811/processed/'
    exp = 'PK_e0v4z13_q6m2y45l800u200'
    uz2 = xr.open_dataset(indir+exp+'_utz.nc', decode_times=False).ucomp[0]
    p2 = uz2.pfull
    lat2 = uz2.lat

    print(datetime.now(), " - plotting")
    fig, ax = plt.subplots(figsize=(6,6))
    norm = cm.TwoSlopeNorm(vmin=min(u_lvls), vmax=max(u_lvls), vcenter=0)
    csa = ax.contourf(lat1, p1, uz1, levels=u_lvls, norm=norm, cmap='RdBu_r')
    cb  = fig.colorbar(csa, extend='both')
    cb.set_label(label=r'NDJF zonal wind (m s$^{-2}$)', size='xx-large')
    cb.ax.tick_params(labelsize='x-large')
    csb = ax.contour(lat2, p2, uz2, colors='k', linewidths=1.5, alpha=0.25, levels=u_lvls)
    csb.collections[list(u_lvls).index(0)].set_linewidth(3)
    ax.set_ylabel('Pressure (hPa)', fontsize='xx-large')
    ax.set_ylim(max(p2), min(p1))
    ax.set_yscale('log')
    ax.set_xlabel(r'Latitude ($\degree$N)', fontsize='xx-large')
    ax.set_xlim(0, max(lat1))
    ax.set_xticks([0, 20, 40, 60, 80], ['0', '20', '40', '60', '80'])
    plt.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
    plt.savefig('ERA5_NDJF_uz.pdf', bbox_inches = 'tight')
    plt.show()
    return plt.close()

uz_plot()
#SPV_finder()
#t1, t2, v1, v2 = EOF_finder()