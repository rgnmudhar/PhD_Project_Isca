"""
    This script attempts to find EOFs from daily data
"""

import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from eofs.xarray import Eof

files = sorted(glob.glob('Polvani_Kushner_4.0_eps0_1y/run*/atmos_daily.nc'))
ds = xr.open_mfdataset(files, decode_times = False)
print(ds)

u = ds.ucomp # zonal wind
uz = u.mean(dim='lon') # zonal mean zonal wind
uz_anom = uz - uz.mean(dim='time') # zonal wind anomalies vs. time-mean zonal wind

coslat = np.cos(np.deg2rad(uz_anom.coords['lat'].values)).clip(0., 1.) # need to weight due to different box sizes over grid
wgts = np.sqrt(coslat)#[...,np.newaxis]

solver = Eof(uz_anom, weights=wgts)