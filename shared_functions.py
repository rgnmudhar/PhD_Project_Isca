"""
A selection of functions used in the analysis of output Isca data.
"""

import os
from glob import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

def discard_spinup1(exp_name, time, file_suffix, years):
    """
    Ignore initial spin-up period of X years.
    Output dataset.
    """
    files = sorted(glob('../isca_data/'+exp_name+'/run*'+'/atmos_'+time+file_suffix+'.nc'))
    max_months = len(files)-1
    min_months = years*12
    files = files[min_months:max_months]
    ds = xr.open_mfdataset(files, decode_times=False)

    return ds

def discard_spinup2(exp_name, time, file_suffix, years):
    """
    Ignore initial spin-up period of X years.
    Output list of files.
    """
    files = sorted(glob('../isca_data/'+exp_name+'/run*'+'/atmos_'+time+file_suffix+'.nc'))
    max_months = len(files)-1
    min_months = years*12
    files = files[min_months:max_months]

    return files

def add_phalf(exp_name, time, file_suffix, years):
    """
    Assign phalf levels from uninterpolated to interpolated datset.
    """    
    files = discard_spinup2(exp_name, time, file_suffix, years)
    files_original = discard_spinup2(exp_name, time, '', years)

    ds = xr.open_mfdataset(files, decode_times=False)
    ds_original = xr.open_mfdataset(files_original, decode_times=False)
    ds = ds.assign_coords({"phalf":ds_original.phalf})

    return ds

def Tz(temp):
    """
    Take mean of average zonal temperature by taking averages along time and longitude dimensions.
    """
    Tz = temp.mean(dim='time').mean(dim='lon').data 
    
    return Tz

def T_potential(p, P_surf, T, lat):
    """
    Function to calculate potential temperature from temperature variable.
    """

    Kappa = 2./7. #taken from constants script
    theta = np.empty_like(T)
    
    for i in range(len(p)):
        for j in range(len(lat)):
            theta[i,j] = T[i,j] * ((P_surf[j]/100)/p[i])**Kappa #potential temperature calculation with P_surf converted to hPa
    
    return theta

def P_surf(ds, lat, lon):
    """
    Take mean of surface temperature by taking a mean along time dimension.
    """
    p = ds.ps.mean(dim='time').data 
    
    return p

def uz(ds):
    """
    Take mean of zonal wind speed by taking a mean along time and longitude dimensions.
    """
    uz = ds.ucomp.mean(dim='time').mean(dim='lon').data
   
    return uz

def calc_streamfn(v, p, lat):
    """
    Calculates the meridional mass streamfunction from v wind in kg/s.
    """
    radius = 6371000
    g = 9.807
    coeff = (2*np.pi*radius)/g

    psi = np.empty_like(v)
    # Do the integration
    for ilat in range(lat.shape[0]):
        psi[0,ilat] = coeff*np.cos(np.deg2rad(lat[ilat])) *  v[0,ilat] * p[0]
        for ilev in range(p.shape[0])[1:]:
            psi[ilev,ilat] = psi[ilev-1,ilat] + coeff*np.cos(np.deg2rad(lat[ilat])) \
                             * v[ilev,ilat] * (p[ilev]-p[ilev-1])
    # Make into an xarray DataArray
    
    return psi

def v(ds, p, lat):
    """
    Take annual mean of meridional wind speed by taking a mean along time and longitude dimensions.
    Use this to calculate the streamfunction using the dedicated function.
    """
    v_anm = ds.vcomp.mean(dim='time').mean(dim='lon').data
    psi = calc_streamfn(v_anm, p, lat)
    
    return psi

def altitude(p):
    """
    Finds altitude from pressure using z = -H*log10(p/p0).
    """
    H = 8 #scale height km
    p0 = 1000 #surface pressure hPa    
      
    z = np.empty_like(p)
    
    for i in range(p.shape[0]):
        z[i] = -H*np.log((p[i])/p0)
        
    # Make into an xarray DataArray
    z_xr = xr.DataArray(z, coords=[z], dims=['pfull'])
    z_xr.attrs['units'] = 'km'
    
    #below is the inverse of the calculation
    #p[i] = p0*np.exp((-1)*z[i]*(10**3)/((R*T/g)))
    
    return z_xr

def use_altitude(x, coord1, coord2, dim1, dim2, unit):
    """
    Creates new DataArray that uses z in place of pfull.
    """
    x_xr = xr.DataArray(x, coords=[coord1, coord2], dims=[dim1, dim2])
    x_xr.attrs['units'] = unit
    return x_xr
