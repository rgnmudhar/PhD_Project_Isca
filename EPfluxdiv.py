"""
    Computes and plots EP flux vectors and divergence terms.
    Based on Martin Jucker's code at https://github.com/mjucker/aostools/blob/d857987222f45a131963a9d101da0e96474dca63/climate.py
"""

import os
from glob import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from shared_functions import *

# Set-up constants for the calculations
a0 = 6376.0e3 # [m]
Omega = 7.292e-5 #[1/s]
p0 = 1e3 # [hPa]
k = 2./7. 

def AxRoll(x,ax,invert=False):
	"""
    From Martin Jucker's code
    Re-arrange array x so that axis 'ax' is first dimension.
	Undo this if invert=True      
	"""
	if ax < 0:
		n = len(x.shape) + ax
	else:
		n = ax
	#
	if invert is False:
		y = np.rollaxis(x,n,0)
	else:
		y = np.rollaxis(x,0,n+1)
	return y

def calc_anom(x,axis=-1):
	"""
    From Martin Jucker's code
    Computes the anomaly of array x along dimension axis.
	INPUTS:
	  x    - array to compute anomalies from
	  axis - axis along dimension for anomalies
	OUTPUTS:
	  x    - anomalous array
	"""    
    #bring axis to the front
	xt= AxRoll(x,axis)
	#compute anomalies
	xt = xt - xt.mean(axis=0)[np.newaxis,:]
	#bring axis back to where it was
	x = AxRoll(xt,axis,invert=True)
	return x

def waves1(x,y=None,wave=-1,axis=-1,do_anomaly=False):
	"""
    From Martin Jucker's code
    Get Fourier mode decomposition of x, or <x*y>, where <.> is zonal mean.
		If y!=[], returns Fourier mode contributions (amplitudes) to co-spectrum zonal mean of x*y. Shape is same as input, except axis which is len(axis)/2+1 due to Fourier symmetry for real signals.
		If y=[] and wave>=0, returns real space contribution of given wave mode. Output has same shape as input.
		If y=[] and wave=-1, returns real space contributions for all waves. Output has additional first dimension corresponding to each wave.
	INPUTS:
		x	   - the array to decompose
		y	   - second array if wanted
		wave	   - which mode to extract. all if <0
		axis	   - along which axis of x (and y) to decompose
		do_anomaly - decompose from anomalies or full data
	OUTPUTS:
		xym	   - data in Fourier space
	"""

	initShape = x.shape
	x = AxRoll(x,axis)

	if y is not None:
		y = AxRoll(y,axis)
    
	# compute anomalies
	if do_anomaly:
		x = calc_anom(x,0)
		if y is not None:
			y = calc_anom(y,0)

	# Fourier decompose
	x = np.fft.fft(x,axis=0)
	nmodes = x.shape[0]//2+1

	if wave < 0:
			if y is not None:
				xym = np.zeros((nmodes,)+x.shape[1:])
			else:
				xym = np.zeros((nmodes,)+initShape)
	else:
		xym = np.zeros(initShape[:-1])

	if y is not None:
			y = np.fft.fft(y,axis=0)
			# Take out the waves
			nl  = x.shape[0]**2
			xyf  = np.real(x*y.conj())/nl
			# due to symmetric spectrum, there's a factor of 2, but not for wave-0
			mask = np.zeros_like(xyf)
			if wave < 0:
				for m in range(xym.shape[0]):
					mask[m,:] = 1
					mask[-m,:]= 1
					xym[m,:] = np.sum(xyf*mask,axis=0)
					mask[:] = 0
				# wavenumber 0 is total of all waves
                                #  this makes more sense than the product of the zonal means
				xym[0,:] = np.nansum(xym[1:,:],axis=0)
				xym = AxRoll(xym,axis,invert=True)
			else:
				xym = xyf[wave,:]
				if wave >= 0:
					xym = xym + xyf[-wave,:]
	else:
			mask = np.zeros_like(x)
			if wave >= 0:
				mask[wave,:] = 1
				mask[-wave,:]= 1 # symmetric spectrum for real signals
				xym = np.real(np.fft.ifft(x*mask,axis=0))
				xym = AxRoll(xym,axis,invert=True)
			else:
				for m in range(xym.shape[0]):
					mask[m,:] = 1
					mask[-m,:]= 1 # symmetric spectrum for real signals
					fourTmp = np.real(np.fft.ifft(x*mask,axis=0))
					xym[m,:] = AxRoll(fourTmp,axis,invert=True)
					mask[:] = 0

	return np.squeeze(xym)

def waves2(x,y=None,wave=-1,dim='lon',anomaly=None):
	"""
    From Martin Jucker's code
    Get Fourier mode decomposition of x, or <x*y>, where <.> is zonal mean.
		If y!=None, returns Fourier mode contributions (amplitudes) to co-spectrum zonal mean of x*y. Dimension along which Fourier is performed is either gone (wave>=0) or has len(axis)/2+1 due to Fourier symmetry for real signals (wave<0).
		If y=None and wave>=0, returns real space contribution of given wave mode. Output has same shape as input.
		If y=None and wave<0, returns real space contributions for all waves. Output has additional first dimension corresponding to each wave.
	INPUTS:
		x	   - the array to decompose. xr.DataArray
		y	   - second array if wanted. xr.DataArray
		wave	   - which mode to extract. all if <0
		dim	   - along which dimension of x (and y) to decompose
		anomaly	   - if not None, name of dimension along which to compute anomaly first.
	OUTPUTS:
		xym	   - data. xr.DataArray
	"""
	if anomaly is not None:
		x = x - x.mean(anomaly)
		if y is not None:
			y = y - y.mean(anomaly)
	sdims = [d for d in x.dims if d != dim]
	xstack = x.stack(stacked=sdims)

	if y is None:
		ystack=None
	else:
		ystack = y.stack(stacked=sdims)

	gw = waves1(xstack,ystack,wave=wave,axis=xstack.get_axis_num(dim))

	if y is None and wave >= 0: # result in real space
		stackcoords = [xstack[d] for d in xstack.dims]
	elif y is None and wave < 0: # additional first dimension of wave number
		stackcoords = [('k',np.arange(gw.shape[0]))] + [xstack[d] for d in xstack.dims]
	elif y is not None and wave >= 0: # original dimension is gone
		stackcoords = [xstack.stacked]
	elif y is not None and wave < 0: # additional dimension of wavenumber
		stackcoords = [('k',np.arange(gw.shape[0])), xstack.stacked]

	gwx = xr.DataArray(gw,coords=stackcoords)

	return gwx.unstack()

def calc_verteddy(v,T,ref,wave):
    """
    Based on Martin Jucker's code
    Computes the vertical eddy components of the residual circulation,
		bar(v'Theta'/Theta_p).
		Output units are [v_bar] = [v], [t_bar] = [v*p]
		INPUTS:
			v    - meridional wind
			T    - temperature
			ref  - how to treat dTheta/dp:
			       - 'rolling-X' : centered rolling mean over X days
			       - 'mean'	     : full time mean
                               - 'instant'   : no time operation
			wave - wave number: if == 0, return total. else passed to GetWavesXr()
		OUPUTS:
			vz - zonal mean meridional wind [v]
			Tz - zonal mean vertical eddy component <v'Theta'/Theta_p> [v*p]
    """
    # pressure quantitites
    pp0 = (p0/T['pfull'])**k
	# convert to potential temperature
    T = T*pp0 # theta

	# zonal means
    vz = v.mean(dim='lon')
    Tz = T.mean(dim='lon') # theta_bar

	# prepare pressure derivative
    dthdp = Tz.differentiate('pfull',edge_order=2) # dthdp = d(theta_bar)/dp
    dthdp = dthdp.where(dthdp != 0)

	# time mean of d(theta_bar)/dp
    if time in dthdp.dims:
        if 'rolling' in ref:
            r = int(ref.split('-')[-1])
            dthdp = dthdp.rolling(dim={'time':r},min_periods=1,center=True).mean()
        elif ref == 'mean':
            dthdp = dthdp.mean(dim='time')
        elif ref == 'instant':
            dthdp = dthdp
    
	# now get wave component
    if isinstance(wave,list):
        vpTp = waves2(v,T,wave=-1).sel(k=wave).sum('k')
    elif wave == 0:
        vpTp = (v - vz)*(T - Tz)
        vpTp = vpTp.mean(dim='lon')  # vpTp = bar(v'Th')
    else:
        vpTp = waves2(v,T,wave=wave) # vpTp = bar(v'Th'_{k=wave})
    
    Tz = vpTp/dthdp # t_bar = bar(v'Th')/(dTh_bar/dp)

    return vz,Tz

def calc_divergence(u,v,T,wave=0):
    """
    Based on Martin Jucker's code
    Compute the EP-flux vectors and divergence terms.
		The vectors are normalized to be plotted in cartesian (linear)
		coordinates, i.e. do not include the geometric factor a*cos\phi.
		Thus, ep1 is in [m2/s2], and ep2 in [hPa*m/s2].
		The divergence is in units of m/s/day, and therefore represents
		the deceleration of the zonal wind. This is actually the quantity
		1/(acos\phi)*div(F).
	INPUTS:
	  u    - zonal wind, shape(time,p,lat,lon) [m/s]
	  v    - meridional wind, shape(time,p,lat,lon) [m/s]
	  T    - temperature, shape(time,p,lat,lon) [K]
	  w    - pressure velocity, optional, shape(time,p,lat,lon) [hPa/s]
	  wave - only include this wave number total if == 0, all waves if <0, single wave if >0, sum over waves if a list. optional
	OUTPUTS:
	  ep1  - meridional EP-flux component, scaled to plot in cartesian [m2/s2]
	  ep2  - vertical   EP-flux component, scaled to plot in cartesian [hPa*m/s2]
	  div1 - horizontal EP-flux divergence, divided by acos\phi [m/s/d]
	  div2 - horizontal EP-flux divergence , divided by acos\phi [m/s/d]
    """
    coslat = np.cos(np.deg2rad(u['lat']))
    sinlat = np.sin(np.deg2rad(u['lat']))
    R = 1./(a0*coslat)
    f = 2*Omega*sinlat
    pp0  = (p0/u['pfull'])**k
    do_ubar = False # do shear and vorticity correction?

    # absolute vorticity
    if do_ubar:
        uz = u.mean(dim='lon')
        fhat = R*np.rad2deg((uz*coslat)).differentiate('lat',edge_order=2)
    else:
        fhat = 0.
    
    fhat = f - fhat # [1/s]

    # compute thickness weighted heat flux [m.hPa/s]
    vbar,vertEddy = calc_verteddy(v,T,'mean',0) # vertEddy = bar(v'Th'/(dTh_bar/dp))

    # find zonal anomalies
    if isinstance(wave,list):
        upvp = waves2(u,v,wave=-1).sel(k=wave).sum('k')
    elif wave == 0:
        u = u - u.mean(dim='lon')
        v = v - v.mean(dim='lon')
        upvp = (u*v).mean(dim='lon')
    else:
        upvp = waves2(u,v,wave=wave)

    # compute the horizontal component
    if do_ubar:
        shear = uz.differentiate('pfull',edge_order=2) # [m/s.hPa]
    else:
        shear = 0.
    ep1_cart = -upvp + shear*vertEddy # [m2/s2 + m/s.hPa*m.hPa/s] = [m2/s2]

    # compute vertical component
    ep2_cart = fhat*vertEddy # [1/s*m.hPa/s] = [m.hPa/s2]

    # We now have to make sure we get the geometric terms right
	# With our definition,
	#  div1 = 1/(a.cosphi)*d/dphi[a*cosphi*ep1_cart*cosphi],
	#    where a*cosphi comes from using cartesian, and cosphi from the derivative
	# With some algebra, we get
	#  div1 = cosphi d/d phi[ep1_cart] - 2 sinphi*ep1_cart
    div1 = coslat*(np.rad2deg(ep1_cart).differentiate('lat',edge_order=2)) - 2*sinlat*ep1_cart
    # Now, we want acceleration, which is div(F)/a.cosphi [m/s2]
    div1 = R * div1 # [m/s2]
    # Similarly, we want acceleration = 1/a.coshpi*a.cosphi*d/dp[ep2_cart] [m/s2]
    div2 = ep2_cart.differentiate('pfull',edge_order=2) # [m/s2]
    # convert to m/s/day
    div1 = div1*86400
    div2 = div2*86400

    # give the DataArrays their names
    ep1_cart.name = 'ep1'
    ep2_cart.name = 'ep2'
    div1.name = 'div1'
    div2.name = 'div2'
    
    return ep1_cart, ep2_cart, div1, div2

def plot(var, lab):
    """
    Filled contour plots of various time-mean EP flux variables
    """
    lim = np.max(abs(var))
    lvls = np.arange(-1*lim, lim+5, 5)
    fig, ax = plt.subplots(figsize=(10,8))
    cs = var.plot.contourf(levels=lvls, cmap='RdBu_r', add_colorbar=False)
    plt.colorbar(cs, label=lab)
    plt.xlabel('Latitude', fontsize='x-large')
    plt.xlim(-90,90)
    plt.xticks([-90, -45, 0, 45, 90], ['90S', '45S', '0', '45N', '90N'])
    plt.ylabel('Pressure (hPa)', fontsize='x-large')
    plt.ylim(max(p), 1) #goes to ~1hPa
    plt.yscale('log')
    plt.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
    return plt.show()

if __name__ == '__main__': 
    #Set-up data to be read in
    basis = 'PK_eps0_vtx1_zoz13'
    experiments = [basis+'_7y']     
    time = 'daily'
    years = 2 # user sets no. of years worth of data to ignore due to spin-up
    file_suffix = '_interp'
    
    files = []
    for i in range(len(experiments)):
        files.append(discard_spinup2(experiments[i], time, file_suffix, years))    
        
    ds = xr.open_mfdataset(files[0], decode_times=False) 
    lat = ds.lat
    lon = ds.lon
    p = ds.pfull
    t = ds.time
    u = ds.ucomp
    v = ds.vcomp
    T = ds.temp

    ep1, ep2, div1, div2 = calc_divergence(u,v,T)
    ep1 = ep1.mean(dim='time')
    ep2 = ep2.mean(dim='time').transpose() # transpose for plotting lat vs. pressure
    div1 = div1.mean(dim='time').transpose() 
    div2 = div2.mean(dim='time').transpose()

    variables = [ep1, ep2, div1, div2]
    labels = [r'Horizontal EP Flux (m$^{2}$s$^{-2}$)',\
        r'Vertical EP Flux (m$\cdot$hPa s$^{-2}$)',\
        r'Horizontal EP Flux Divergence (ms$^{-1}$day$^{-1}$)',\
        r'Vertical EP Flux Divergence (ms$^{-1}$day$^{-1}$)']
    
    for i in range(len(variables)):
        print('Plotting '+labels[i])
        plot(variables[i], labels[i])