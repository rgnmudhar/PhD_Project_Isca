"""
    Computes and plots EP flux vectors and divergence terms, based on Martin Jucker's code at https://github.com/mjucker/aostools/blob/d857987222f45a131963a9d101da0e96474dca63/climate.py
    Computes and plots meridional heat flux 
"""
from glob import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
from aostools import climate
from datetime import datetime
from shared_functions import *

def ComputeRefractiveIndexXr(uz,Tz,k,lat='lat',pres='level',N2const=None,ulim=None):
	'''
		Refractive index as in Simpson et al (2009) doi 10.1175/2008JAS2758.1 and also Matsuno (1970) doi 10.1175/1520-0469(1970)027<0871:VPOSPW>2.0.CO;2
		Stationary waves are assumed, ie c=0.

		Setting k=0 means the only term depending on wave number is left out. This could be more efficient if n2(k) for different values of k is of interest.

		meridonal PV gradient is
		q_\phi = A - B + C, where
				A = 2*Omega*cos\phi
				B = \partial_\phi[\partial_\phi(ucos\phi)/acos\phi]
				C = af^2/Rd*\partial_p(p\theta\partial_pu/(T\partial_p\theta))
		Total refractive index is
		n2 = a^2*[D - E - F], where
				D = q_\phi/(au)
				E = (k/acos\phi)^2
				F = (f/2NH)^2

		Inputs are:
			uz    - zonal mean zonal wind, xarray.DataArray
			Tz    - zonal mean temperature, xarray.DataArray
			k     - zonal wave number. [.]
			lat   - name of latitude [degrees]
			pres  - name of pressure [hPa]
			N2const - if not None, assume N2 = const = N2const [1/s2]
			ulim  - only compute n2 where |u| > ulim to avoid divisions by zero.
		Outputs are:
			n2  - refractive index, dimension pres x lat [.]
	'''
	# some constants
	from .constants import Rd,cp,a0,Omega
	H     = 7.e3 # [m]

	#
	## term D
	dqdy = ComputeMeridionalPVGradXr(uz,Tz,lat,pres,Rd,cp,a0)
	if ulim is not None:
		utmp = uz.where(np.abs(uz)>ulim)
	else:
		utmp = uz
	D = dqdy/(a0*utmp)

	#
	## term E
	coslat = np.cos(np.deg2rad(uz[lat]))
	E = ( k/(a0*coslat) )**2

	#
	## term F
	sinlat = np.sin(np.deg2rad(uz[lat]))
	f = 2*Omega*sinlat
	f2 = f*f
	if N2const is None:
		N2 = ComputeN2Xr(Tz,pres,H,Rd,cp)
	else:
		N2 = N2const
	H2 = H*H
	F = f2/(4*N2*H2)

	return a0*a0*(D-E-F)

def ComputeN2Xr(Tz,pres='infer',H=7.e3,Rd=287.04,cp=1004):
	''' Compute the Brunt-Vaisala frequency from zonal mean temperature
		 N2 = -Rd*p/(H**2.) * (dTdp - Rd*Tz/(p*cp))
		 this is equivalent to
		 N2 = g/\theta d\theta/dz, with p = p0 exp(-z/H)

		INPUTS:
			Tz    - zonal mean temperature [K], xarray.DataArray
			pres  - name of pressure [hPa]
			H     - scale height [m]
			Rd    - specific gas constant for dry air
			cp    - specific heat of air at constant pressure
		OUTPUTS:
			N2  - Brunt-Vaisala frequency, [1/s2], dim pres x lat
	'''
	if pres == 'infer':
		dim_names = FindCoordNames(Tz)
		pres = dim_names['pres']
	dTdp = Tz.differentiate(pres,edge_order=2)*0.01
	p = Tz[pres]*100. # [Pa]
	N2 = -Rd*p/(H**2.) * (dTdp - Rd*Tz/(p*cp))
	return N2

def ComputeN2Xr_CONST(N2):
	''' Find a constant N2 following Weinberger et al. (2021).
		Average over 100-10 hPa, 40-80N
	'''
	N2_const = N2.sel(pfull=slice(10,100)).mean('pfull')
	N2_const = N2_const.sel(lat=slice(40,80)).mean('lat')
	# make a new array
	N2_const_full = np.full_like(N2, N2_const)
	N2_const_full = xr.DataArray(N2_const_full, coords=[N2.pfull, N2.lat], dims=['pfull', 'lat'])
	return N2_const_full

def ComputeMatsunoTermF(Tz,lat,pres):
    '''Compute term F in the Matsuno (1970) form of the refractive index.
		Computed following Simpson et al JAS (2009) DOI 10.1175/2008JAS2758.1.
		This quantity has three terms,
		F = C * (W + X + Y + Z), where
				C = - f^2/N^2
                X = 1/4*H^2
                Y = -1/N * (3p/H^2 * \partial_p N + p^2/H^2 * \partial_p^2 N)
                Z = 2p^2/N^2*H^2 * (\partial_pN)^2

		INPUTS:
			Tz        - zonal mean temperature [K], dim pres x lat OR N x pres x lat
			lat       - name of latitude [degrees]
			pres      - name of pressure [hPa]
		OUTPUTS:
			F
	'''
    from aostools.constants import Rd,cp,a0,Omega
    H = 7.e3 # [m]
    H2 = H * H

    sinlat = np.sin(np.deg2rad(uz[lat]))
    f = 2*Omega*sinlat
    f2 = f*f

    N2 = climate.ComputeN2Xr(Tz,pres,H,Rd,cp)
    N = np.sqrt(N2)
    
    C = -f2/N2

    X = 1 / (4 * H2)
    
    p = uz[pres]
    dNdp = N.differentiate(pres,edge_order=2)
    d2Ndp2 = dNdp.differentiate(pres,edge_order=2)

    Y = (-1/N) * ((3*p/H2)*dNdp + (p**2/H2)*d2Ndp2)

    Z = ((2*p**2) / (H2 * N2)) * (dNdp)**2

    return C * (X + Y + Z)

def ComputeRefractiveIndexXr_FULL(uz,Tz,k,lat='lat',pres='level',ulim=None):
	'''
        Adapted from M. Jucker's aostools which found refractive index as in Simpson et al (2009) doi 10.1175/2008JAS2758.1 and also Matsuno (1970) doi 10.1175/1520-0469(1970)027<0871:VPOSPW>2.0.CO;2
		Here, we follow form 3 (M70n_Nphi,z) from Weinberger et al. (2021) doi 10.1175/JAS-D-20-0267.1, which is a change to term F in the below.

        Stationary waves are assumed, ie c=0.

		Setting k=0 means the only term depending on wave number is left out. This could be more efficient if n2(k) for different values of k is of interest.

		meridonal PV gradient is
		q_\phi = A - B + C, where
				A = 2*Omega*cos\phi
				B = \partial_\phi[\partial_\phi(ucos\phi)/acos\phi]
				C = af^2/Rd*\partial_p(p\theta\partial_pu/(T\partial_p\theta))
		Total refractive index is
		n2 = a^2*[D - E - F], where
				D = q_\phi/(au)
				E = (k/acos\phi)^2
				F = \sqrt(1/\rho0)*f^2/N(\phi,z) * \partial_z^2\sqrt(\rho0/)N(\phi,z)^2)

		Inputs are:
			uz    - zonal mean zonal wind, xarray.DataArray
			Tz    - zonal mean temperature, xarray.DataArray
			k     - zonal wave number. [.]
			lat   - name of latitude [degrees]
			pres  - name of pressure [hPa]
			ulim  - only compute n2 where |u| > ulim to avoid divisions by zero.
		Outputs are:
			n2  - refractive index, dimension pres x lat [.]
	'''
	# some constants
	from aostools.constants import Rd,cp,a0

	## term D is UNCHANGED
	dqdy = climate.ComputeMeridionalPVGradXr(uz,Tz,lat,pres,Rd,cp,a0)
	if ulim is not None:
		utmp = uz.where(np.abs(uz)>ulim)
	else:
		utmp = uz
	D = dqdy/(a0*utmp)

	#
	## term E is UNCHANGED
	coslat = np.cos(np.deg2rad(uz[lat]))
	E = ( k/(a0*coslat) )**2

	#
	## term F has been CHANGED
	F = ComputeMatsunoTermF(Tz,lat,pres)

	return a0*a0*(D-E-F)

if __name__ == '__main__': 
	#Set-up data to be read in
	indir = '/disco/share/rm811/processed/'
	exp = 'PK_e0v4z13_q6m2y45l800u200'
	letters = ['a) ', 'b) ', 'c) ', 'd) ', 'e) ', 'f) ', 'g) '] 
	uz = xr.open_dataset(indir+exp+'_utz.nc', decode_times=False).ucomp[0]
	Tz = xr.open_dataset(indir+exp+'_Ttz.nc', decode_times=False).temp[0]
	p = uz.pfull
	lat = uz.lat
	k = 1

	N2_vary = ComputeN2Xr(Tz, 'pfull')
	N2_const = ComputeN2Xr_CONST(N2_vary)
	N2 = [N2_vary, N2_const]
	labels = [r'$N^2(\phi,p)$', r'$N^2_{const}$']
	n = len(N2)
	print(datetime.now(), " - plotting N2")
	nlvls = np.arange(0, 0.00055, 0.00005)
	cmap = 'plasma'

	indir = '/disco/share/rm811/isca_data/'
	exp_L40 = 'PK_e0v4z13_q6m2y45l800u200'
	file_L40 = glob(indir+exp_L40+'/run0025/*interp.nc')[0]
	ds_L40 = xr.open_dataset(file_L40, decode_times=False)
	p_L40 = ds_L40.pfull

	fig, axes = plt.subplots(1, n, figsize=(n*4.5,5), layout="constrained")
	axes[0].set_ylabel('Pressure (hPa)', fontsize='xx-large')
	for i in range(n):
		csa = axes[i].contourf(lat, p, N2[i], levels=nlvls, extend='both', cmap=cmap)
		axes[i].set_ylim(max(p), 1) #goes to ~1hPa
		axes[i].set_yscale('log')
		axes[i].set_xlabel(r'Latitude ($\degree$N)', fontsize='xx-large')
		axes[i].set_xlim(0, max(lat))
		axes[i].set_xticks([0, 20, 40, 60, 80], ['0', '20', '40', '60', '80'])
		axes[i].text(2, 1.75, letters[i]+labels[i], color='k', fontsize='xx-large')
		axes[i].tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
		if i > 0:
			axes[i].tick_params(axis='y',label1On=False)
		elif i == 0:
			for lev in p_L40:
				axes[i].axhline(lev, c='w', lw=0.5)
	left, bottom, width, height = (40, 100, 40, -90)
	rect = plt.Rectangle((left, bottom), width, height, edgecolor='k', facecolor='none', linewidth=2)
	axes[0].add_patch(rect)
	cb  = fig.colorbar(csa, orientation='vertical', extend='both', pad=0.1)
	cb.set_label(label=r'Buoyancy Frequency, $N^{2}$', size='x-large')
	cb.ax.tick_params(labelsize='x-large')
	plt.savefig('N2_test.pdf', bbox_inches = 'tight')
	plt.show()
	plt.close()

	n2_N2_vary = climate.ComputeRefractiveIndexXr(uz,Tz,k,'lat','pfull')
	n2_N2_const = climate.ComputeRefractiveIndexXr(uz,Tz,k,'lat','pfull',N2const=N2_const)
	n2_N2_deriv_vary = ComputeRefractiveIndexXr_FULL(uz,Tz,k,'lat','pfull')
	n2 = [n2_N2_vary, n2_N2_const, n2_N2_deriv_vary]
	labels = [r'$M70n_{CR92}$', r'$M70n_{N_{const}}$', r'$M70n_{N_{\phi,p}}$']
	n = len(n2)

	print(datetime.now(), " - plotting n2")
	nlvls = np.arange(-200, 225, 25)
	N = len(nlvls)
	colors = find_colors('RdYlBu', N) # ('Blues', N)
	cmap = matplotlib.colors.ListedColormap(colors)
	cmap.set_under('#eeeeee')
	cmap.set_over('w')
	ulvls = np.arange(-70, 100, 10)

	fig, axes = plt.subplots(1, n, figsize=(n*4.5,5), layout="constrained")
	axes[0].set_ylabel('Pressure (hPa)', fontsize='xx-large')
	for i in range(n):
		csa = axes[i].contourf(lat, p, n2[i], levels=nlvls, extend='both', cmap=cmap)
		csb = axes[i].contour(lat, p, uz, colors='k', levels=ulvls, linewidths=1.5, alpha=0.25)
		csb.collections[list(ulvls).index(0)].set_linewidth(3)
		axes[i].set_ylim(max(p), 1) #goes to ~1hPa
		axes[i].set_yscale('log')
		axes[i].set_xlabel(r'Latitude ($\degree$N)', fontsize='xx-large')
		axes[i].set_xlim(0, max(lat))
		axes[i].set_xticks([0, 20, 40, 60, 80], ['0', '20', '40', '60', '80'])
		axes[i].text(2, 1.75, letters[i]+labels[i], color='k', fontsize='xx-large')
		axes[i].tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
		if i > 0:
			axes[i].tick_params(axis='y',label1On=False)
	cb  = fig.colorbar(csa, orientation='vertical', extend='both', pad=0.1)
	cb.set_label(label=r'Refractive Index Squared, $n_{k=1}^{2}$', size='x-large')
	cb.ax.tick_params(labelsize='x-large')        
	plt.savefig('n2_compare.pdf', bbox_inches = 'tight')
	plt.show()