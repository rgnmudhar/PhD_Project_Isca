"""
    Computes and plots EP flux vectors and divergence terms, based on Martin Jucker's code at https://github.com/mjucker/aostools/blob/d857987222f45a131963a9d101da0e96474dca63/climate.py
    Computes and plots meridional heat flux 
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as cm
import matplotlib
from aostools import climate
from shared_functions import *
from datetime import datetime

def PlotEPfluxArrows(x,y,ep1,ep2,fig,ax,xlim=None,ylim=None,xscale='linear',yscale='linear',invert_y=True, newax=False, pivot='tail',scale=None,quiv_args=None):
	"""Correctly scales the Eliassen-Palm flux vectors for plotting on a latitude-pressure or latitude-height axis.
		x,y,ep1,ep2 assumed to be xarray.DataArrays.
	INPUTS:
		x	: horizontal coordinate, assumed in degrees (latitude) [degrees]
		y	: vertical coordinate, any units, but usually this is pressure or height
		ep1	: horizontal Eliassen-Palm flux component, in [m2/s2]. Typically, this is ep1_cart from
				   ComputeEPfluxDiv()
		ep2	: vertical Eliassen-Palm flux component, in [U.m/s2], where U is the unit of y.
				   Typically, this is ep2_cart from ComputeEPfluxDiv(), in [hPa.m/s2] and y is pressure [hPa].
		fig	: a matplotlib figure object. This figure contains the axes ax.
		ax	: a matplotlib axes object. This is where the arrows will be plotted onto.
		xlim	: axes limits in x-direction. If None, use [min(x),max(x)]. [None]
		ylim	: axes limits in y-direction. If None, use [min(y),max(y)]. [None]
		xscale	: x-axis scaling. currently only 'linear' is supported. ['linear']
		yscale	: y-axis scaling. 'linear' or 'log' ['linear']
		invert_y: invert y-axis (for pressure coordinates). [True]
		newax	: plot on second y-axis. [False]
		pivot	: keyword argument for quiver() ['tail']
		scale	: keyword argument for quiver(). Smaller is longer [None]
				  besides fixing the length, it is also usefull when calling this function inside a
				   script without display as the only way to have a quiverkey on the plot.
               quiv_args: further arguments passed to quiver plot.
	OUTPUTS:
	   Fphi*dx : x-component of properly scaled arrows. Units of [m3.inches]
	   Fp*dy   : y-component of properly scaled arrows. Units of [m3.inches]
	   ax	: secondary y-axis if newax == True
	"""
	import numpy as np
	import matplotlib.pyplot as plt
	#
	def Deltas(z,zlim):
		# if zlim is None:
		return np.max(z)-np.min(z)
		# else:
			# return zlim[1]-zlim[0]
	# Scale EP vector components as in Edmon, Hoskins & McIntyre JAS 1980:
	cosphi = np.cos(np.deg2rad(x))
	a0 = 6376000.0 # Earth radius [m]
	grav = 9.81
	# first scaling: Edmon et al (1980), Eqs. 3.1 & 3.13
	Fphi = 2*np.pi/grav*cosphi**2*a0**2*ep1 # [m3.rad]
	Fp   = 2*np.pi/grav*cosphi**2*a0**3*ep2 # [m3.hPa]
	#
	# Now comes what Edmon et al call "distances occupied by 1 radian of
	#  latitude and 1 [hecto]pascal of pressure on the diagram."
	# These distances depend on figure aspect ratio and axis scale
	#
	# first, get the axis width and height for
	#  correct aspect ratio
	width,height = climate.GetAxSize(fig,ax)
	# we use min(),max(), but note that if the actual axis limits
	#  are different, this will be slightly wrong.
	delta_x = Deltas(x,xlim)
	delta_y = Deltas(y,ylim)
	#
	#scale the x-axis:
	if xscale == 'linear':
		dx = width/delta_x/np.pi*180
	else:
		raise ValueError('ONLY LINEAR X-AXIS IS SUPPORTED AT THE MOMENT')
	#scale the y-axis:
	if invert_y:
		y_sign = -1
	else:
		y_sign = 1
	if yscale == 'linear':
		dy = y_sign*height/delta_y
	elif yscale == 'log':
		dy = y_sign*height/y/np.log(np.max(y)/np.min(y))
	#
	# plot the arrows onto axis
	quivArgs = {'angles':'uv','scale_units':'inches','pivot':pivot}
	if quiv_args is not None:
		for key in quiv_args.keys():
			quivArgs[key] = quiv_args[key]
	if scale is not None:
		quivArgs['scale'] = scale
	if newax:
		ax = ax.twinx()
		ax.set_ylabel('pressure [hPa]')
	try:
		Q = ax.quiver(x,y,Fphi*dx,Fp*dy,**quivArgs)
	except:
		Q = ax.quiver(x,y,dx*Fphi.transpose(),dy*Fp.transpose(),**quivArgs)
	if scale is None:
		fig.canvas.draw() # need to update the plot to get the Q.scale
		U = Q.scale
	else:
		U = scale
	if U is not None: # when running inside a script, the figure might not exist and therefore U is None
		ax.quiverkey(Q,0.9,1.02,U/width,label=r'{0:.1e}$\,m^3$'.format(U),labelpos='W',coordinates='axes') # CHANGED LABELPOS FROM E TO W !!!
	if invert_y:
		ax.invert_yaxis()
	if xlim is not None:
		ax.set_xlim(xlim)
	if ylim is not None:
		ax.set_ylim(ylim)
	ax.set_yscale(yscale)
	ax.set_xscale(xscale)
	#
	if newax:
		return Fphi*dx,Fp*dy,ax
	else:
		return Fphi*dx,Fp*dy

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

    sinlat = np.sin(np.deg2rad(Tz[lat]))
    f = 2*Omega*sinlat
    f2 = f*f

    N2 = climate.ComputeN2Xr(Tz,pres,H,Rd,cp)
    N = np.sqrt(N2)
    
    C = -f2/N2

    X = 1 / (4 * H2)
    
    p = Tz[pres]
    dNdp = N.differentiate(pres,edge_order=2)
    d2Ndp2 = dNdp.differentiate(pres,edge_order=2)

    Y = (-1/N) * ((3*p/H2)*dNdp + (p**2/H2)*d2Ndp2)

    Z = ((2*p**2) / (H2 * N2)) * (dNdp)**2

    return C * (X + Y + Z)

def ComputeN2Xr_CONST(Tz):
    '''
    Find a constant N2 following Weinberger et al. (2021).
    Average over 100-10 hPa, 40-80N
    '''
    N2_vary = climate.ComputeN2Xr(Tz, 'pfull')
    N2_const = N2_vary.sel(pfull=slice(10,100)).mean('pfull')
    N2_const = N2_const.sel(lat=slice(40,80)).mean('lat')
    # make a new array
    N2_const_full = np.full_like(N2_vary, N2_const)
    N2_const_full = xr.DataArray(N2_const_full, coords=[N2_vary.pfull, N2_vary.lat], dims=['pfull', 'lat'])
    return N2_const_full

def ComputeRefractiveIndexXr_NEW(uz,Tz,k,lat='lat',pres='level',ulim=None):
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

def open_data1(dir, exp):
    u = xr.open_dataset(dir+exp+'_u.nc', decode_times=False).ucomp
    v = xr.open_dataset(dir+exp+'_v.nc', decode_times=False).vcomp
    w = xr.open_dataset(dir+exp+'_w.nc', decode_times=False).omega/100 # Pa --> hPa
    T = xr.open_dataset(dir+exp+'_T.nc', decode_times=False).temp
    utz = xr.open_dataset(dir+exp+'_utz.nc', decode_times=False).ucomp[0]
    return utz, u, v, w, T

def open_data2(exp, extension):
    path = '/disco/share/rm811/'
    folder = 'epflux/'
    filename = exp + '_' + extension
    variable = xr.open_dataset(path + folder + filename + '.nc', decode_times=False)
    return variable

def save_variable(variable, exp, extension):
    print(datetime.now(), " - saving "+extension)
    path = '/disco/share/rm811/'
    folder = 'epflux/'
    filename = exp + '_' + extension
    coord_list = ["pfull", "lat"]
    coords = variable.coords
    variable = xr.Dataset(data_vars=dict(variable = (coord_list, variable.transpose("pfull","lat").values)), coords=coords)
    variable = variable.rename({"variable" : extension})
    variable.to_netcdf(path + folder + filename + '.nc', format="NETCDF3_CLASSIC",
            encoding = {extension: {"dtype": 'float32', '_FillValue': None},
                "lat": {'_FillValue': None}, "pfull": {'_FillValue': None}})
    
def calc_ep(exp, u, v, w, t, k):
    print(datetime.now(), " - finding ep fluxes")
    ep1, ep2, div1, div2 = climate.ComputeEPfluxDivXr(u, v, t, 'lon', 'lat', 'pfull', 'time', w=w, do_ubar=True, wave=k) # default w=None and do_ubar=False for QG approx.
    # take time mean of relevant quantities
    print(datetime.now(), ' - taking time mean')
    div = div1 + div2
    div = div.mean(dim='time')
    ep1 = ep1.mean(dim='time')
    ep2 = ep2.mean(dim='time')

    save_variable(div, exp, "div")
    save_variable(ep1, exp, "ep1")
    save_variable(ep2, exp, "ep2")

def plot_ep(uz, div, ep1, ep2, k, exp_name, heat, type):
    print(datetime.now(), " - plotting EP Fluxes")
    p = uz.coords['pfull']
    lat = uz.coords['lat']
    if type == 'single':
        divlvls = np.arange(-12,13,1)

    elif type == 'diff':
        divlvls = np.arange(-5,5.5,0.5)
        exp_name = exp_name+'_diff'

    #Filled contour plot of time-mean EP flux divergence plus EP flux arrows and zonal wind contours
    fig, ax = plt.subplots(figsize=(6,6), constrained_layout=True)
    print(datetime.now(), " - plot uz")
    uz.plot.contour(colors='k', linewidths=0.5, alpha=0.4, levels=ulvls)
    #print(datetime.now(), " - plot polar heat")
    #plt.contour(lat, p, heat, colors='g', linewidths=0.25, alpha=0.4, levels=11)
    print(datetime.now(), " - plot EP flux divergence")
    cs = div.plot.contourf(levels=divlvls, cmap='RdBu_r', add_colorbar=False)
    cb = plt.colorbar(cs)
    cb.set_label(label=r'Divergence (m s$^{-1}$ day$^{-1}$)', size='large')
    cb.ax.set_yticks(divlvls)
    fig.canvas.draw()
    ticklabs = cb.ax.get_yticklabels()
    cb.ax.set_yticklabels(ticklabs, fontsize='large')
    print(datetime.now(), " - plot EP flux arrows")
    ax = climate.PlotEPfluxArrows(lat, p, ep1, ep2, fig, ax, yscale='log')
    plt.yscale('log')
    plt.ylim(max(p), 1) #to 1 hPa
    plt.xlabel(r'Latitude ($\degree$N)', fontsize='xx-large')
    plt.xlim(0,90)
    plt.xticks([10, 30, 50, 70, 90], ['10', '30', '50', '70', '90'])
    plt.ylabel('Pressure (hPa)', fontsize='xx-large')
    plt.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
    plt.savefig(exp_name+'_EPflux_k{0:.0f}.pdf'.format(k), bbox_inches = 'tight')
    return plt.close()

def vT_calc(exp):
    print(datetime.now(), " - opening files for ", exp)
    v = xr.open_dataset(indir+exp+'_v.nc', decode_times=False).vcomp
    T = xr.open_dataset(indir+exp+'_T.nc', decode_times=False).temp
    vz = xr.open_dataset(indir+exp+'_vz.nc', decode_times=False).vcomp
    Tz = xr.open_dataset(indir+exp+'_Tz.nc', decode_times=False).temp
    print(datetime.now(), " - finding anomalies")
    vp = v - vz
    Tp = T - Tz
    print(datetime.now(), " - finding v'T'")
    vpTp = (vp*Tp)        
    return vpTp

def uv_calc(exp):
    print(datetime.now(), " - opening files for ", exp)
    u = xr.open_dataset(indir+exp+'_u.nc', decode_times=False).ucomp
    v = xr.open_dataset(indir+exp+'_v.nc', decode_times=False).vcomp
    uz = xr.open_dataset(indir+exp+'_uz.nc', decode_times=False).ucomp
    vz = xr.open_dataset(indir+exp+'_vz.nc', decode_times=False).vcomp
    print(datetime.now(), " - finding anomalies")
    up = u - uz
    vp = v - vz
    print(datetime.now(), " - finding u'v'")
    upvp = (up*vp)        
    return upvp

def vT_level(vpTp, p, lats):
    # Meridional heat flux weighted by cos(lat) and meridionally averaged from 75 to 90 N at p hPa
    # Based on Dunn-Sigouin & Shaw (2015) but their polar cap was 60 to 90 N
    vpTp_w = vpTp.sel(pfull=p, method='nearest') / np.cos(np.deg2rad(vpTp.lat))
    vpTp_sub = vpTp_w.sel(lat=lats).mean('lat')
    vpTp_bar = vpTp_sub.mean('lon')
    return vpTp_bar

def comparison(var, lats):
    print(datetime.now(), " - addition")
    if lats == 60:
        compare = [var[0].sel(lat=lats, method='nearest').mean(('lon', 'time')),\
            var[1].sel(lat=lats, method='nearest').mean(('lon', 'time')),\
            var[2].sel(lat=lats, method='nearest').mean(('lon', 'time'))]
    else:
        compare = [var[0].sel(lat=lats).mean(('lat', 'lon', 'time')),\
            var[1].sel(lat=lats).mean(('lat', 'lon', 'time')),\
            var[2].sel(lat=lats).mean(('lat', 'lon', 'time'))]
    compare.append(compare[0]+compare[1]) # addition after taking means
    compare.append(compare[3]-compare[2])
    return compare

def linear_add(compare, p, label, lats_label):
    xlabel = lats_label+r" mean v'T' magnitude (K m s$^{-1}$)"
    names = ['mid-lat heat only (a)', 'polar heat only (b)', 'combined simulation (c)', 'linear component (d=a+b)', '-1 x non-linear component -(c-d)']
    colors = ['#B30000', '#0099CC', 'k', '#4D0099', '#CC0080']
    lines = ['--', ':', '-', '-.', ':']   
    print(datetime.now(), " - plotting")
    fig, ax = plt.subplots(figsize=(8,5.5))
    for i in range(len(compare)):
        ax.plot(compare[i].transpose(), p, color=colors[i], linestyle=lines[i], label=names[i], linewidth=1.75)
    ax.set_xlim(-5, 125)
    ax.axvline(0, color='k', linewidth=0.25)
    ax.set_xlabel(xlabel, fontsize='large')
    ax.set_ylabel('Pressure (hPa)', fontsize='large')
    ax.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
    plt.legend(fancybox=False, ncol=1, loc='lower right', fontsize='large', labelcolor = colors)
    plt.ylim(max(p), 1)
    plt.yscale('log')
    plt.savefig('vT_addvcombo_'+label+lats_label+'.pdf', bbox_inches = 'tight')
    return plt.close()

def plot_vT(u, vT, exp, heat, lvls, colors):
    fig, ax = plt.subplots(figsize=(6,6))
    cs1 = ax.contourf(lat, p, vT, levels=lvls, cmap=colors)
    ax.contourf(cs1, colors='none')
    cb = plt.colorbar(cs1)
    cb.set_label(label=r"v'T' (K m s$^{-1}$)", size='x-large')
    cb.ax.tick_params(labelsize='x-large')
    cs2 = ax.contour(lat, p, u, colors='k', levels=ulvls, linewidths=0.5, alpha=0.2)
    cs2.collections[int(len(ulvls)/2)].set_linewidth(1)
    plt.contour(lat, p, heat, colors='g', linewidths=1, alpha=0.4, levels=11)
    plt.xlabel(r'Latitude ($\degree$N)', fontsize='x-large')
    plt.xlim(0,90)
    plt.xticks([10, 30, 50, 70, 90], ['10', '30', '50', '70', '90'])
    plt.ylabel('Pressure (hPa)', fontsize='x-large')
    plt.ylim(max(p), 1) #goes to ~1hPa
    plt.yscale('log')
    plt.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    plt.savefig(exp+'_vT.pdf', bbox_inches = 'tight')
    return plt.close()

def plot_stats(stat, p, exp, ext, lab):
    print(datetime.now(), " - plotting v'T' ", lab)
    colors = ['#B30000', '#00B300', '#0099CC', 'k']
    fig, ax = plt.subplots(figsize=(8,6))
    for j in range(len(stat)):
        ax.plot(labels, stat[j], marker='o', linewidth=1.25, color=colors[j], linestyle=':', label='{:.0f} hPa'.format(p[j]))
    ax.set_xticks(labels)
    ax.set_xlabel(xlabel, fontsize='x-large')
    ax.set_ylabel("Polar Cap Average v'T' "+lab+r" (K m s$^{-1}$)", fontsize='x-large')
    ax.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    plt.legend(loc='right',fancybox=False, shadow=True, ncol=1, fontsize='large')
    plt.savefig(exp+ext+'_vT_'+lab+'.pdf', bbox_inches = 'tight')
    return plt.close()

def check_vs_MERRA(exp):
    print(datetime.now(), " - setting up MERRA2 data")
    vars = ['uv', 'vT', 'z1', 'z2']
    merra2_p = [100, 70, 50, 30, 10]
    merra2_vT_4575mean = [17.2502638623327, 21.3728413001912, 26.7985621414914, 38.4797055449332, 75.6868871892927]
    merra2_uv_4575mean = [-3.64865774378585, -2.9196845124283, 1.50874952198854, 17.1705812619504, 92.7788451242831]
    merra2_z1_60mean = [198.617317399618, 251.389999999999, 319.728965583174, 451.971359464627, 781.923244741875]
    merra2_z2_60mean = [192.398684512428, 210.373868068833, 231.328782026768, 267.965894837477, 338.587994263863]
    #merra2_vT_4575lims = [(-17.89,76.16), (-21.52,103.93), (-33.87,143.06), (-61.26,236.09), (-103.91,564.32)]
    #merra2_uv_4575lims = [(-75.68,74.33), (-90.66,110.7), (-114.64,142.76), (-140.05,266.22), (-208.51,575.8)]
    #merra2_z1_60lims = [(2.53,569.56), (4.01,750.82), (8.37,931.77), (2.54,1196.42), (6.79,1748.62)]
    #merra2_z2_60lims = [(1.19,643.51), (2.61,764.96), (4.02,879.78), (1.45,1023.24), (6.38,1214.21)]
    merra2_data = [merra2_uv_4575mean, merra2_vT_4575mean, merra2_z1_60mean, merra2_z2_60mean]
    #merra2_lims = [merra2_uv_4575lims, merra2_vT_4575lims, merra2_z1_60lims, merra2_z2_60lims]

    print(datetime.now(), ' - opening Isca data')
    uv = uv_calc(exp)
    vT = vT_calc(exp)
    print(datetime.now(), ' - wave decomposition')
    z_60 = xr.open_dataset(indir+exp+'_h.nc', decode_times=False).height.sel(lat=60, method='nearest')
    waves = climate.GetWavesXr(z_60)
    print(datetime.now(), ' - select relevant data')
    z1_60mean = np.abs(waves.sel(k=1)).mean(('lon', 'time'))
    z2_60mean = np.abs(waves.sel(k=2)).mean(('lon', 'time'))
    p = uv.pfull
    uv_4575mean = uv.sel(lat=slice(45,75)).mean(('lat', 'lon', 'time'))
    vT_4575mean = vT.sel(lat=slice(45,75)).mean(('lat', 'lon', 'time'))
    isca_data = [uv_4575mean, vT_4575mean, z1_60mean, z2_60mean]

    print(datetime.now(), " - plotting")
    colors = ['k', '#B30000']
    lines = ['--', '-']
    labels = ['MERRA2 data', 'control simulation']
    units = [r"v'T' (K m s$^{-1}$)", r"u'v' (m$^{2}$ s$^{-2}$)", 'Wave-1 GPH (m)', 'Wave-2 GPH (m)']
    for i in range(len(vars)):
        fig, ax = plt.subplots(figsize=(6,6))
        ax.plot(merra2_data[i], merra2_p, color=colors[0], linestyle=lines[0], label=labels[0])
        ax.plot(isca_data[i], p, linewidth=1.25, color=colors[1], linestyle=lines[1], label=labels[1])
        plt.xlabel(r"$45-75\degree$N mean "+units[i], fontsize='x-large')
        plt.ylabel('Pressure (hPa)', fontsize='x-large')
        plt.ylim(max(p), 1)
        plt.yscale('log')
        plt.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
        plt.legend(fancybox=False, ncol=1, fontsize='x-large')
        plt.savefig(exp+'_'+vars[i]+'_vsMERRA2.pdf', bbox_inches = 'tight')
        plt.show()
    return plt.close()

def refractive_index(u, T, k):
    print(datetime.now(), " - finding n2")
    #n2 = climate.ComputeRefractiveIndexXr(u,T,k,'lat','pfull',N2const=ComputeN2Xr_CONST(T))
    n2 = ComputeRefractiveIndexXr_NEW(u,T,k,'lat','pfull')
    return n2

def plot_n2_1(exp, k, name):
    # Plots vortex experiments
    print(datetime.now(), " - opening files")
    u = []
    n2 = []
    n = len(exp)
    for i in range(n):
        utz = xr.open_dataset(indir+exp[i]+'_utz.nc', decode_times=False).ucomp[0]
        Ttz = xr.open_dataset(indir+exp[i]+'_Ttz.nc', decode_times=False).temp[0]
        r = refractive_index(utz, Ttz, k)
        u.append(utz)
        n2.append(r)

    p = utz.pfull
    lat = utz.lat
        

    print(datetime.now(), " - plotting")
    nlvls = np.arange(-200, 225, 25)
    N = len(nlvls)
    colors = find_colors('RdYlBu', N) # ('Blues', N)
    #colors = ['#bbd6eb', '#88bedc', '#549ecd',  '#2a7aba', '#0c56a0', '#08306b']
    cmap = matplotlib.colors.ListedColormap(colors)
    cmap.set_under('#eeeeee')
    cmap.set_over('w')

    fig, axes = plt.subplots(1, n, figsize=(n*4.5,5), layout="constrained")
    axes[0].set_ylabel('Pressure (hPa)', fontsize='xx-large')
    for i in range(n):
        csa = axes[i].contourf(lat, p, n2[i], levels=nlvls, extend='both', cmap=cmap)
        csb = axes[i].contour(lat, p, u[i], colors='k', levels=ulvls, linewidths=1.5, alpha=0.25)
        csb.collections[list(ulvls).index(0)].set_linewidth(3)
        if i > 0:
            h_name = exp[i][11:27]
            h = xr.open_dataset('../Inputs/' + h_name + '.nc')
            heat = h.mean('lon').variables[h_name]
            h_p = h.pfull
            h_lat = h.lat
            h_lvls = np.arange(2.5e-6, 1e-4, 5e-6)
            axes[i].contour(h_lat, h_p, heat, alpha=0.5, colors='g', levels=h_lvls)
        axes[i].text(2, 1.75, labels[i], color='k', fontsize='xx-large') #letters[i]+labels[i]
        axes[i].set_ylim(max(p), 1) #goes to ~1hPa
        axes[i].set_yscale('log')
        axes[i].set_xlabel(r'Latitude ($\degree$N)', fontsize='xx-large')
        axes[i].set_xlim(0, max(lat))
        axes[i].set_xticks([0, 20, 40, 60, 80], ['0', '20', '40', '60', '80'])
        axes[i].tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
        if i > 0:
            axes[i].tick_params(axis='y',label1On=False)
    cb  = fig.colorbar(csa, orientation='vertical', extend='both', pad=0.1)
    cb.set_label(label=r'Refractive Index Squared, $n_{k=1}^{2}$', size='x-large')
    cb.ax.tick_params(labelsize='x-large')        
    plt.savefig(name+'_n2_k{0:.0f}.pdf'.format(k), bbox_inches = 'tight')
    plt.show()
    return plt.close()

def plot_n2_2(exp, k, name):
    # Plots control, then 3 experiments of your choice
    print(datetime.now(), " - opening files")
    u = []
    n2 = []
    heat = []
    for i in range(len(exp)):
        utz = xr.open_dataset(indir+exp[i]+'_utz.nc', decode_times=False).ucomp[0]
        Ttz = xr.open_dataset(indir+exp[i]+'_Ttz.nc', decode_times=False).temp[0]
        n = refractive_index(utz, Ttz, k)
        u.append(utz)
        n2.append(n)
        if i == 0:
             heat.append(0)
        elif i > 0:
            h_name = exp[i][11:27]
            h = xr.open_dataset('../Inputs/' + h_name + '.nc')
            heat.append(h.mean('lon').variables[h_name])

    p = utz.pfull
    lat = utz.lat
    nlvls = np.arange(-200, 225, 25)
    N = len(nlvls)
    colors = find_colors('RdYlBu', N) # ('Blues', N)
    #colors = ['#bbd6eb', '#88bedc', '#549ecd',  '#2a7aba', '#0c56a0', '#08306b']
    cmap = matplotlib.colors.ListedColormap(colors)
    cmap.set_under('#eeeeee')
    cmap.set_over('w')
    h_p = h.pfull
    h_lat = h.lat
    h_lvls = np.arange(2.5e-6, 1e-4, 5e-6)

    print(datetime.now(), " - plotting")
    n = len(exp)
    #fig, axes = plt.subplots(1, n, figsize=(n*4.5,5), layout="constrained")
    fig, axes = plt.subplots(1, n, figsize=(n*4.2,6.4), layout="constrained")
    for i in range(n):
        csa = axes[i].contourf(lat, p, n2[i], levels=nlvls, extend='both', cmap=cmap)
        csb = axes[i].contour(lat, p, u[i], colors='k', levels=ulvls, linewidths=1.5, alpha=0.25)
        csb.collections[list(ulvls).index(0)].set_linewidth(3)
        axes[i].text(2, 1.75, labels[i], color='k', fontsize='xx-large') #letters[i]+labels[i]
        axes[i].set_ylim(max(p), 1) #goes to ~1hPa
        axes[i].set_yscale('log')
        axes[i].set_xlabel(r'Latitude ($\degree$N)', fontsize='xx-large')
        axes[i].set_xlim(0, max(lat))
        axes[i].set_xticks([0, 20, 40, 60, 80], ['0', '20', '40', '60', '80'])
        axes[i].tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
        if i == 0:
            axes[i].set_ylabel('Pressure (hPa)', fontsize='xx-large')
        if i > 0:
            axes[i].contour(h_lat, h_p, heat[i], alpha=0.5, colors='g', levels=h_lvls)
            axes[i].tick_params(axis='y',label1On=False)

    #cb  = fig.colorbar(csa, orientation='vertical', extend='both', pad=0.1)
    cb  = fig.colorbar(csa, ax=axes.ravel().tolist()[1:3], orientation='horizontal', extend='both', shrink=0.6)
    cb.set_label(label=r'Refractive Index Squared, $n_{k=1}^{2}$', size='x-large')
    cb.ax.tick_params(labelsize='x-large')
    plt.savefig(name+'_n2_k{0:.0f}.pdf'.format(k), bbox_inches = 'tight')
    plt.show()
    return plt.close()

def plot_EP_1(u, div_response, ep1_response, ep2_response, labels, heat, name, n):
    # Plots difference between no polar heat and with polar heat
    print(datetime.now(), " - plotting")
    p = u[0].pfull
    lat = u[0].lat
    lvls = np.arange(-6,7,1)
    h_lvls = np.arange(2.5e-6, 1e-4, 5e-6)

    fig, axes = plt.subplots(1, n, figsize=(n*4.5,5), layout="constrained")
    axes[0].set_ylabel('Pressure (hPa)', fontsize='xx-large')
    for i in range(len(axes)):
        csa = axes[i].contourf(lat, p, div_response[i], levels=lvls, extend='both', cmap='RdBu_r')
        csb = axes[i].contour(lat, p, u[i], colors='k', levels=ulvls, linewidths=1.5, alpha=0.25)
        csb.collections[list(ulvls).index(0)].set_linewidth(3)
        axes[i].contour(h_lat, h_p, heat, alpha=0.5, colors='w', levels=h_lvls)
        PlotEPfluxArrows(lat, p, ep1_response[i], ep2_response[i], fig, axes[i], yscale='log')
        axes[i].text(2, 1.75, letters[i]+labels[i], color='k', fontsize='xx-large')
        axes[i].set_ylim(max(p), 1) #goes to ~1hPa
        axes[i].set_yscale('log')
        axes[i].set_xlabel(r'Latitude ($\degree$N)', fontsize='xx-large')
        axes[i].set_xlim(0, max(lat))
        axes[i].set_xticks([0, 20, 40, 60, 80], ['0', '20', '40', '60', '80'])
        axes[i].tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
        if i > 0:
            axes[i].tick_params(axis='y',label1On=False)

    cb  = fig.colorbar(csa, orientation='vertical', extend='both', pad=0.1)
    cb.set_label(label=r'Divergence Response (m s$^{-1}$ day$^{-1}$)', size='x-large')
    cb.ax.tick_params(labelsize='x-large')
    plt.savefig(name+'_EP.pdf', bbox_inches = 'tight')
    return plt.close()

def plot_EP_2(u, div_ctrl, ep1_ctrl, ep2_ctrl, div_response, ep1_response, ep2_response, heat, name):
    # Plots 3 experiments of your choice
    lvls = [np.arange(-12,13,1), np.arange(-6,7,1)]
    p = u[0].pfull
    lat = u[0].lat
    h_lvls = np.arange(2.5e-6, 1e-4, 5e-6)

    print(datetime.now(), " - plotting control")
    n = len(exp)-1
    fig, axes = plt.subplots(1, n, figsize=(n*4.5,5), layout="constrained")
    # Uncomment the following to include the control as panel 1
    #csa_ctrl = axes[0].contourf(lat, p, div_ctrl, levels=lvls[0], cmap='RdBu_r', extend='both')
    #cb_ctrl  = fig.colorbar(csa_ctrl, ax=axes[0], orientation='horizontal', extend='both')
    #cb_ctrl.set_label(label=r'Divergence (m s$^{-1}$ day$^{-1}$)', size='xx-large')
    #cb_ctrl.ax.tick_params(labelsize='x-large')
    #csb_ctrl = axes[0].contour(lat, p, u[0], colors='k', levels=ulvls, linewidths=1.5, alpha=0.25)
    #csb_ctrl.collections[list(ulvls).index(0)].set_linewidth(3)
    #PlotEPfluxArrows(lat, p, ep1_ctrl, ep2_ctrl, fig, axes[0], yscale='log')
    axes[0].set_ylabel('Pressure (hPa)', fontsize='xx-large')    

    print(datetime.now(), " - plotting responses")
    for i in range(len(exp)):
        if i > 0:
            csa = axes[i-1].contourf(lat, p, div_response[i-1], levels=lvls[1], cmap='RdBu_r', extend='both')
            csb = axes[i-1].contour(lat, p, u[i], colors='k', levels=ulvls, linewidths=1.5, alpha=0.25)
            csb.collections[list(ulvls).index(0)].set_linewidth(3)
            axes[i-1].contour(h_lat, h_p, heat[i-1], colors='g', linewidths=1.5, alpha=0.5, levels=h_lvls)

    cb  = fig.colorbar(csa, orientation='vertical', extend='both', pad=0.1)
    cb.set_label(label=r'Divergence Response (m s$^{-1}$ day$^{-1}$)', size='xx-large')
    cb.ax.tick_params(labelsize='x-large')

    for i in range(len(axes)):
        axes[i].contour(h_lat, h_p, heat[i], alpha=0.5, colors='g', levels=h_lvls)
        PlotEPfluxArrows(lat, p, ep1_response[i], ep2_response[i], fig, axes[i], yscale='log')
        axes[i].text(2, 1.75, letters[i]+labels[i+1], color='k', fontsize='xx-large')
        axes[i].set_ylim(max(p), 1) #goes to ~1hPa
        axes[i].set_yscale('log')
        axes[i].set_xlabel(r'Latitude ($\degree$N)', fontsize='xx-large')
        axes[i].set_xlim(0, max(lat))
        axes[i].set_xticks([0, 20, 40, 60, 80], ['0', '20', '40', '60', '80'])
        axes[i].tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
        if i == 0:
            axes[i].set_ylabel('Pressure (hPa)', fontsize='xx-large')
        if i > 0:
            axes[i].tick_params(axis='y',label1On=False)
    plt.savefig(name+'_EP.pdf', bbox_inches = 'tight', pad_inches=0.25)
    return plt.close()

def control_plot(exp, label, u, div_ctrl, ep1_ctrl, ep2_ctrl, heat, name):
     # Plots control
    lvls = np.arange(-12,13,1)
    p = u.pfull
    lat = u.lat
    h_lvls = np.arange(5e-6, 7e-5, 1e-5)

        
    print(datetime.now(), " - plotting control")
    n = 1
    #fig, ax = plt.subplots(1, n, figsize=(6,5), layout="constrained")
    fig, ax = plt.subplots(1, n, figsize=(5,6.5), layout="constrained")
    csa_ctrl = ax.contourf(lat, p, div_ctrl, levels=lvls, cmap='PuOr_r', extend='both')
    #cb_ctrl  = fig.colorbar(csa_ctrl, orientation='vertical', extend='both', pad=0.1)
    cb_ctrl  = fig.colorbar(csa_ctrl, orientation='horizontal', extend='both')
    cb_ctrl.set_label(label=r'Divergence (m s$^{-1}$ day$^{-1}$)', size='xx-large')
    cb_ctrl.ax.tick_params(labelsize='x-large')
    csb_ctrl = ax.contour(lat, p, u, colors='k', levels=ulvls, linewidths=1.5, alpha=0.25)
    csb_ctrl.collections[list(ulvls).index(0)].set_linewidth(3)
    PlotEPfluxArrows(lat, p, ep1_ctrl, ep2_ctrl, fig, ax, yscale='log')
    #ax.hlines(y=70, xmin=45, xmax=55, linewidth=2.5, color='r')
    #ax.contour(h_lat, h_p, heat, alpha=0.5, colors='g', levels=h_lvls)

    #print(datetime.now(), " - finding tropopause")
    #trop = tropopause(indir, basis)[-1]
    #ax.plot(lat, trop, linewidth=2, color='k', linestyle='--')

    ax.text(2, 1.5, label, color='k', fontsize='xx-large')
    ax.set_ylabel('Pressure (hPa)', fontsize='xx-large')    
    ax.set_ylim(max(p), 1) #goes to ~1hPa
    ax.set_yscale('log')
    ax.set_xlabel(r'Latitude ($\degree$N)', fontsize='xx-large')
    ax.set_xlim(0, max(lat))
    ax.set_xticks([0, 20, 40, 60, 80], ['0', '20', '40', '60', '80'])
    ax.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
    plt.savefig(name+'_EP.pdf', bbox_inches = 'tight')
    return plt.close()

if __name__ == '__main__': 
    #Set-up data to be read in
    indir = '/disco/share/rm811/processed/'
    basis = 'PK_e0v4z13'
    flux = input("a) calculate ep fluxes or ... plot b) upward EP Flux, c) v'T', d) vs. MERRA2, or e) report plot?")
    var_type = input("Plot a) depth, b) width, c) location, d) strength, e) vortex experiments? or f) test?")
    if var_type == 'a':
        extension = '_depth'
    elif var_type == 'b':
        extension = '_width'
    elif var_type == 'c':
        extension = '_loc'
    elif var_type == 'd':
        extension = '_strength'
    elif var_type == 'e':
        basis = 'PK_e0vXz13'
        extension = '_vtx'
    elif var_type == 'f':
        extension = '_test'
    exp, labels, xlabel = return_exp(extension)
    colors = ['k', '#B30000', '#FF9900', '#FFCC00', '#00B300', '#0099CC', '#4D0099', '#CC0080']
    blues = ['k', '#dbe9f6', '#bbd6eb', '#88bedc', '#549ecd',  '#2a7aba', '#0c56a0', '#08306b']
    letters = ['d) ', 'e) ', 'f) ', 'g) ', 'a) ', 'b) ', 'c) ']
    ulvls = np.arange(-70, 100, 10)
    k = int(input('Which wave no.? (i.e. 0 for all, 1, 2, etc.)'))

    if flux == 'a':
        n = len(exp)
        if n == 10:
            for j in range(n):
                for i in range(len(exp[j])):
                    print(datetime.now(), " - opening files ({0:.0f}/{1:.0f})".format(i+1, len(exp[j])))
                    utz, u, v, w, T = open_data1(indir, exp[j][i])
                    print(datetime.now(), " - finding EP flux")
                    calc_ep(exp[j][i], u, v, w, T, k)
        else:
            for i in range(len(exp)):
                print(datetime.now(), " - opening files ({0:.0f}/{1:.0f})".format(i+1, len(exp)))
                utz, u, v, w, T = open_data1(indir, exp[i])
                print(datetime.now(), " - finding EP flux")
                calc_ep(exp[i], u, v, w, T, k)

    elif flux == 'b':
        p = int(input('At which pressure level? (i.e. 10 or 100 hPa) '))
        plot_or_not = input('a) plot PDFs or b) find mean upward EP flux only? ')
        if plot_or_not == 'a':
            print(datetime.now(), " - plotting PDFs at {:.0f} hPa".format(p))
            x_min = x_max = 0
            fig, ax = plt.subplots(figsize=(6,6))
            for i in range(len(exp)):
                print(datetime.now(), " - opening files ({0:.0f}/{1:.0f})".format(i+1, len(exp)))
                u, v, w, t = open_data(indir, exp[i])[1:]
                print(datetime.now(), " - finding EP flux")
                ep1, ep2, div1, div2 = climate.ComputeEPfluxDivXr(u, v, t, 'lon', 'lat', 'pfull', 'time', w=w, do_ubar=True, wave=0)
                x = ep2.sel(pfull=p,method='nearest').sel(lat=slice(45,75)).mean('lat')
                x_sort, f, m = pdf(x)
                if max(x) > x_max:
                    x_max = max(x)
                if min(x) < x_min:
                    x_min = min(x)
                print(datetime.now(), ' - plotting')
                ax.plot(x_sort, f, linewidth=1.25, color=blues[i], label=labels[i])
            ax.axvline(0, color='k', linewidth=0.25)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(bottom=0)
            ax.set_xlabel(r'45-75$\degree$N,'+str(p)+r'hPa EP$_z$ (hPa m s$^{-2}$)', fontsize='xx-large')
            ax.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
            plt.legend(fancybox=False, ncol=1, fontsize='x-large')
            plt.savefig(basis+'_{:.0f}pdf.pdf'.format(p), bbox_inches = 'tight')
            plt.show()
            plt.close()
        elif plot_or_not == 'b':
            for i in range(len(exp)):
                ep2 = open_data2(exp[i], "ep2").ep2
                x = ep2.sel(pfull=p,method='nearest').sel(lat=slice(40,60)).mean('lat')
                print('{0} upward EP flux through 100 hPa averaged across 40-60N: {1:.6f} '.format(exp[i], x), r'(hPa m s$^{-2}$)')
    
    elif flux == 'c':
        plot_type = input("Plot a) lat-p climatology and variability or b) linear addition?")
        if plot_type == 'a':
            vpTp = []
            for i in range(len(exp)):
                utz = xr.open_dataset(indir+exp[i]+'_utz.nc', decode_times=False).ucomp[0]
                vT = vT_calc(exp[i])
                vpTp.append(vT)
                vT_iz = vpTp[i].mean('lon')
                vT_itz = vT_iz.mean('time')
                lat, p, sd = find_sd(vT_iz)
                if i == 0:
                    sd_og = sd
                    vT_itz_og = vT_itz
                    print("skipping control")

                elif i != 0:
                    #Read in data to plot polar heat contours
                    file = '/disco/share/rm811/isca_data/' + exp[i]+ '/run0100/atmos_daily_interp.nc'
                    ds = xr.open_dataset(file)
                    heat = ds.local_heating.sel(lon=180, method='nearest').mean(dim='time')

                    print(datetime.now(), " - plotting vT")
                    plot_vT(utz, vT_itz, exp[i], heat, np.arange(-20, 190, 10), 'Blues')

                    print(datetime.now(), " - plotting s.d.")
                    NH_zonal(lat, p, sd, utz, np.arange(0, 300, 20), ulvls, 'Blues', r"v'T' SD (K m s$^{-1}$)", exp[i]+'_vTsd.pdf')

                    vT_diff = vT_itz - vT_itz_og
                    vT_sd_diff = sd - sd_og
                    plot_vT(utz, vT_diff, exp[i]+'_diff', heat, np.arange(-40, 42, 2), 'RdBu_r')
                    NH_zonal(lat, p, vT_sd_diff, utz, np.arange(-45, 50, 5), ulvls, 'RdBu_r', r"v'T' SD (K m s$^{-1}$)", exp[i]+'_vTsd_diff.pdf') 

            p = 10
            lats = slice(75, 90)
            me, mo, sd, e, sk, k = plot_pdf('vT', indir, exp, '', vpTp, p, labels, lats, r"75-90N average v'T' (K m s$^{-1}$)", colors, exp[0]+extension+'_vT')    
            plot_stats(mo, p, exp[0], extension, 'mode')
            plot_stats(sd, p, exp[0], extension, 'SD')
        
        elif plot_type == 'b':
            heat_type = input('Plot a) zonally symmetric pole-centred or b) off-pole heat?')
            if heat_type == 'a':
                polar_heat = '_w15a4p800f800g50'
                midlat_heat = '_q6m2y45l800u200'
                exp = [basis+midlat_heat, basis+polar_heat, basis+polar_heat+midlat_heat]
                label = 'polar'
            elif heat_type == 'b':
                polar_heat = '_a4x75y225w5v30p600'
                midlat_heat = '_q6m2y45'
                exp = [basis+midlat_heat+'l800u200', basis+polar_heat+'_s', basis+polar_heat+midlat_heat+'_s']
                label = 'offpole225'
           
            print(datetime.now(), " - opening files")
            vT_exp = []
            for i in range(len(exp)):
                vT = vT_calc(exp[i])
                vT_exp.append(vT)
            p = vT.pfull

            # polar cap average following Dunn-Sigouin and Shaw (2015) for meridional heat flux
            # mid-latitude average following NASA Ozone watch vT 
            lats = [60, slice(60, 90), slice(45, 75)]
            lats_labels = [r'$60\degree$N', 'polar cap', r'$45-75\degree$N']
            for i in range(len(lats)):
                compare = comparison(vT_exp, lats[i])
                linear_add(compare, p, label, lats_labels[i]) 

    elif flux == 'd':
        check_vs_MERRA('PK_e0v4z13_q6m2y45l800u200')

    elif flux =='e':
        variable = input('Plot a) n2 or b) EP Flux?')
        if variable == 'a':
            if extension == '_vtx':
                i = 0
                k = 1
                exp = [exp[0][-1], exp[1][-1]] #[exp[i][1], exp[i][3], exp[i][-1]]
                labels = [labels[-1], labels[-1]] #[labels[1], labels[3], labels[-1]]
                plot_n2_1(exp, k, basis+extension+'6') #+'_heat')
            else:
                exp = [exp[0], exp[1], exp[4], exp[-1]]
                labels = [labels[0], labels[1], labels[4], labels[-1]]
                plot_n2_2(exp, k, basis+extension)
        elif variable == 'b':
            if extension == '_vtx':
                h_name = 'w15a4p600f800g50' #exp[1][0][11:27]
                h = xr.open_dataset('../Inputs/' + h_name + '.nc')
                heat = h.mean('lon').variables[h_name]
                h_p = h.pfull
                h_lat = h.lat
                exp = [[exp[0][0], exp[0][3], exp[0][-1]], [exp[1][0], exp[1][3], exp[1][-1]]]
                #exp = [[exp[0][1], exp[0][3], exp[0][-1]], [exp[1][1], exp[1][3], exp[1][-1]]]
                labels = [labels[1], labels[3], labels[-1]]
                
                n = len(exp[0])
                utz = []
                div_response = []
                ep1_response = []
                ep2_response = []
                for i in range(n):
                    print(datetime.now(), " - opening files ({0:.0f}/{1:.0f})".format(i+1, len(exp[0])))
                    utz.append(open_data1(indir, exp[1][i])[0])
                    div_0 = open_data2(exp[0][i], "div").div
                    div_1 = open_data2(exp[1][i], "div").div
                    div_response.append(div_1 - div_0)
                    ep1_0 = open_data2(exp[0][i], "ep1").ep1
                    ep1_1 = open_data2(exp[1][i], "ep1").ep1
                    ep1_response.append(ep1_1 - ep1_0)
                    ep2_0 = open_data2(exp[0][i], "ep2").ep2
                    ep2_1 = open_data2(exp[1][i], "ep2").ep2
                    ep2_response.append(ep2_1 - ep2_0)
                plot_EP_1(utz, div_response, ep1_response, ep2_response, labels, heat, basis+extension, n)     
            else:
                #exp = [exp[0], exp[2], exp[3], exp[4]]
                #labels = [labels[0], labels[2], labels[3], labels[4]]
                div_response = []
                ep1_response = []
                ep2_response = []
                uz = []
                heat = []
                for i in range(len(exp)):
                    print(datetime.now(), " - opening files")
                    utz = open_data1(indir, exp[i])[0]
                    uz.append(utz)
                    print(datetime.now(), " - finding EP flux")
                    div = open_data2(exp[i], "div").div
                    ep1 = open_data2(exp[i], "ep1").ep1
                    ep2 = open_data2(exp[i], "ep2").ep2
                    if i == 0:
                        div_ctrl = div
                        ep1_ctrl = ep1
                        ep2_ctrl = ep2
                        plot = 'y' #input("Plot only control? y/n ")
                        if plot == 'y':
                            h_name = exp[i][11:26]
                            h = xr.open_dataset('../Inputs/' + h_name + '.nc')
                            heat_ctrl = h.sel(lon=180, method='nearest').variables[h_name]
                            h_p = h.pfull
                            h_lat = h.lat                            
                            control_plot(exp[i], labels[i], utz, div_ctrl, ep1_ctrl, ep2_ctrl, heat_ctrl, exp[i])
                    #elif i > 0:
                    #    print(datetime.now(), " - taking differences")
                    #    div_response.append(div - div_ctrl)
                    #    ep1_response.append(ep1 - ep1_ctrl)
                    #    ep2_response.append(ep2 - ep2_ctrl)
                    #    h_name = exp[i][11:27]
                    #    h = xr.open_dataset('../Inputs/' + h_name + '.nc')
                    #    heat.append(h.mean('lon').variables[h_name])
                #h_p = h.pfull
                #h_lat = h.lat
                #plot_EP_2(uz, div_ctrl, ep1_ctrl, ep2_ctrl, div_response, ep1_response, ep2_response, heat, basis+extension)


"""
# Following commented functions/code is for checking against Neil Lewis' code
def get_pt(t, p, Rd=287., cp=1005., p0=1000.): 
    #Neil's code
    return t * (p0/p)**(Rd/cp)

def TEM(ds, om=7.29e-5, a=6.371e6): 
    #Neil's code
    u = ds.ucomp 
    v = ds.vcomp 
    w = ds.omega 
    t = ds.temp 
    p = ds.pfull*100.
    latr = np.deg2rad(ds.lat)
    pt = get_pt(t, p, p0=1.e5)
    
    coslat = np.cos(latr)
    f = 2 * om * np.sin(latr)
    
    ub = u.mean('lon')
    vb = v.mean('lon')
    wb = w.mean('lon')
    ptb = pt.mean('lon')
    
    dub_dp = ub.differentiate('pfull', edge_order=2) / 100. # hPa -> Pa
    dptb_dp = ptb.differentiate('pfull', edge_order=2) / 100.
    
    up = u - ub 
    vp = v - vb
    wp = w - wb
    ptp = pt - ptb
    
    psi = (vp*ptp).mean('lon') / dptb_dp
    dpsi_dp = psi.differentiate('pfull', edge_order=2) / 100.
    
    F_lat =  (-(up*vp).mean('lon') + psi*dub_dp) * a * coslat
    F_p =  (-(up*wp).mean('lon') - psi * ((ub*coslat).differentiate('lat',edge_order=2)*180/np.pi / (a*coslat) - f)) * a * coslat
    
    v_star = vb - dpsi_dp 
    w_star = wb + (psi*coslat).differentiate('lat',edge_order=2)*180/np.pi / (a*coslat)
    
    return F_lat, F_p, v_star, w_star

def divF(F_lat, F_p, lat):
    a = 6.371e6 #earth radius m
    coslat = np.cos(lat)
    dF_lat = (coslat * F_lat).differentiate('lat',edge_order=2) * 1/(a*coslat) * (180./np.pi)
    dF_p = F_p.differentiate('pfull',edge_order=2) / 100.
    divF = dF_lat + dF_p
    return divF * 1/(a*coslat) * 1e5

def neil_plot(ds, p, lat):
    F_lat, F_p, v_star, w_star = TEM(ds)
    div_F = divF(F_lat, F_p, np.deg2rad(lat))
    div_F_mean = div_F.mean('time').transpose()

    fig, ax = plt.subplots(figsize=(6,6), constrained_layout=True)
    cs = plt.contourf(lat, p, div_F_mean, levels=np.arange(-15, 16, 1), cmap='RdBu_r', add_colorbar=False)
    cb = plt.colorbar(cs)
    plt.yscale('log')
    plt.ylim(max(p), 10) #to 10 hPa
    plt.xlim(0, 90)
    plt.xlabel(r'Latitude ($\degree$N)', fontsize='xx-large')
    plt.ylabel('Pressure (hPa)', fontsize='xx-large')
    plt.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
    plt.savefig('Neils_EPflux.pdf', bbox_inches = 'tight')
    return plt.close()
    
def aostools_plot(ds, p, lat):
    u = ds.ucomp
    v = ds.vcomp
    t = ds.temp
    w = ds.omega/100
    div, ep1, ep2 = calc_ep(u, v, w, t)

    fig, ax = plt.subplots(figsize=(6,6), constrained_layout=True)
    cs = plt.contourf(lat, p, div, levels=np.arange(-15, 16, 1), cmap='RdBu_r', add_colorbar=False)
    cb = plt.colorbar(cs)
    plt.yscale('log')
    plt.ylim(max(p), 10) #to 10 hPa
    plt.xlim(0, 90)
    plt.xlabel(r'Latitude ($\degree$N)', fontsize='xx-large')
    plt.ylabel('Pressure (hPa)', fontsize='xx-large')
    plt.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
    plt.savefig('aostools_EPflux.pdf', bbox_inches = 'tight')
    return plt.close()

#Compare Neil and aostools' code
file = '/scratch/rm811/isca_data/PK_e0v4z13_q6m2y45l800u200/run0001/atmos_daily.nc'
ds = xr.open_dataset(file, decode_times = False)
p = ds.coords['pfull']
lat = ds.coords['lat']
neil_plot(ds, p, lat)
aostools_plot(ds, p, lat)
"""