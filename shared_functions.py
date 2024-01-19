"""
A selection of functions used in the analysis of output Isca data.
"""

import os
from glob import glob
import xarray as xr
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
from fluxes import *
from datetime import datetime

def return_exp(extension):
    basis = 'PK_e0v4z13'
    perturb = '_q6m2y45l800u200' 
    heat = '_w15a4p600f800g50'
    if extension == '_ctrl':
        exp = [basis+perturb]
        labels = ['ctrl']
        xlabel = 'ctrl'
    if extension == '_depth':
        exp = [basis+perturb,\
        basis+'_w15a4p900f800g50'+perturb,\
        basis+'_w15a4p800f800g50'+perturb,\
        basis+'_w15a4p700f800g50'+perturb,\
        basis+'_w15a4p600f800g50'+perturb,\
        basis+'_w15a4p500f800g50'+perturb,\
        basis+'_w15a4p400f800g50'+perturb,\
        basis+'_w15a4p300f800g50'+perturb]
        labels = ['control', r'$p_{top}=900$ hPa', r'$p_{top}=800$ hPa', r'$p_{top}=700$ hPa',\
                   r'$p_{top}=600$ hPa', r'$p_{top}=500$ hPa', r'$p_{top}=400$ hPa', r'$p_{top}=300$ hPa']
        labels = ['control', r'$p_{top}=900$', r'$p_{top}=800$', r'$p_{top}=700$',\
                   r'$p_{top}=600$', r'$p_{top}=500$', r'$p_{top}=400$', r'$p_{top}=300$']
        labels = ['control', '900', '800', '700', '600', '500', '400', '300']
        xlabel = r'Depth of Heating ($p_{top}$, hPa)'
    elif extension == '_width':
        perturb = '_q6m2y45_s'
        exp = [basis+'_q6m2y45l800u200',\
        basis+'_w15a4p600f800g50'+'_q6m2y45l800u200',\
        basis+'_w20a4p600f800g50'+perturb,\
        basis+'_w25a4p600f800g50'+perturb,\
        basis+'_w30a4p600f800g50'+perturb,\
        basis+'_w35a4p600f800g50'+perturb]
        labels = ['control', r'15$\degree$', r'20$\degree$', r'25$\degree$', r'30$\degree$', r'35$\degree$']
        labels = ['control', '15', '20', '25', '30', '35']
        xlabel = r'Extent of Heating ($\degree$)'
    elif extension == '_strength':
        perturb = '_q6m2y45'
        exp = [basis+perturb+'l800u200',\
        basis+'_w15a0p600f800g50'+perturb,\
        basis+'_w15a1p600f800g50'+perturb,\
        basis+'_w15a2p600f800g50'+perturb,\
        basis+'_w15a4p600f800g50'+perturb+'l800u200',\
        basis+'_w15a8p600f800g50'+perturb]
        labels = ['control', r'$A=0.5$ K day$^{-1}$', r'$A=1$ K day$^{-1}$', r'$A=2$ K day$^{-1}$',\
                   r'$A=4$ K day$^{-1}$', r'$A=8$ K day$^{-1}$']
        labels = ['control', r'$A=0.5$', r'$A=1$', r'$A=2$',\
                   r'$A=4$', r'$A=8$']
        labels = ['control', '0.5', '1', '2', '4', '8']
        xlabel = r'Strength of Heating ($A$, K day$^{-1}$)'
    elif extension == '_loc1':   
        perturb = '_q6m2y45_s'
        exp = [basis+'_q6m2y45l800u200',\
            basis+'_a4x75y90w5v30p600'+perturb,\
            basis+'_a4x75y135w5v30p600'+perturb,\
            basis+'_a4x75y180w5v30p600'+perturb,\
            basis+'_a4x75y225w5v30p600'+perturb]
        labels = ['control', r'$\lambda=90\degree$E', r'$\lambda=135\degree$E', r'$\lambda=180\degree$E', r'$\lambda=225\degree$E']
        #labels = ['control', '90', '135', '180', '225']
        xlabel = r'Longitude of heating ($\degree$E)'
    elif extension == '_loc2':
        exp1 = ['PK_e0v4z13_a4x45y180w5v15p800_s', 'PK_e0v4z13_a4x45y90w5v15p800_s']
        exp2 = ['PK_e0v4z13_a4x45y180w5v15p800_q6m2y45_s', 'PK_e0v4z13_a4x45y90w5v15p800_q6m2y45_s']
        labels = [r'$\lambda=180\degree$E', r'$\lambda=90\degree$E']
        xlabel = r'Longitude of heating ($\degree$E)'
        exp = [exp1, exp2]
    elif extension == '_topo':
        topo = '_h4000m2l25u65'
        exp = [basis+topo,\
        basis+'_w15a4p900f800g50'+topo,\
        basis+'_w15a4p800f800g50'+topo,\
        basis+'_w15a4p600f800g50'+topo,\
        basis+'_w15a4p300f800g50'+topo]
        labels = ['control', '900 hPa', '800 hPa', '600 hPa', '300 hPa']
        xlabel = 'Depth of Heating (hPa)'
    elif extension == '_vtx':
        perturb1 = '_q6m2y45l800u200' 
        perturb2 = '_w15a4p600f800g50_q6m2y45l800u200'
        exp1 = ['PK_e0v1z13'+perturb1,\
        'PK_e0v2z13'+perturb1,\
        'PK_e0v3z13'+perturb1,\
        'PK_e0v4z13'+perturb1,\
        'PK_e0v5z13'+perturb1,\
        'PK_e0v6z13'+perturb1]
        exp2 = ['PK_e0v1z13'+perturb2,\
        'PK_e0v2z13'+perturb2,\
        'PK_e0v3z13'+perturb2,\
        'PK_e0v4z13'+perturb2,\
        'PK_e0v5z13'+perturb2,\
        'PK_e0v6z13'+perturb2]
        labels = [r'$\gamma = 1$ K km$^{-1}$', r'$\gamma = 2$ K km$^{-1}$', r'$\gamma = 3$ K km$^{-1}$',\
                  r'$\gamma = 4$ K km$^{-1}$', r'$\gamma = 5$ K km$^{-1}$', r'$\gamma = 6$ K km$^{-1}$']
        labels = [r'$\gamma = 1$', r'$\gamma = 2$', r'$\gamma = 3$',\
                  r'$\gamma = 4$', r'$\gamma = 5$', r'$\gamma = 6$']
        labels = ['1', '2', '3', '4', '5', '6']
        xlabel = r'$\gamma$ (K km$^{-1}$)'
        exp = exp1 #[exp1, exp2]
    elif extension == '_jetfix':
        exp1 = ['PK_e0v1z13_a0b0p2'+perturb,\
            'PK_e0v1z13_a0b10p2'+perturb,\
            'PK_e0v1z13_a5b4p1'+perturb,\
            'PK_e0v1z13_a5b12p1'+perturb,\
            'PK_e0v1z13_a5b20p1'+perturb]
        exp2 = ['PK_e0v1z13_a0b0p2'+heat+perturb,\
            'PK_e0v1z13_a0b10p2'+heat+perturb,\
            'PK_e0v1z13_a5b4p1'+heat+perturb,\
            'PK_e0v1z13_a5b12p1'+heat+perturb,\
            'PK_e0v1z13_a5b20p1'+heat+perturb]
        labels = ['J30', 'J30-40', 'J40', 'J40-50', 'J50']
        xlabel = 'Experiment'
        exp = [exp1, exp2]       
    elif extension == '_test':
        basis = 'PK_e0v4z13'
        exp = [basis+perturb,\
               basis+'_q6m2y45l800u300_s']
               #basis+heat+perturb]
            #[basis+perturb+'_T85' ,\
            #basis+heat+perturb+'_T85'] #,\
        #labels = ['T42 control', 'T42 + polar heat'] #'T85 control', 'T85 + polar heat']
        #xlabel = 'Experiment' 
        labels = [r'$p_t = 200$ hPa', r'$p_t = 300$ hPa'] #, r'$p_t = 400$ hPa']
        xlabel = r'$p_t$ (hPa)' 
    return exp, labels, xlabel

def add_phalf(exp_name, file_name):
    """
    Assign phalf levels from uninterpolated to interpolated datset.
    """

    ds = xr.open_dataset(exp_name+file_name, decode_times=False)
    ds_original = xr.open_mfdataset('../atmos_daily_T42_p40.nc', decode_times=False)
    ds = ds.assign_coords({"phalf":ds_original.phalf})

    return ds

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

def calc_streamfn(v, p, lat): #KEEP
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

def altitude(p): #KEEP
    """
    Finds altitude from pressure using z = -H*log10(p/p0).
    """
    H = 7 #scale height km
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

def use_altitude(x, coord1, coord2, dim1, dim2, unit): #KEEP
    """
    Creates new DataArray that uses z in place of pfull.
    """
    x_xr = xr.DataArray(x, coords=[coord1, coord2], dims=[dim1, dim2])
    x_xr.attrs['units'] = unit
    return x_xr

def inv_altitude(z): #KEEP
    """
    Finds pressure from altitude using z = -H*log10(p/p0).
    Single value conversion
    """
    H = 7 #scale height km
    p0 = 1000 #surface pressure hPa    
      
    p = p0*np.exp((-1)*z/H)
    
    return p

def difference(a1, a2, coord1, coord2, dim1, dim2, unit): #KEEP
    """
    Take the difference between 2 datasets and create an xarray DataArray.
    """
    
    diff = a1 - a2
    
    diff_xr = xr.DataArray(diff, coords=[coord1, coord2], dims=[dim1, dim2])
    diff_xr.attrs['units'] = unit
    
    return diff_xr

def diff_variables(x, lat, p, units): #KEEP
    """
    Find difference between datasets.
    Start with zonal wind and temperature.
    """
    x_diff = difference(x[0], x[1], p, lat, 'lat', 'pfull', units)

    return x_diff

def save_file(dir, exp, var, input):
    textfile = open(dir+exp+'_'+input+'.txt', 'w')
    if isinstance(var, list):
        l = var
    else:
        l = var.to_numpy().tolist()
    for j in l:
        textfile.write(str(j) + '\n')
    return textfile.close()

def open_file(dir, exp, input):
    textfile = open(dir+exp+'_'+input+'.txt', 'r')
    list = textfile.read().replace('\n', ' ').split(' ')
    list = list[:len(list)-1]
    textfile.close()
    list = np.asarray([float(j) for j in list])
    return list

def fillnas(dir, exp):
    dir = dir+exp
    run_list = sorted(glob(dir+'/run*'))
    for i in range(len(run_list)):
        print(i)
        file = run_list[i]+'/atmos_daily_interp.nc'
        ds = xr.open_dataset(file, decode_times=False)
        ds_new = ds.fillna(0)
        ds_new.to_netcdf(file, format="NETCDF3_CLASSIC")

def min_lapse(T, z):
    # Finds where lapse rate reaches < 2 K/km (in hPa)
    dtdz = []
    for i in range(len(T)-1):
        dtdz.append( -(T[i+1] - T[i]) / (z[i+1] - z[i]) )
    for j in range(len(dtdz)):
        if dtdz[j] < 2:
            idx = (np.abs(z - (z[j]+2)).argmin())
            if dtdz[idx] < 2:
                h_tropo = z[j]
                break
    return inv_altitude(h_tropo) # convert back to pressure

def tropopause(dir, exp):
    # Finds tropopause
    T = xr.open_dataset(dir+exp+'_Ttz.nc', decode_times=False).temp[0]
    T_sort = T.transpose().reindex(pfull=list(reversed(T.pfull)))
    p = T.pfull
    z = list(reversed(altitude(p).data))
    tropo = []
    for i in range(len(T_sort)):
        tropo.append(min_lapse(T_sort[i], z))
    return p, T.lat.data, tropo

def pdf(x, plot=False):
    x = np.sort(x)
    ae, loce, scalee = sps.skewnorm.fit(x)
    p = sps.skewnorm.pdf(x, ae, loce, scalee)
    if plot==True:
        s = np.std(x)
        mean = np.mean(x)
        f = (1 / (s * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean)/s)**2)
        plt.hist(x, bins = 50, density=True)
        plt.plot(x, f)
        plt.plot(x, p)
        plt.savefig('pdf.pdf')
    mode = x[int(np.argmax(p))]
    return x, p, mode

def plot_pdf(var, dir, exp, ext, z, p, lats, labels, xlabel, colors, name):
    mode = []
    mean = []
    sd = []
    err = []
    skew = []
    kurt = []
    x_min = x_max = 0
    fig, ax = plt.subplots(figsize=(6,6))
    for i in range(len(exp)):
        if var == 'u':
            if type(lats) == int:
                x = xr.open_dataset(dir+exp[i]+ext, decode_times=False).ucomp.sel(pfull=p, method='nearest').sel(lat=lats, method='nearest')
            elif type(lats) == slice:
                x = xr.open_dataset(dir+exp[i]+ext, decode_times=False).ucomp.sel(pfull=p, method='nearest').sel(lat=lats).mean('lat')
        elif var == 'vT':
            x = vT_level(z[i], p, lats)
        elif var == 'gph':
            x = z[i]
        x_sort, f, m = pdf(x)
        mode.append(m)
        sd.append(np.std(x))
        mean.append(np.mean(x))
        err.append(np.std(x/np.sqrt(len(x))))
        skew.append(sps.skew(x))
        kurt.append(sps.kurtosis(x))
        if max(x) > x_max:
            x_max = max(x)
        if min(x) < x_min:
            x_min = min(x)
        print(datetime.now(), " - plotting ({0:.0f}/{1:.0f})".format(i+1, len(exp)))
        ax.plot(x_sort, f, linewidth=1.25, color=colors[i], label=labels[i])
    ax.axvline(0, color='k', linewidth=0.25)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(xlabel, fontsize='xx-large')
    ax.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
    plt.legend(fancybox=False, ncol=1, fontsize='x-large')
    plt.savefig(name+'_{:.0f}pdf.pdf'.format(p), bbox_inches = 'tight')
    plt.close()
    return mean, mode, sd, err, skew, kurt

def find_sd(x):
    print(datetime.now(), " - opening files")
    lat = x.coords['lat']
    p = x.coords['pfull']
    sd = np.empty_like(x[0])
    print(datetime.now(), " - finding zonal mean s.d. over latitude-pressure")
    for i in range(len(p)):
        for j in range(len(lat)):
            sd[i,j] = np.std(x[:,i,j])
    return lat, p, sd

def NH_zonal(lat, p, x, y, xlvls, ylvls, colors, lab, name):
    print(datetime.now(), " - plotting")
    fig, ax = plt.subplots(figsize=(6,6))
    cs1 = ax.contourf(lat, p, x, levels=xlvls, cmap=colors)
    ax.contourf(cs1, colors='none')
    cs2 = ax.contour(lat, p, y, colors='k', levels=ylvls, linewidths=0.5, alpha=0.2)
    cs2.collections[int(len(ylvls)/2)].set_linewidth(1)
    cb = plt.colorbar(cs1, extend='both')
    cb.set_label(label=lab, size='x-large')
    cb.ax.tick_params(labelsize='x-large')
    #plt.scatter(lat.sel(lat=60, method='nearest'), p.sel(pfull=10, method='nearest'), marker='x', color='#B30000')
    plt.xlabel(r'Latitude ($\degree$N)', fontsize='x-large')
    plt.xlim(0,90)
    plt.xticks([10, 30, 50, 70, 90], ['10', '30', '50', '70', '90'])
    plt.ylabel('Pressure (hPa)', fontsize='x-large')
    plt.ylim(max(p), 1) #goes to ~1hPa
    plt.yscale('log')
    plt.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
    plt.savefig(name, bbox_inches = 'tight')
    return plt.close()