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
        labels = ['control', '900 hPa', '800 hPa', '700 hPa', '600 hPa', '500 hPa', '400 hPa', '300 hPa']
        xlabel = 'Depth of Heating (hPa)'
    elif extension == '_width':
        exp = [basis+perturb,\
        basis+'_w10a4p800f800g50'+perturb,\
        basis+'_w15a4p800f800g50'+perturb,\
        basis+'_w20a4p800f800g50'+perturb,\
        basis+'_w25a4p800f800g50'+perturb,\
        basis+'_w30a4p800f800g50'+perturb,\
        #basis+'_w35a4p800f800g50'+perturb,\
        basis+'_w35a4p800f800g50'+perturb+'_SCALED',\
        basis+'_w40a4p800f800g50'+perturb]
        labels = ['control', r'10$\degree$', r'15$\degree$', r'20$\degree$', r'25$\degree$', r'30$\degree$', r'35$\degree$', r'40$\degree$']
        xlabel = r'Extent of Heating ($\degree$)'
    elif extension == '_strength':
        level = input("For depth a) 800, b) 600, or c) 400 hPa?")
        if level == 'a':
            exp = [basis+perturb,\
            basis+'_w15a0p800f800g50'+perturb,\
            basis+'_w15a2p800f800g50'+perturb,\
            basis+'_w15a4p800f800g50'+perturb,\
            basis+'_w15a6p800f800g50'+perturb,\
            basis+'_w15a8p800f800g50'+perturb]
            labels = ['control', r'0.5 K day$^{-1}$', r'2 K day$^{-1}$', r'4 K day$^{-1}$', r'6 K day$^{-1}$', r'8 K day$^{-1}$']
        elif level == 'b':
            exp = [basis+perturb,\
            basis+'_w15a2p600f800g50'+perturb,\
            basis+'_w15a4p600f800g50'+perturb,\
            basis+'_w15a6p600f800g50'+perturb,\
            basis+'_w15a8p600f800g50'+perturb]
            labels = ['control', r'2 K day$^{-1}$', r'4 K day$^{-1}$', r'6 K day$^{-1}$', r'8 K day$^{-1}$']
        elif level == 'c':
            exp = [basis+perturb,\
            basis+'_w15a2p400f800g50'+perturb,\
            basis+'_w15a4p400f800g50'+perturb,\
            basis+'_w15a6p400f800g50'+perturb,\
            basis+'_w15a8p400f800g50'+perturb]
            labels = ['control', r'2 K day$^{-1}$', r'4 K day$^{-1}$', r'6 K day$^{-1}$', r'8 K day$^{-1}$']
        xlabel = r'Strength of Heating (K day$^{-1}$)'
    elif extension == '_loc':   
        perturb = '_q6m2y45'
        exp = [basis+perturb+'l800u200',\
            basis+'_a4x75y0w5v30p800'+perturb,\
            basis+'_a4x75y90w5v30p800'+perturb,\
            basis+'_a4x75y180w5v30p800'+perturb,\
            basis+'_a4x75y270w5v30p800'+perturb]
        labels = ['control', r'0$\degree$', r'90$\degree$', r'180$\degree$', r'270$\degree$']
        xlabel = r'Longitude of Heating ($\degree$E)'
    elif extension == '_topo':
        topo = '_h4000m2l25u65'
        exp = [basis+topo,\
        basis+'_w15a4p800f800g50'+topo,\
        basis+'_w15a4p300f800g50'+topo]
        #basis+perturb,\
        #basis+'_w15a4p800f800g50'+perturb,\
        #basis+'_w15a4p300f800g50'+perturb,\
        labels = ['control', '800 hPa', '300 hPa']
        xlabel = 'Depth of Heating (hPa)'
    elif extension == '_poster':
        exp = [basis+perturb,\
            basis+'_w15a0p800f800g50'+perturb,\
            basis+'_w15a4p900f800g50'+perturb,\
            basis+'_w10a4p800f800g50'+perturb]
        labels = ['control', 'weak', 'shallow', 'narrow']
        xlabel = ''
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

def plot_pdf(var, dir, exp, input, z, p, labels, xlabel, colors, name):
    mode = []
    mean = []
    sd = []
    err = []
    skew = []
    kurt = []
    for j in range(len(p)):
        if type(p) == int:
            p1 = p
        elif type(p) == list:
            p1 = p[j]
        print(datetime.now(), " - plotting PDFs at {:.0f} hPa".format(p1))
        x_min = x_max = 0
        sub_mode = []
        sub_mean = []
        sub_sd = []
        sub_err = []
        sub_skew = []
        sub_kurt = []
        fig, ax = plt.subplots(figsize=(6,6))
        for i in range(len(exp)):
            if var == 'u':
                x = xr.open_dataset(dir+exp[i]+input, decode_times=False).ucomp.sel(pfull=p1, method='nearest').sel(lat=60, method='nearest')
            elif var == 'vT':
                x = vT_level(z[i], p1)
            elif var == 'gph':
                x = z[i]
            x_sort, f, m = pdf(x)
            sub_mode.append(m)
            sub_sd.append(np.std(x))
            sub_mean.append(np.mean(x))
            sub_err.append(np.std(x/np.sqrt(len(x))))
            sub_skew.append(sps.skew(x))
            sub_kurt.append(sps.kurtosis(x))
            if max(x) > x_max:
                x_max = max(x)
            if min(x) < x_min:
                x_min = min(x)
            print(datetime.now(), ' - plotting')
            ax.plot(x_sort, f, linewidth=1.25, color=colors[i], label=labels[i])
        mode.append(sub_mode)
        sd.append(sub_sd)
        mean.append(sub_mean)
        err.append(sub_err)
        skew.append(sub_skew)
        kurt.append(sub_kurt)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(xlabel, fontsize='x-large')
        ax.tick_params(axis='both', labelsize = 'x-large', which='both', direction='in')
        plt.legend(fancybox=False, shadow=True, ncol=1, fontsize='large')
        plt.savefig(name+'_{:.0f}pdf.pdf'.format(p1), bbox_inches = 'tight')
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
