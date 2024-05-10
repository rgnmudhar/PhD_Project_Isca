"""
A selection of functions used in the analysis of output Isca data.
"""

import os
from glob import glob
import xarray as xr
import numpy as np
import scipy.stats as sps
import matplotlib
from pylab import *
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
        #labels = ['control', '900', '800', '700', '600', '500', '400', '300']
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
        #labels = ['control', '0.5', '1', '2', '4', '8']
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
        #labels = ['1', '2', '3', '4', '5', '6']
        xlabel = r'$\gamma$ (K km$^{-1}$)'
        exp = [exp1, exp2]
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
        perturb = '_q6m2y45l800u200'
        exp = [basis, basis+perturb]
        labels = ['P-K only', 'control']
        xlabel = 'Experiment'
        test_type = input('a) vertical resolution, b) horizontal resolution, c) alternative heat perturb or d) gamma = 5? ')
        if test_type == 'a':
            exp = [basis+perturb+'_L60', basis+'_w15a4p600f800g50'+perturb+'_L60', basis+'_w15a4p300f800g50'+perturb+'_L60']
            labels = [r'd) L60'+'\n'+r'    $\gamma = 4$', r'$p_{top}=600$', r'$p_{top}=300$']
            xlabel = 'Vertical Resolution'
        elif test_type == 'b':
            exp = [basis+perturb+'_T85', basis+'_w15a4p600f800g50'+perturb+'_T85', basis+'_w15a4p300f800g50'+perturb+'_T85']
            labels = [r'd) T85'+'\n'+r'    $\gamma = 4$', r'$p_{top}=600$', r'$p_{top}=300$']
            xlabel = 'Horizontal Resolution'
        elif test_type == 'c':
            exp = [basis+'_q6m2y45l800u300_s', basis+'_w15a4p600f800g50_q6m2y45u300_s', basis+'_w15a4p300f800g50_q6m2y45u300_s']
            labels = [r'd) $p_{midlat}^t = 300$'+'\n'+r'    $\gamma = 4$' , r'$p_{top} = 600$', r'$p_{top} = 300$']
            xlabel = 'Experiment'
        elif test_type == 'd':
            exp = ['PK_e0v5z13'+perturb, 'PK_e0v5z13'+'_w15a4p600f800g50'+perturb, 'PK_e0v5z13'+'_w15a4p300f800g50'+perturb]
            labels = [r'a) $p_{midlat}^t = 200$'+'\n'+r'    $\gamma = 5$', r'$p_{top} = 600$', r'$p_{top} = 300$']
            xlabel = 'Experiment'
        #exp = [exp1, exp2]
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

def check_levels():
    """
    Plots levels vs. pressure for different vertical resolution runs.
    """
    indir = '/disco/share/rm811/isca_data/'
    exp_L40 = 'PK_e0v4z13_q6m2y45l800u200'
    exp_L60 = 'PK_e0v4z13_q6m2y45l800u200_L60'
    file_L40 = glob(indir+exp_L40+'/run0025/*interp.nc')[0]
    file_L60 = glob(indir+exp_L60+'/run0025/*interp.nc')[0]
    ds_L40 = xr.open_dataset(file_L40, decode_times=False)
    ds_L60 = xr.open_dataset(file_L60, decode_times=False)
    p_L40 = ds_L40.pfull
    p_L60 = ds_L60.pfull
    z_L40 = altitude(p_L40)
    z_L60 = altitude(p_L60)
    N_strat_L40 = len(p_L40.where(p_L40<=200,drop=True))
    N_strat_L60 = len(p_L60.where(p_L60<=200,drop=True))
    print('Levels between TOA to 200 hPa for L40 = {0:.0f}, and L60 = {1:.0f}'.format(N_strat_L40, N_strat_L60))

    fig, axs = plt.subplots(1,2, figsize=(5,7), sharey=True)
    for lev in p_L40:
        axs.flat[0].axhline(lev, c='#B30000', lw=1, zorder=-1)
    for lev in p_L60:
        axs.flat[1].axhline(lev, c='#4D0099', lw=1, zorder=-1)
    for ax in axs:
        ax.set_ylim(1000,1e-2)
        ax.set_yscale('log')
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')    
    axs.flat[0].spines['right'].set_visible(False)
    axs.flat[1].spines['left'].set_visible(False)
    axs.flat[0].set_ylabel('Pressure (hPa)', fontsize='xx-large')
    axs.flat[0].set_xlabel('40 Levels', fontsize='xx-large', color='#B30000')
    axs.flat[1].set_xlabel('60 Levels', fontsize='xx-large', color='#4D0099')
    axs.flat[1].yaxis.tick_right()
    plt.savefig('L40_vs_L60 v2.pdf', bbox_inches = 'tight')
    return plt.close()

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

def zero_crossing(x, y):
    """
    From: https://www.ncl.ucar.edu/Document/Functions/Built-in/trop_wmo.shtml - WMO (1992): International meteorological vocabulary:
    The first tropopause is defined as the lowest level at which
    the lapse rate decreases to 2 deg K per kilometer or less,
    provided also the average lapse rate between this level and
    all higher levels within 2 kilometers does not exceed 2 deg K.
    """
    zero_level_idxs = []
    for i in range(len(x)):
        if i > 0:
            if x[i] > 0 and x[i] < x[i-1] and x[i+1] < 0:
                idx_checkdtdz = np.where(y==sorted(y, key=lambda l: abs((y[i]+2) - l))[0])[0][0] # are lapse rates at higher levels < 2 K/km?
                for x_sub_sub in x[i:idx_checkdtdz]: # rolling window
                    if x_sub_sub < 2:
                        zero_level_idxs.append(i)
    zero_level_idx = zero_level_idxs[0]                    
    target_x = 0
    if x[zero_level_idx] < 0:
        m = (y[zero_level_idx - 1] - y[zero_level_idx]) / (x[zero_level_idx - 1] - x[zero_level_idx])
        target_y = (target_x - x[zero_level_idx]) * m + y[zero_level_idx]
    else:
        m = (y[zero_level_idx] - y[zero_level_idx+1]) / (x[zero_level_idx] - x[zero_level_idx+1])
        target_y = (target_x - x[zero_level_idx]) * m + y[zero_level_idx]
    #plt.plot(x, y)
    #plt.scatter(target_x, target_y, c='r')
    #plt.axvline(0, c='k')
    #plt.show()
    return target_y

def tropopause(dir, exp):
    # Finds tropopause
    T = xr.open_dataset(dir+exp+'_Ttz.nc', decode_times=False).temp[0]
    T_sort = T.transpose().reindex(pfull=list(reversed(T.pfull)))
    p = T.pfull
    lat = T.lat
    z = list(reversed(altitude(p).data))
    dtdz = []
    for i in range(len(T_sort)):
        dtdz.append(np.diff(T_sort[i])/np.diff(z))

    z_new = z[1:]
    tropo = []
    tropo_z = []
    for j in range(len(dtdz)):
        # Find where lapse rate reaches < 2 K/km (in hPa)
        dtdz_new = dtdz[j]
        condition = np.abs(dtdz_new) - 2
        target = zero_crossing(condition, z_new)
        target_p = inv_altitude(target)  # convert back to pressure
        tropo.append(target_p)
        tropo_z.append(target)
    
    #fig, ax = plt.subplots(figsize=(7.5,7))
    #plt.plot(lat, tropo_z, linewidth=1.25, color='k', linestyle='--')
    #plt.xticks([-80, -60, -40, -20, 0, 20, 40, 60, 80], ['-80', '-60', '-40', '-20','0', '20', '40', '60', '80'])
    #plt.xlabel(r'Latitude ($\degree$N)', fontsize='xx-large')
    #plt.ylabel('Approximate Tropopause Height (km)', fontsize='xx-large')
    #plt.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
    #plt.savefig(exp+'_tropopause_km.pdf', bbox_inches = 'tight')
    #plt.show()

    return p, lat, tropo

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

def find_colors(colormap, N):
    # https://stackoverflow.com/questions/33596491/extract-matplotlib-colormap-in-hex-format
    cmap = cm.get_cmap(colormap, N) # e.g. ('RdBu', 6)
    cmap_colours = []
    for i in range(cmap.N):
        rgba = cmap(i)
        # rgb2hex accepts rgb or rgba
        cmap_colours.append('{0}'.format(matplotlib.colors.rgb2hex(rgba)))
    return cmap_colours