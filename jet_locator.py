"""
Function for finding location and strength of maximum given zonal wind u(lat) field - based on WSeviour code.
Amended to only look at NH tropospheric jet.
Also includes timeseries plots of vortex strength at 60N and 10 hPa.
"""

from glob import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from shared_functions import *

def calc_jet_lat_quad(u, lats, p, plot=False):
    # Restrict to 3 points around maximum
    u_new = u.mean(dim='time').mean(dim='lon').sel(pfull=p, method='nearest')
    u_max = np.where(u_new == np.ma.max(u_new))[0][0]
    u_near = u_new[u_max-1:u_max+2]
    lats_near = lats[u_max-1:u_max+2]
    # Quadratic fit, with smaller lat spacing
    coefs = np.ma.polyfit(lats_near,u_near,2)
    fine_lats = np.linspace(lats_near[0], lats_near[-1],200)
    quad = coefs[2]+coefs[1]*fine_lats+coefs[0]*fine_lats**2
    # Find jet lat and max
    jet_lat = fine_lats[np.where(quad == max(quad))[0][0]]
    jet_max = coefs[2]+coefs[1]*jet_lat+coefs[0]*jet_lat**2
    # Plot fit?
    if plot:
        print(jet_max)
        print(jet_lat)
        plt.plot(lats_near, u_near)
        plt.plot(fine_lats, quad)
        plt.xlabel("Latitude")
        plt.ylabel("Wind Speed")
        plt.title("Jet Latitude at p={:.0f}hPa".format(p))
        plt.text(jet_lat-2, jet_max-1, "Jet max = {0:.2f}m/s at {1:.2f} deg latitude".format(jet_max, jet_lat))
        plt.show()
 
    return jet_lat, jet_max

def jet_timeseries(files, iter, p):
    """
    Steps through each dataset to find jet latitude/maximum over time.
    """
    jet_maxima = []
    jet_lats = []

    for i in iter:
        #print(i)
        file  = files[i]
        ds = xr.open_dataset(file, decode_times=False)
        lat = ds.coords['lat'].data
        u = ds.ucomp
        # restrict to NH:
        u = u[:,:,int(len(lat)/2):len(lat),:]
        lat = lat[int(len(lat)/2):len(lat)] 
        # find and store jet maxima and latitudes for each month
        jet_lat, jet_max = calc_jet_lat_quad(u, lat, p)
        jet_maxima.append(jet_max)
        jet_lats.append(jet_lat)

    return jet_lats, jet_maxima

def vtx_timeseries(files, iter):
    """
    Steps through each dataset to find vortex strength over time.
    Uses 60N and 10hPa as per SSW definiton.
    """
    p = 10 # hPa
    l = 60 # degrees north

    vtx_strength = []

    for i in iter:
        file = files[i]
        ds = xr.open_dataset(file, decode_times=False)
        for j in range(len(ds.time)):
            vtx_strength.append(ds.ucomp[j].mean(dim='lon').sel(pfull=p, method='nearest').sel(lat=l, method='nearest'))

    return vtx_strength

def plot_vtx(files1, files2, files3, files4, labels, colors, style, cols, fig_name):
    
    iter = np.arange(0,len(files1))

    vtx1 = vtx_timeseries(files1, iter)
    vtx2 = vtx_timeseries(files2, iter)
    vtx3 = vtx_timeseries(files3, iter)
    vtx4 = vtx_timeseries(files4, iter)

    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(vtx1, color=colors[0], linewidth=1, linestyle=style[0], label=labels[0])
    ax.plot(vtx2, color=colors[1], linewidth=1, linestyle=style[1], label=labels[1])
    ax.plot(vtx3, color=colors[2], linewidth=1, linestyle=style[2], label=labels[2])
    ax.plot(vtx4, color=colors[3], linewidth=1, linestyle=style[3], label=labels[3])
    ax.set_xlim(1,len(vtx1)+1)
    ax.set_xlabel('Day', fontsize='large')       
    ax.set_ylabel(r'Zonal Wind Speed (ms$^{-1}$)', color='k', fontsize='large')
    ax.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
    plt.legend(loc='upper center' , bbox_to_anchor=(0.5, -0.07), fancybox=False, shadow=True, ncol=cols, fontsize='large')
    plt.title(r'Vortex Strength at $p \sim 10$ hPa, $\theta \sim 60 \degree$N', fontsize='x-large')
    plt.savefig(fig_name+'_vtx.png', bbox_inches = 'tight')
    
    return plt.close()

def plot_jet(files1, files2, files3, files4, p, labels, colors, style, cols, fig_name):

    iter = np.arange(0,len(files1))

    lat1, max1 = jet_timeseries(files1, iter, p)
    lat2, max2 = jet_timeseries(files2, iter, p)
    lat3, max3 = jet_timeseries(files3, iter, p)
    lat4, max4 = jet_timeseries(files4, iter, p)

    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(iter+1, lat1, color=colors[0], linewidth=1, linestyle=style[0], label=labels[0])
    ax.plot(iter+1, lat2, color=colors[1], linewidth=1, linestyle=style[1], label=labels[1])
    ax.plot(iter+1, lat3, color=colors[2], linewidth=1, linestyle=style[2], label=labels[2])
    ax.plot(iter+1, lat4, color=colors[3], linewidth=1, linestyle=style[3], label=labels[3])
    ax.set_xlim(1,len(files1))
    ax.set_xlabel('Month', fontsize='large')       
    ax.set_ylabel(r'Jet Latitude ($\degree$N)', fontsize='large')
    ax.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
    plt.legend(loc='upper center' , bbox_to_anchor=(0.5, -0.07), fancybox=False, shadow=True, ncol=cols, fontsize='large')
    plt.title('NH Jet Latitude at p ~{0:.0f} hPa'.format(p), fontsize='x-large')
    plt.savefig(fig_name+'_jetlat.png', bbox_inches = 'tight')
    plt.close()


    fig2, ax2 = plt.subplots(figsize=(12,8))
    ax2.plot(iter+1, max1, color=colors[0], linewidth=1, linestyle=style[0], label=labels[0])
    ax2.plot(iter+1, max2, color=colors[1], linewidth=1, linestyle=style[1], label=labels[1])
    ax2.plot(iter+1, max3, color=colors[2], linewidth=1, linestyle=style[2], label=labels[2])
    ax2.plot(iter+1, max4, color=colors[3], linewidth=1, linestyle=style[3], label=labels[3])
    ax2.set_xlim(1,len(files1))
    ax2.set_xlabel('Month', fontsize='large')       
    ax2.set_ylabel(r'Jet Max (ms$^{-1}$)', color='k', fontsize='large')
    ax2.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
    plt.legend(loc='upper center' , bbox_to_anchor=(0.5, -0.07), fancybox=False, shadow=True, ncol=cols, fontsize='large')
    plt.title('NH Jet Strength at p ~{0:.0f} hPa'.format(p), fontsize='x-large')
    plt.savefig(fig_name+'_jetmax.png', bbox_inches = 'tight')
    
    return plt.close()

if __name__ == '__main__': 
    #Set-up data to be read in
    exp_name = ['PK_eps0_vtx4_zoz18_7y','PK_eps10_vtx4_zoz18_7y','PK_eps0_vtx4_zoz13_7y','PK_eps10_vtx4_zoz13_7y']
    time = 'daily'
    years = 0 # user sets no. of years worth of data to ignore due to spin-up
    file_suffix = '_interp'
    p = 850 # pressure level at which we want to find the jet (hPa)

    files1 = discard_spinup2(exp_name[0], time, file_suffix, years)
    files2 = discard_spinup2(exp_name[1], time, file_suffix, years)
    files3 = discard_spinup2(exp_name[2], time, file_suffix, years)
    files4 = discard_spinup2(exp_name[3], time, file_suffix, years)

    #labels = [r'$\gamma$ = 1',r'$\gamma$ = 2',r'$\gamma$ = 3',r'$\gamma$ = 4']
    #colors = ['k', '#C0392B', '#27AE60', '#9B59B6']
    #style = ['-', '-', '-', '-']
    #cols = 4
    labels = [r'$\epsilon = 0, p_{trop} \sim 100$ hPa', r'$\epsilon = 10, p_{trop} \sim 100$ hPa', r'$\epsilon = 0, p_{trop} \sim 200$ hPa', r'$\epsilon = 10, p_{trop} \sim 200$ hPa']
    colors = ['#2980B9', '#2980B9', 'k', 'k']
    style = ['--', '-', '--', '-']
    cols = 2

    #plot_jet(files1, files2, files3, files4, p, labels, colors, style, cols, 'PK_eps0+10_zoz13+18_vtx4')

    plot_vtx(files1, files2, files3, files4, labels, colors, style, cols, 'PK_eps0+10_zoz13+18_vtx4')

"""
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(iter+1, jet_maxima, color='k', linewidth=1)
ax.set_xlim(1,len(files))
ax.set_xlabel('Month', fontsize='large')       
ax.set_ylabel(r'Jet Max (ms$^{-1}$)', color='k', fontsize='large')
ax.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
ax2 = ax.twinx()
ax2.plot(iter+1, jet_lats, color='#2980B9', linewidth=1.5, linestyle='--')
ax2.set_ylabel(r'Jet Latitude ($\degree$N)', color='#2980B9', fontsize='large')
ax2.tick_params(axis='y', colors='#2980B9')
ax2.tick_params(axis='both', labelsize = 'large', which='both', direction='in')
plt.title('NH Jet at p ~{0:.0f} hPa'.format(p), fontsize='x-large')
"""