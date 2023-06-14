"""
Script for functions involving winds - near-surface, tropospheric jet and stratospheric polar vortex.
"""

from glob import glob
import xarray as xr
import numpy as np
from datetime import datetime
from shared_functions import *

def calc_jet_lat_quad(u, lat, p):
    """
    Function for finding location and strength of maximum given zonal wind u(lat) field.
    Based on Will Seviour code.
    """
    # Restrict to 3 points around maximum
    u_new = u.sel(pfull=p, method='nearest')
    u_max = np.where(u_new == np.ma.max(u_new))[0][0]
    u_near = u_new[u_max-1:u_max+2]
    lats_near = lat[u_max-1:u_max+2]
    # Quadratic fit, with smaller lat spacing
    coefs = np.ma.polyfit(lats_near,u_near,2)
    fine_lats = np.linspace(lats_near[0], lats_near[-1],200)
    quad = coefs[2]+coefs[1]*fine_lats+coefs[0]*fine_lats**2
    # Find jet lat and max
    jet_lat = fine_lats[np.where(quad == max(quad))[0][0]]
    jet_max = coefs[2]+coefs[1]*jet_lat+coefs[0]*jet_lat**2
    return jet_lat, jet_max

def jet_timeseries(u, lat, p):
    """
    Steps through each dataset to find jet latitude/maximum over time.
    Amended to only look at NH tropospheric jet.
    """
    print(datetime.now(), " - finding jet maxima over time")
    jet_maxima = []
    jet_lats = []
    # restrict to NH:
    u = u[:,:,int(len(lat)/2):len(lat)]
    lat = lat[int(len(lat)/2):len(lat)] 
    for i in range(len(u)):
        # find and store jet maxima and latitudes for each month
        jet_lat, jet_max = calc_jet_lat_quad(u[i], lat, p)
        jet_maxima.append(jet_max)
        jet_lats.append(jet_lat)

    return jet_lats, jet_maxima

def winds_errs(indir, outdir, exp, p, name):
    """
    Uses jet_locator functions to find location and strength of the tropospheric jet (850 hPa) or SPV (10 hPa).
    """
    maxlats = []
    maxwinds = []
    maxlats_sd = []
    maxwinds_sd = []
    print(datetime.now(), " - calculating standard deviation errors")
    for i in range(len(exp)):
        ds = xr.open_dataset(indir+exp[i]+'_uz.nc', decode_times=False)
        lat = ds.coords['lat'].data
        u = ds.ucomp
        n = len(ds.time)
        lat, max = jet_timeseries(u, lat, p)
        maxlats.append(np.mean(lat))
        maxwinds.append(np.mean(max))
        maxlats_sd.append(np.std(lat)/np.sqrt(n))
        maxwinds_sd.append(np.std(max)/np.sqrt(n))
    save_file(outdir, name, maxlats, 'maxlats'+str(p))
    save_file(outdir, name, maxwinds, 'maxwinds'+str(p))
    save_file(outdir, name, maxlats_sd, 'maxlats_sd'+str(p))
    save_file(outdir, name, maxwinds_sd, 'maxwinds_sd'+str(p))

def find_EDJ(indir, exp):
    """
    Finds EDJ strength and location over time.
    Based on Fig 1. in Waugh et al. (2018), EDJ is defined as max. winds at 850 hPa.
    """
    print(datetime.now(), " - finding EDJ for ", exp)
    u = xr.open_dataset(indir+exp+'_uz.nc', decode_times=False).ucomp.sel(pfull=850, method='nearest').sel(lat=slice(0,90))
    n = len(u)
    edjs = []
    edj_lats = []
    for j in range(n):
        u_max = np.max(u[j])
        edjs.append(u_max)
        edj_lats.append(u.lat[np.where(u[j] == u_max)])
    edj = np.mean(edjs)
    edj_err = np.std(edjs)/np.sqrt(n)
    edj_lat = np.mean(edj_lats)
    edj_lat_err = np.std(edj_lats)/np.sqrt(n)
    
    """
    # OR ...
    # Better to find max of mean, or mean of maxs? This is the former:
    u_mean = xr.open_dataset(indir+exp+'_utz.nc', decode_times=False).ucomp.sel(pfull=850, method='nearest').sel(lat=slice(0,90))[0]
    u_mean_max = np.max(u_mean)
    u_mean_lat = u_mean.lat[np.where(u_mean == u_mean_max)]
    """

    return edj, edj_lat, edj_err, edj_lat_err
    
def find_STJ(indir, exp):
    """
    Finds STJ strength and location over time.
    Based on Fig 1. in Waugh et al. (2018), STJ is max of (mean of [100 - 400 hPa winds] - 850 hPa winds)
    """
    print(datetime.now(), " - finding STJ for ", exp)
    u850 = xr.open_dataset(indir+exp+'_uz.nc', decode_times=False).ucomp.sel(pfull=850, method='nearest').sel(lat=slice(0,90))
    u100_400 = xr.open_dataset(indir+exp+'_uz.nc', decode_times=False).ucomp.sel(pfull=slice(100,400)).sel(lat=slice(0,90)).mean(dim='pfull')
    new_u = u100_400 - u850
    n = len(new_u)
    stjs = []
    stj_lats = []
    for j in range(n):
        u_max = np.max(new_u[j])
        stjs.append(u_max)
        stj_lats.append(np.mean(new_u.lat[np.where(new_u[j] == u_max)][0])) # added a mean for when max is the same across 2 consecutive lats
    stj = np.mean(stjs)
    stj_err = np.std(stjs)/np.sqrt(n)
    stj_lat = np.mean(stj_lats)
    stj_lat_err = np.std(stj_lats)/np.sqrt(n)
    
    """
    #OR ...
    # Better to find max of mean, or mean of maxs? The former is...
    u_mean850 = xr.open_dataset(indir+exp+'_utz.nc', decode_times=False).ucomp.sel(pfull=850, method='nearest').sel(lat=slice(0,90))[0]
    u_mean100_400 = xr.open_dataset(indir+exp+'_uz.nc', decode_times=False).ucomp.sel(pfull=slice(100,400)).sel(lat=slice(0,90)).mean(dim='pfull')[0]
    new_u_mean = u_mean100_400 - u_mean850
    new_u_mean_max = np.max(new_u_mean)
    new_u_mean_lat = new_u_mean.lat[np.where(new_u_mean == new_u_mean_max)]
    """

    return stj, stj_lat, stj_err, stj_lat_err

def find_jets(indir, exp):
    edj_lats = []
    edj_lat_errs = []
    stj_lats = []
    stj_lat_errs = []
    for e in exp:
        edj, edj_lat, edj_err, edj_lat_err = find_EDJ(indir, e)
        stj, stj_lat, stj_err, stj_lat_err = find_STJ(indir, e)
        edj_lats.append(edj_lat)
        edj_lat_errs.append(edj_lat_err)
        stj_lats.append(stj_lat)
        stj_lat_errs.append(stj_lat_err)
    return edj_lats, edj_lat_errs, stj_lats, stj_lat_errs

def find_SPV(indir, outdir, exp):
    """
    Steps through each dataset to find vortex strength over time.
    Uses 60N and 10hPa as per SSW definiton.
    Saves as a file.
    """
    print(datetime.now(), " - finding wind speeds at 60N, 10 and 100 hPa")
    for i in range(len(exp)):
        print(datetime.now(), " - ", exp[i])
        u = xr.open_dataset(indir+exp[i]+'_uz.nc', decode_times=False).ucomp.sel(lat=60, method='nearest')
        save_file(outdir, exp[i], u.sel(pfull=10, method='nearest'), 'u10')
        save_file(outdir, exp[i], u.sel(pfull=100, method='nearest'), 'u100')

def calc_error(nevents, nyears):
    """
    For the SSW frequency finder from Will Seviour
    Returns the 95% error interval assuming a binomial distribution:
    e.g. http://www.sigmazone.com/binomial_confidence_interval.htm
    """
    p = nevents / float(nyears)
    e = 1.96 * np.sqrt(p * (1 - p) / nyears)
    return e

def find_SSW(SPV):
    #finding SSWs
    print(datetime.now(), " - finding SSWs")
    SPV_flag = np.select([SPV<0, SPV>0], [True, False], True)
    days = len(SPV)
    count = 0
    for k in range(days):
        if SPV[k] < 0:
            if SPV[k-1] > 0:
                subset = SPV_flag[k-20:k]
                if True not in subset:
                    count += 1
    #winters = (12/4) * (days/360)
    #SSWs_w = (count/winters)
    #SSWs_w_err = calc_error(count, winters)
    SSWs_h = (count/days)*100
    SSWs_h_err = calc_error(count, days)*100
    return SSWs_h, SSWs_h_err

def find_SSWs(dir, exp):
    h_list = []
    h_err_list = []
    for i in range(len(exp)):
        SPV = open_file(dir, exp[i], 'u10')
        h, h_err = find_SSW(SPV)
        h_list.append(h)
        h_err_list.append(h_err)
    return h_list, h_err_list

def identify_SSWs(outdir, exp):
    # Simply finds dates that SSWs occur by finding indices of SSW days in the timeseries
    print(datetime.now(), " - finding SSW dates")
    SPV = open_file(outdir, exp, 'u10')
    SPV_flag = np.select([SPV<0, SPV>0], [True, False], True)
    indices = []
    for k in range(len(SPV)):
        if SPV[k] < 0:
            if SPV[k-1] > 0:
                subset = SPV_flag[k-20:k]
                if True not in subset:
                    indices.append(k)
    return(indices)

def SSWsperrun(dir, exp):
    SPV = open_file(dir, exp, 'u10')
    years = np.arange(5, int(len(SPV)/360)+5, 5)
    SSWs = []
    errors = []
    for n in years:
        h, h_err = find_SSW(SPV[:int(n*360)])
        SSWs.append(h)
        errors.append(h_err)
    return SSWs, errors, years

def Rossby(U, L): #(U, lat, L):
    O = 2 * np.pi / 86400
    Ro = U / (2 * O * L) #Ro = U / (2 * O * 2 * np.sin(lat) * L)
    return Ro

def calc_Ro(indir, exp, p):
    u = xr.open_dataset(indir+exp+'_utz.nc', decode_times=False).ucomp[0].sel(pfull=p, method='nearest')
    lat_max = 90
    lat_min = 60
    lat_range = np.arange(lat_min, lat_max, 1)
    L = len(lat_range) * 6.371e6
    u_sub = u.sel(lat=slice(lat_min, lat_max)).data
    U = np.max(u_sub)
    print(U)
    Ro = Rossby(U, L)
    return Ro
        
if __name__ == '__main__': 
    #Set-up data to be read in
    indir = '/disco/share/rm811/processed/'
    outdir = '../Files/'
    basis = 'PK_e0v4z13'
    var_type = 0 #input("Run a) depth, b) width, c) location, d) strength or e) vortex experiments?")
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
    #exp = return_exp(extension)[0]
    exp = ['PK_e0v4z13_a4x45y90w5v15p800_q6m2y45_s', 'PK_e0v4z13_a4x45y180w5v15p800_q6m2y45_s', 'PK_e0v4z13_a4x45y90w5v15p800_s', 'PK_e0v4z13_a4x45y180w5v15p800_s']

    #Ro = []
    #for i in range(len(exp)):
        #Ro.append(calc_Ro(indir, exp[i], p))
    #print(Ro)

    extension = '_offpole'
    find_SPV(indir, outdir, exp)
    p = 850
    winds_errs(indir, outdir, exp, p, basis+extension)
    p = 10
    winds_errs(indir, outdir, exp, p, basis+extension)