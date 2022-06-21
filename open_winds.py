"""
Script for functions involving winds - near-surface, tropospheric jet and stratospheric polar vortex.
"""

from glob import glob
import xarray as xr
import numpy as np
from datetime import datetime

def save_file(exp, var, input):
    textfile = open('../Files/'+exp+'_'+input+'.txt', 'w')
    if isinstance(var, list):
        l = var
    else:
        l = var.to_numpy().tolist()
    for j in l:
        textfile.write(str(j) + '\n')
    return textfile.close()

def open_file(exp, input):
    textfile = open('../Files/'+exp+'_'+input+'.txt', 'r')
    list = textfile.read().replace('\n', ' ').split(' ')
    list = list[:len(list)-1]
    textfile.close()
    list = np.asarray([float(j) for j in list])
    return list

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

def winds_errs(indir, exp, p, name):
    """
    Uses jet_locator functions to find location and strength of the tropospheric jet (850 hPa) or SPV (10 hPa).
    """
    print(datetime.now(), " - calculating standard deviation errors")
    maxlats = []
    maxwinds = []
    maxlats_sd = []
    maxwinds_sd = []
    for i in range(len(exp)):
        ds = xr.open_dataset(indir+exp[i]+'_zmean.nc', decode_times=False)
        lat = ds.coords['lat'].data
        u = ds.ucomp
        n = len(ds.time)
        lat, max = jet_timeseries(u, lat, p)
        maxlats.append(np.mean(lat))
        maxwinds.append(np.mean(max))
        maxlats_sd.append(np.std(lat)/np.sqrt(n))
        maxwinds_sd.append(np.std(max)/np.sqrt(n))
    save_file(name, maxlats, 'maxlats')
    save_file(name, maxwinds, 'maxwinds')
    save_file(name, maxlats_sd, 'maxlats_sd')
    save_file(name, maxwinds_sd, 'maxwinds_sd')

def find_SPV(indir, exp):
    """
    Steps through each dataset to find vortex strength over time.
    Uses 60N and 10hPa as per SSW definiton.
    Saves as a file.
    """
    print(datetime.now(), " - finding wind speeds at 60N, 10hPa")
    for i in range(len(exp)):
        print(datetime.now(), " - ", exp[i])
        SPV = xr.open_dataset(indir+exp[i]+'_zmean.nc', decode_times=False).ucomp.sel(pfull=10, method='nearest').sel(lat=60, method='nearest')
        save_file(exp[i], SPV, 'SPV')

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

def find_SSWs(exp):
    h_list = []
    h_err_list = []
    for i in range(len(exp)):
        SPV = open_file(exp[i], 'SPV')
        h, h_err = find_SSW(SPV)
        h_list.append(h)
        h_err_list.append(h_err)
    return h_list, h_err_list

def SSWsperrun(exp):
    SPV = open_file(exp, 'SPV')
    years = np.arange(5, int(len(SPV)/360)+5, 5)
    SSWs = []
    errors = []
    for n in years:
        h, h_err = find_SSW(SPV[:int(n*360)])
        SSWs.append(h)
        errors.append(h_err)
    return SSWs, errors, years

if __name__ == '__main__': 
    #Set-up data to be read in
    indir = '/disco/share/rm811/processed/'
    basis = 'PK_e0v4z13'
    exp = ['test', 'test'] #[basis, basis+'_q6m2y45l800u200']#,\
        #basis+'_w10a4p800f800g50_q6m2y45l800u200',\
        #basis+'_w15a4p800f800g50_q6m2y45l800u200',\
        #basis+'_w20a4p800f800g50_q6m2y45l800u200',\
        #basis+'_w25a4p800f800g50_q6m2y45l800u200',\
        #basis+'_w30a4p800f800g50_q6m2y45l800u200',\
        #basis+'_w35a4p800f800g50_q6m2y45l800u200',\
        #basis+'_w40a4p800f800g50_q6m2y45l800u200']
    #exp2 = [basis+'_q6m2y45l800u200',\
        #basis+'_w15a4p900f800g50_q6m2y45l800u200',\
        #basis+'_w15a4p800f800g50_q6m2y45l800u200',\
        #basis+'_w15a4p700f800g50_q6m2y45l800u200',\
        #basis+'_w15a4p600f800g50_q6m2y45l800u200',\
        #basis+'_w15a4p500f800g50_q6m2y45l800u200',\
        #basis+'_w15a4p400f800g50_q6m2y45l800u200',\
        #basis+'_w15a4p300f800g50_q6m2y45l800u200']

    winds_errs(indir, exp, 10, exp[0])
    find_SPV(indir, exp)