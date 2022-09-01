from nco import Nco
from glob import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from shared_functions import *
from datetime import datetime

def take_zonal_means(indir, outdir):
    os.chdir(indir)
    exp = sorted(glob('*'))
    os.chdir(outdir)
    for i in range(len(exp)):
        print(datetime.now(), ' - ' + exp[i] + ' zonal means')
        nco.ncwa(input = exp[i]+'_T.nc', output = exp[i]+'_Tz.nc', options = ['-a lon'])

def calc_w(dir, exp):
    print(datetime.now(), ' - ', exp)
    print(datetime.now(), ' - opening u and v')
    u = xr.open_dataset(dir+exp+'_u.nc', decode_times=False).ucomp
    v = xr.open_dataset(dir+exp+'_v.nc', decode_times=False).vcomp
    print(datetime.now(), ' - finding divergence')
    div = u.differentiate('lon', edge_order=2) / deg2rad / acoslat + (v*coslat).differentiate('lat', edge_order=2) / deg2rad / acoslat
    print(datetime.now(), ' - integrating')
    wcalc_half = -np.cumsum(div.values * (dp*100)[None,:,None,None], axis=1)
    wcalc_half = np.concatenate((np.zeros((len(u.time),1,len(u.lat),len(u.lon))), wcalc_half), axis=1)
    print(datetime.now(), ' - interpolating to full pressure levels')
    wcalc_full = (wcalc_half[:,1:] + wcalc_half[:,:-1]) / 2. 

    print(datetime.now(), ' - saving file')
    coord_list = ['time', 'pfull', 'lat', 'lon']
    wcalc_ds = xr.Dataset(
         data_vars=dict(
             wcalc_full = (coord_list, wcalc_full)
         ),
         coords=u.coords
    )
    wcalc_ds = wcalc_ds.rename({'wcalc_full' : 'omega'})
    wcalc_ds.to_netcdf(dir + exp + '_w.nc', format="NETCDF4_CLASSIC",
             encoding = {'omega': {"dtype": 'float32', '_FillValue': None},
                    "time": {'_FillValue': None}, "pfull": {'_FillValue': None},
                    "lat": {'_FillValue': None}, "lon": {'_FillValue': None}}
                )

def calc_TKE(u,v):
    upv = u*u + v*v
    return 0.5 * upv

def find_TKE(dir, exp):
    """
    This attempts to use TKE for determining spin-up.
    For every 'time'  (i.e. atmos_monthly file) take the zonal mean of u and v along longitude.
    Then at each (lat,p) find 0.5*(u^2 + v^2).
    Then take a weighted average along lat, and finally a mean along pressure.
    Based on script by Penelope Maher and discussions with William Seviour.
    """
    KE = []
    print("finding KE")
    uz = xr.open_dataset(dir+exp+'_uz.nc', decode_times=False).ucomp[0]
    vz = xr.open_dataset(dir+exp+'_vz.nc', decode_times=False).vcomp[0]
    coslat = np.cos(np.deg2rad(uz.coords['lat'].values)).clip(0., 1.) # need to weight due to different box sizes over grid
    lat_wgts = np.sqrt(coslat)
    for j in range(len(uz)):
        TKE_box = np.empty_like(uz[j])
        for q in range(len(uz.coords['pfull'].data)):
            for k in range(len(uz.coords['lat'].data)):
                TKE_box[q,k] = calc_TKE(uz[j][q,k], vz[j][q,k])
        TKE_box = np.average(TKE_box, axis=1, weights=lat_wgts)
        TKE_avg = np.nanmean(TKE_box) # should I weight pressures too? How?
        KE.append(TKE_avg)
    save_file(exp, KE, 'KE')

    print("plotting")
    KE = open_file(exp, 'KE')
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(len(KE), KE, color='k')
    ax.set_xlim(1,len(KE))
    ax.set_xlabel("Days Simulated")       
    ax.set_ylabel('total KE', color='k')
    plt.savefig(exp+'_spinup.pdf', bbox_inches = 'tight')
    return plt.close()

def postprocess(exp):
    print(datetime.now(), ' - ', exp)
    # run plevel interpolation
    print(datetime.now(), ' - interpolate')
    os.chdir(indir + exp)
    runs = sorted(glob('run*'))
    n = len(runs) - 1
    # PLEASE NOTE WITH THE FOLLOWING EXCLUDED, RUN_PLEVEL.PY MUST BE RUN BEFORE THIS SCRIPT
    #os.chdir(plevdir)
    #run_plevel.main(indir, exp, n)

    # delete pre-interpolation files
    os.chdir(indir + exp)
    files = glob('*/atmos_daily.nc', recursive=True)
    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print(e.strerror, ':', f)

    # ignore X years of spin-up
    print(datetime.now(), ' - ignore spin-up')
    X = 2
    n = 0
    while n < (X * 12):
        os.rename(runs[n], 'spinup'+str(n))
        n += 1
    
    # concatenate all
    print(datetime.now(), ' - concatenate')
    nco.ncrcat(input = 'run*/*interp.nc', output = exp+'_all.nc', use_shell = True)

    # extract variables
    print(datetime.now(), ' - extract variables')
    all = exp+'_all.nc'
    nco.ncks(input = all, output = exp+'_u.nc', options = ['-v ucomp'])
    nco.ncks(input = all, output = exp+'_v.nc', options = ['-v vcomp'])
    nco.ncks(input = all, output = exp+'_w.nc', options = ['-v omega'])
    nco.ncks(input = all, output = exp+'_T.nc', options = ['-v temp'])
    nco.ncks(input = all, output = exp+'_h.nc', options = ['-v height'])

    # time means
    print(datetime.now(), ' - time means')
    os.chdir(indir + exp)
    nco.ncra(input = exp+'_h.nc', output = exp+'_ht.nc')
    nco.ncra(input = exp+'_u.nc', output = exp+'_ut.nc')
    nco.ncra(input = exp+'_v.nc', output = exp+'_vt.nc')
    nco.ncra(input = exp+'_T.nc', output = exp+'_Tt.nc')

    # time means
    print(datetime.now(), ' - time and zonal means')
    nco.ncwa(input = exp+'_ht.nc', output = exp+'_htz.nc', options = ['-a lon'])
    nco.ncwa(input = exp+'_ut.nc', output = exp+'_utz.nc', options = ['-a lon'])
    nco.ncwa(input = exp+'_vt.nc', output = exp+'_vtz.nc', options = ['-a lon'])
    nco.ncwa(input = exp+'_Tt.nc', output = exp+'_Ttz.nc', options = ['-a lon'])

    # zonal means
    # NOTE THAT THE FOLLOWING ONLY SEEMS TO WORK ON GV3 OR GV4
    #print(datetime.now(), ' - zonal means')
    nco.ncwa(input = exp+'_u.nc', output = exp+'_uz.nc', options = ['-a lon'])
    nco.ncwa(input = exp+'_v.nc', output = exp+'_vz.nc', options = ['-a lon'])
    nco.ncwa(input = exp+'_T.nc', output = exp+'_Tz.nc', options = ['-a lon'])

    # remove file with all included
    # move created files to folder for processed data
    print(datetime.now(), ' - re-arrange files')
    os.chdir(indir + exp)
    os.remove(exp+'_all.nc')
    os.remove(exp+'_h.nc')
    os.remove(exp+'_ut.nc')
    os.remove(exp+'_vt.nc')
    os.remove(exp+'_Tt.nc')
    newfiles = glob('*.nc', recursive=True)
    for f in newfiles:
        try:
            os.rename(f, outdir+f)
        except OSError as e:
            print(e.strerror, ':', f)

if __name__ == '__main__': 
    indir = '/disco/share/rm811/isca_data/'
    outdir = '/disco/share/rm811/processed/'
    plevdir = '/home/links/rm811/Isca/postprocessing/plevel_interpolation/scripts'
    analysisdir = '/home/links/rm811/scratch/PhD_Project_Isca'

    nco = Nco(debug=True)

    #sys.path.append(os.path.abspath(plevdir))
    #import run_plevel 

    exp = ['PK_e0v4z13_q6m2y45l800u200',\
    'PK_e0v4z13_w15a4p900f800g50_q6m2y45l800u200',\
    'PK_e0v4z13_w15a4p700f800g50_q6m2y45l800u200',\
    'PK_e0v4z13_w15a4p500f800g50_q6m2y45l800u200',\
    'PK_e0v4z13_w15a4p300f800g50_q6m2y45l800u200',\
    'PK_e0v4z13_w10a4p800f800g50_q6m2y45l800u200',\
    'PK_e0v4z13_w20a4p800f800g50_q6m2y45l800u200',\
    'PK_e0v4z13_w30a4p800f800g50_q6m2y45l800u200',\
    'PK_e0v4z13_w40a4p800f800g50_q6m2y45l800u200',\
    'PK_e0v4z13_w15a4p800f800g50_q6m2y45l800u200',\
    'PK_e0v4z13_w25a4p800f800g50_q6m2y45l800u200',\
    'PK_e0v4z13_w35a4p800f800g50_q6m2y45l800u200',\
    'PK_e0v4z13_w15a2p800f800g50_q6m2y45l800u200',\
    'PK_e0v4z13_w15a2p400f800g50_q6m2y45l800u200',\
    'PK_e0v4z13_w15a6p800f800g50_q6m2y45l800u200',\
    'PK_e0v4z13_w15a8p800f800g50_q6m2y45l800u200',\
    'PK_e0v4z13_w15a4p400f800g50_q6m2y45l800u200',\
    'PK_e0v4z13_w15a6p400f800g50_q6m2y45l800u200',\
    'PK_e0v4z13_w15a8p400f800g50_q6m2y45l800u200',\
    'PK_e0v4z13_w15a2p600f800g50_q6m2y45l800u200',\
    'PK_e0v4z13_w15a4p600f800g50_q6m2y45l800u200',\
    'PK_e0v4z13_w15a6p600f800g50_q6m2y45l800u200',\
    'PK_e0v4z13_w15a8p600f800g50_q6m2y45l800u200',\
    'PK_e0v4z13_w15a4p800f800g50',\
    'PK_e0v4z13_a4x75y180w5v30p400_q6m2y45',\
    'PK_e0v4z13_a4x75y180w5v30p800_q6m2y45',\
    'PK_e0v4z13_a4x75y0w5v30p800_q6m2y45',\
    'PK_e0v4z13_a4x75y270w5v30p800_q6m2y45',\
    'PK_e0v4z13_a4x75y90w5v30p800_q6m2y45',\
    'PK_e0v4z13']
    
    #take_zonal_means(indir, outdir)

    #print(datetime.now(), ' - calculating constants')
    #ds = xr.open_dataset('../atmos_daily_T42_p40.nc', decode_times=False)
    #deg2rad = np.pi / 180
    #coslat = np.cos(np.deg2rad(ds.lat))
    #acoslat = 6.371e6 * coslat
    #print(datetime.now(), ' - taking pressure differential')
    #p = ds.phalf
    #dp = np.diff(p)
    ##p = ds.pfull
    ##dp = np.gradient(p)
    
    for i in range(len(exp)):
        postprocess(exp[i])
        #find_TKE(outdir, exp[i])
        #calc_w(outdir, exp[i])

    

    
