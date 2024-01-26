from nco import Nco
from glob import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import shutil
from datetime import datetime
from shared_functions import *

def retrospective_calcs(indir, outdir):
    os.chdir(indir)
    exp = sorted(glob('*'))
    os.chdir(outdir)
    for i in range(len(exp)):
        print(datetime.now(), ' - {0:.0f}/{1:.0f} - '.format(i+1, len(exp)), exp[i])
        #nco.ncwa(input = exp[i]+'_T.nc', output = exp[i]+'_Tz.nc', options = ['-a lon']) #zonal mean

        print(datetime.now(), ' - concatenate')
        os.chdir(indir + exp[i])
        nco.ncrcat(input = 'run*/*interp.nc', output = exp[i]+'_all.nc', use_shell = True)
        # extract variables
        print(datetime.now(), ' - extract variable')
        nco.ncks(input = exp[i]+'_all.nc', output = exp[i]+'_h.nc', options = ['-v height'])
        print(datetime.now(), ' - (re)move files')
        os.remove(exp[i]+'_all.nc')
        os.rename(exp[i]+'_h.nc', outdir+exp[i]+'_h.nc')

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

def find_TKE(indir, outdir):
    """
    This attempts to use TKE for determining spin-up.
    For every 'time'  (i.e. atmos_monthly file) take the zonal mean of u and v along longitude.
    Then at each (lat,p) find 0.5*(u^2 + v^2).
    Then take a weighted average along lat, and finally a mean along pressure.
    Based on script by Penelope Maher and discussions with William Seviour.
    """
    os.chdir(indir)
    exp = sorted(glob('*'))
    for i in range(len(exp)):
        os.chdir(indir+exp[i])
        print(datetime.now(), ' - finding KE for {0:.0f}/{1:.0f} - '.format(i+1, len(exp)), exp[i])
        KE = []
        files1 = glob('spin*/*.nc')
        files2 = sorted(glob('run*/*.nc'))
        files = files1 + files2
        ds = xr.open_mfdataset(files, decode_times=False)
        uz = ds.ucomp.mean('lon')
        vz = ds.vcomp.mean('lon')
        coslat = np.cos(np.deg2rad(ds.coords['lat'].values)).clip(0., 1.) # need to weight due to different box sizes over grid
        lat_wgts = np.sqrt(coslat)
        for j in range(len(uz)):
            TKE_box = np.empty_like(uz[j])
            for q in range(len(ds.coords['pfull'].data)):
                for k in range(len(ds.coords['lat'].data)):
                    TKE_box[q,k] = calc_TKE(uz[j][q,k], vz[j][q,k])
            TKE_box = np.average(TKE_box, axis=1, weights=lat_wgts)
            TKE_avg = np.nanmean(TKE_box) # should I weight pressures too? How?
            KE.append(TKE_avg)
        save_file(outdir, exp[i], KE, 'KE')

        print(datetime.now(), " - plotting")
        KE = open_file(outdir, exp[i], 'KE')
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(len(KE), KE, color='k')
        ax.set_xlim(1,len(KE))
        ax.set_xlabel("Days Simulated")       
        ax.set_ylabel('total KE', color='k')
        plt.savefig(outdir+exp[i]+'_spinup.pdf', bbox_inches = 'tight')
        plt.close()

def remove_uninterp(indir, exp):
    print(datetime.now(), ' - interpolation processing already done')
    #os.chdir(indir + exp)
    #runs = sorted(glob('run*'))
    #n = len(runs) - 1
    # PLEASE NOTE WITH THE FOLLOWING EXCLUDED, RUN_PLEVEL.PY MUST BE RUN BEFORE THIS SCRIPT
    #os.chdir(plevdir)
    #run_plevel.main(indir, exp, n)

    os.chdir(indir)
    for i in range(len(exp)):
        # delete pre-interpolation files
        print(datetime.now(), ' - {0:.0f}/{1:.0f} - '.format(i+1, len(exp)), exp[i])
        os.chdir(indir + exp[i])
        files = glob('*/atmos_daily.nc', recursive=True)
        for f in files:
            try:
                os.remove(f)
            except OSError as e:
                print(e.strerror, ':', f)

def delete_spinup(indir):
    print(datetime.now(), ' - delete spin-up data')
    os.chdir(indir)
    folders = glob('*/spin*', recursive=True)
    for f in folders:
        print(datetime.now(), ' - ', f)
        try:
            shutil.rmtree(f)
        except OSError as e:
            print(e.strerror, ':', f)

def delete_data(indir, exp):
    print(datetime.now(), ' - remove surplus data')
    os.chdir(indir)
    folder_keep = 'run0025'
    for e in exp:
        # find folders
        os.chdir(indir + e)
        os.rename(folder_keep, 'keep_'+folder_keep)
        folders = glob('run*', recursive=True)
        print(datetime.now(), ' - remove everything but restarts and run0025 folders')
        for f in folders:
            try:
                shutil.rmtree(f)
            except OSError as err:
                print(err.strerror, ':', f)
        os.chdir(indir + e)
        os.rename('keep_'+folder_keep, folder_keep)
    
def postprocess(exp):
    print(datetime.now(), ' - ', exp)
    os.chdir(indir + exp)
    """runs = sorted(glob('run*'))
    n = len(runs) - 1
    
    # ignore X years of spin-up
    print(datetime.now(), ' - ignore spin-up')
    X = 2
    n = 0
    while n < (X * 12):
        os.rename(runs[n], 'spinup'+str(n))
        n += 1
    """
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
    # NOTE THAT THE FOLLOWING ONLY SEEMS TO WORK ON GV3 OR GV4?
    print(datetime.now(), ' - zonal means')
    nco.ncwa(input = exp+'_u.nc', output = exp+'_uz.nc', options = ['-a lon'])
    nco.ncwa(input = exp+'_v.nc', output = exp+'_vz.nc', options = ['-a lon'])
    nco.ncwa(input = exp+'_T.nc', output = exp+'_Tz.nc', options = ['-a lon'])

    # remove file with all included
    # move created files to folder for processed data
    print(datetime.now(), ' - re-arrange files')
    os.chdir(indir + exp)
    os.remove(exp+'_all.nc')
    os.remove(exp+'_vt.nc')
    os.remove(exp+'_Tt.nc')
    newfiles = glob('*.nc', recursive=True)
    for f in newfiles:
        try:
            os.rename(f, outdir+f)
        except OSError as e:
            print(e.strerror, ':', f)

    # in original folder remove everything but restarts and run0025 folders
    print(datetime.now(), ' - remove surplus data')
    print(datetime.now(), ' - rename run0025 folder')
    folder_keep = 'run0025'
    os.rename(folder_keep, 'keep_'+folder_keep)

    print(datetime.now(), ' - remove everything but restarts and run0025 folders')
    folders = glob('run*', recursive=True)
    for f in folders: # delete all folders starting with 'run'
        try:
            shutil.rmtree(f)
        except OSError as e:
            print(e.strerror, ':', f)
    print(datetime.now(), ' - rename run0025 folder')
    os.rename('keep_'+folder_keep, folder_keep)


if __name__ == '__main__': 
    indir = '/disco/share/rm811/isca_data/'
    outdir = '/disco/share/rm811/processed/'
    plevdir = '/home/links/rm811/Isca/postprocessing/plevel_interpolation/scripts/'
    analysisdir = '/home/links/rm811/scratch/PhD_Project_Isca/'

    nco = Nco(debug=True)

    func = input('Do you want to a) postprocess, b) remove uninterpolated files, c) find TKE, d) back-calculate w, e) retrospectively extract variables, or f) delete spin-up data?')

    exp = ['PK_e0v4z13_w15a4p300f800g50_q6m2y45u300_s']
    
    if func == 'b':
        remove_uninterp(indir, exp)    
    elif func == 'c':
        find_TKE(indir, analysisdir)  
    elif func == 'e':
        retrospective_calcs(indir, outdir)
    elif func == 'f':
        delete_spinup(indir)
    else:
        for i in range(len(exp)):
            if func == 'a':
                postprocess(exp[i])
            elif func == 'd':
                calc_w(outdir, exp[i])


    #sys.path.append(os.path.abspath(plevdir))
    #import run_plevel 

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
