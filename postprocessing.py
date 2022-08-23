from nco import Nco
from glob import glob
import xarray as xr
import numpy as np
import os
import sys
from datetime import datetime

indir = '/disco/share/rm811/isca_data/'
outdir = '/disco/share/rm811/processed/'
plevdir = '/home/links/rm811/Isca/postprocessing/plevel_interpolation/scripts'
analysisdir = '/home/links/rm811/scratch/PhD_Project_Isca'

nco = Nco(debug=True)

#sys.path.append(os.path.abspath(plevdir))
#import run_plevel 

def take_zonal_means(indir, outdir):
    os.chdir(indir)
    exp = sorted(glob('*'))
    os.chdir(outdir)
    for i in range(len(exp)):
        print(datetime.now(), ' - ' + exp[i] + ' zonal means')
        nco.ncwa(input = exp[i]+'_u.nc', output = exp[i]+'_uz.nc', options = ['-a lon'])
        nco.ncwa(input = exp[i]+'_v.nc', output = exp[i]+'_vz.nc', options = ['-a lon'])

def calc_w():
    ds = xr.open_dataset('../atmos_daily_T42_p40.nc', decode_times=False)
    deg2rad = np.pi / 180
    coslat = np.cos(np.deg2rad(ds.lat))
    acoslat = 6.371e6 * coslat
    p = ds.phalf
    dp = np.diff(p)
    #p = ds.pfull
    #dp = np.gradient(p)

    u = xr.open_dataset(outdir+exp+'_u.nc', decode_times=False).ucomp
    v = xr.open_dataset(outdir+exp+'_v.nc', decode_times=False).vcomp
    div = u.differentiate('lon', edge_order=2) / deg2rad / acoslat + (v*coslat).differentiate('lat', edge_order=2) / deg2rad / acoslat
    wcalc_half = -np.cumsum(div.values * (dp*100)[None,:,None,None], axis=1)
    wcalc_half = np.concatenate((np.zeros((len(ds.time),1,len(ds.lat),len(ds.lon))), wcalc_half), axis=1)
    wcalc_full = (wcalc_half[:,1:] + wcalc_half[:,:-1]) / 2.
    ds['omega'] = (['time', 'pfull', 'lat', 'lon'], wcalc_full)

    print(ds.omega.shape)

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


basis = 'PK_e0v4z13'
perturb = '_q6m2y45' #l800u200'
polar = '_w15a4p800f800g50'
exp = [basis+'_a4x75y180w5v30p800']

#take_zonal_means(indir, outdir)

for i in range(len(exp)):
    postprocess(exp[i])