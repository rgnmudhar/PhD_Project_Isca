from nco import Nco
from glob import glob
import os
import sys
from datetime import datetime

indir = '/disco/share/rm811/isca_data/'
outdir = '/disco/share/rm811/processed/'
plevdir = '/home/links/rm811/Isca/postprocessing/plevel_interpolation/scripts'
analysisdir = '/home/links/rm811/scratch/PhD_Project_Isca'

sys.path.append(os.path.abspath(plevdir))
import run_plevel 

def postprocess(exp):
    print(datetime.now(), ' - ', exp)

    # run plevel interpolation
    print(datetime.now(), ' - interpolate')
    os.chdir(indir + exp)
    runs = sorted(glob('run*'))
    n = len(runs) - 1
    run_plevel.main(indir, exp, n)

    # delete pre-interpolation files
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
    nco = Nco(debug=True)
    nco.ncrcat(input = 'run*/*.nc', output = exp+'_all.nc', use_shell = True)

    # extract variables
    print(datetime.now(), ' - extract variables')
    all = exp+'_all.nc'
    nco.ncks(input = all, output = exp+'_u.nc', options = ['-v ucomp'])
    nco.ncks(input = all, output = exp+'_v.nc', options = ['-v vcomp'])
    nco.ncks(input = all, output = exp+'_T.nc', options = ['-v temp'])
    nco.ncks(input = all, output = exp+'_h.nc', options = ['-v height'])
    
    # zonal means
    print(datetime.now(), ' - zonal means')
    nco.ncwa(input = exp+'_u.nc', output = exp+'_uz.nc', options = ['-a lon'])
    nco.ncwa(input = exp+'_v.nc', output = exp+'_vz.nc', options = ['-a lon'])
    nco.ncwa(input = exp+'_T.nc', output = exp+'_Tz.nc', options = ['-a lon'])
    nco.ncwa(input = exp+'_h.nc', output = exp+'_hz.nc', options = ['-a lon'])

    # time means
    print(datetime.now(), ' - time means')
    nco.ncra(input = exp+'_h.nc', output = exp+'_ht.nc')
    nco.ncra(input = exp+'_uz.nc', output = exp+'_utz.nc')
    nco.ncra(input = exp+'_vz.nc', output = exp+'_vtz.nc')
    nco.ncra(input = exp+'_Tz.nc', output = exp+'_Ttz.nc')
    nco.ncra(input = exp+'_hz.nc', output = exp+'_htz.nc')
    
    # remove file with all included
    # move created files to folder for processed data
    print(datetime.now(), ' - re-arrange files')
    os.remove(exp+'_all.nc')
    os.remove(exp+'_h.nc')
    os.remove(exp+'_hz.nc')
    newfiles = glob('*.nc', recursive=True)
    for f in newfiles:
        try:
            os.rename(f, outdir+f)
        except OSError as e:
            print(e.strerror, ':', f)


basis = 'PK_e0v4z13'
perturb = '_q6m2y45l800u200'
exp = [basis, basis+perturb] #, \
    #basis+'_w15a4p900f800g50'+perturb, \
    #basis+'_w15a4p800f800g50'+perturb, \
    #basis+'_w15a4p700f800g50'+perturb, \
    #basis+'_w15a4p600f800g50'+perturb, \
    #basis+'_w15a4p400f800g50'+perturb, \
    #basis+'_w15a4p500f800g50'+perturb, \
    #basis+'_w15a4p300f800g50'+perturb]

for i in range(len(exp)):
    postprocess(exp[i])