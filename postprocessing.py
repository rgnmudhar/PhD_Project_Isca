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
    """
    # run plevel interpolation
    print(datetime.now(), ' - interpolate')
    os.chdir(indir + exp)
    runs = sorted(glob('run*'))
    n = len(runs) - 1
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
    nco = Nco(debug=True)
    nco.ncrcat(input = 'run*/*interp.nc', output = exp+'_all.nc', use_shell = True)

    # extract variables
    print(datetime.now(), ' - extract variables')
    all = exp+'_all.nc'
    nco.ncks(input = all, output = exp+'_u.nc', options = ['-v ucomp'])
    nco.ncks(input = all, output = exp+'_v.nc', options = ['-v vcomp'])
    nco.ncks(input = all, output = exp+'_w.nc', options = ['-v omega'])
    nco.ncks(input = all, output = exp+'_T.nc', options = ['-v temp'])
    nco.ncks(input = all, output = exp+'_h.nc', options = ['-v height'])
    """
    # time means
    print(datetime.now(), ' - time means')
    os.chdir(indir + exp)
    nco = Nco(debug=True)
    nco.ncra(input = exp+'_h.nc', output = exp+'_ht.nc')
    nco.ncra(input = exp+'_u.nc', output = exp+'_ut.nc')
    nco.ncra(input = exp+'_v.nc', output = exp+'_vt.nc')
    nco.ncra(input = exp+'_T.nc', output = exp+'_Tt.nc')

    # time means
    print(datetime.now(), ' - time and zonal means')
    nco.ncra(input = exp+'_ht.nc', output = exp+'_htz.nc')
    nco.ncra(input = exp+'_ut.nc', output = exp+'_utz.nc')
    nco.ncra(input = exp+'_vt.nc', output = exp+'_vtz.nc')
    nco.ncra(input = exp+'_Tt.nc', output = exp+'_Ttz.nc')

    # zonal means
    # PLEASE NOTE THIS DOES NOT SEEM TO WORK ANYMORE!!!
    #print(datetime.now(), ' - zonal means')
    #nco.ncwa(input = exp+'_u.nc', output = exp+'_uz.nc', options = ['-a lon'])
    #nco.ncwa(input = exp+'_v.nc', output = exp+'_vz.nc', options = ['-a lon'])
    #nco.ncwa(input = exp+'_T.nc', output = exp+'_Tz.nc', options = ['-a lon'])

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

for i in range(len(exp)):
    postprocess(exp[i])

    """
    [basis+polar, \
    basis+'_w15a4p900f800g50'+perturb, \
    basis+polar+perturb, \
    basis+'_w15a4p700f800g50'+perturb, \
    basis+'_w15a4p600f800g50'+perturb, \
    basis+'_w15a4p400f800g50'+perturb, \
    basis+'_w15a4p500f800g50'+perturb, \
    basis+'_w15a4p300f800g50'+perturb, \
    basis+'_w10a4p800f800g50'+perturb, \
    basis+'_w20a4p800f800g50'+perturb, \
    basis+'_w25a4p800f800g50'+perturb, \
    basis+'_w30a4p800f800g50'+perturb, \
    basis+'_w35a4p800f800g50'+perturb, \
    basis+'_w40a4p800f800g50'+perturb]

    ['PK_e10v1z18',\
    'PK_e10v2z18',\
    'PK_e10v3z18',\
    'PK_e10v4z18',\
    'PK_e10v1z13',\
    'PK_e10v2z13',\
    'PK_e10v3z13',\
    'PK_e10v4z13']
    """
