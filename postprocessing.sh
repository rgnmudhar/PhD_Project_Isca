#!/bin/bash 
indir=/disco/share/rm811/isca_data
outdir=/disco/share/rm811/processed
plevdir=/home/links/rm811/Isca/postprocessing/plevel_interpolation/scripts
analysisdir=/home/links/rm811/scratch/PhD_Project_Isca
exp=PK_e0v4z13_w15a4p300f800g50_q6m2y45l800u200
#_w15a4p300f800g50_q6m2y45l800u200
echo $exp

# run plevel interpolation
echo "$(date)"
echo interpolate
cd ${indir}/${exp}
nfiles=$(ls | wc -l)
cd $plevdir
nice -19 ipython run_plevel.py $indir $exp $nfiles

# delete pre-interpolation files
cd ${indir}/${exp}
find . -name 'atmos_daily.nc' -type f -delete

# ignore 2 years of spin-up
echo ignore_spin_up
cd ${indir}/${exp}
for i in $(seq 24 $END); do 
name=run*0$i;
mv $name spinup_run$i;
done

# concatenate all
echo "$(date)"
echo concatenate
ncrcat run*/*.nc ${exp}_all.nc

# extract variables
echo $exp
echo "$(date)"
echo extract_variables
ncks -v ucomp ${exp}_all.nc ${exp}_u.nc
ncks -v vcomp ${exp}_all.nc ${exp}_v.nc
ncks -v temp ${exp}_all.nc ${exp}_T.nc
ncks -v height ${exp}_all.nc ${exp}_h.nc

# take zonal means
echo "$(date)"
echo zonal_means
ncwa -a lon ${exp}_u.nc ${exp}_uz.nc
ncwa -a lon ${exp}_v.nc ${exp}_vz.nc
ncwa -a lon ${exp}_T.nc ${exp}_Tz.nc
ncwa -a lon ${exp}_h.nc ${exp}_hz.nc

# take time mean
echo "$(date)"
echo time_means
#ncra ${exp}_u.nc ${exp}_ut.nc
#ncra ${exp}_v.nc ${exp}_vt.nc
#ncra ${exp}_T.nc ${exp}_Tt.nc
ncra ${exp}_h.nc ${exp}_ht.nc
# take time mean of zonal means
ncra ${exp}_uz.nc ${exp}_utz.nc
ncra ${exp}_vz.nc ${exp}_vtz.nc
ncra ${exp}_Tz.nc ${exp}_Ttz.nc
ncra ${exp}_hz.nc ${exp}_htz.nc

# check KE spin-up
echo "$(date)"
echo spin_up
cd $analysisdir
nice -19 ipython spinup_KE.py $indir $exp

# remove file with all included
# move created files to folder for processed data
echo $exp
echo "$(date)"
echo rearrange_files
rm ${exp}_all.nc
rm ${exp}_h.nc
rm ${exp}_hz.nc
mv ${exp}*.nc $outdir/