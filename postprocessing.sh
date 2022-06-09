#!/bin/bash 
indir=/disco/share/rm811/isca_data
outdir=/disco/share/rm811/processed
plevdir=/home/links/rm811/Isca/postprocessing/plevel_interpolation/scripts
analysisdir=/home/links/rm811/scratch/PhD_Project_Isca
exp=test

# run plevel interpolation
echo interpolation
cd ${indir}/${exp}
nfiles=$(ls | wc -l)
cd $plevdir
ipython run_plevel.py $indir $exp $nfiles

# delete pre-interpolation files
cd ${indir}/${exp}
find . -name 'atmos_daily.nc' -type f -delete

# check KE spin-up
cd $analysisdir
ipython spinup_KE.py $indir $exp

# ignore 2 years of spin-up
cd ${indir}/${exp}
for i in $(seq 24 $END); do 
name=run*0$i;
mv $name spinup_run$i;
done

# concatenate all
ncrcat run*/*.nc ${exp}_all.nc

# take zonal mean
echo zonal_mean
ncwa -a lon ${exp}_all.nc ${exp}_zmean.nc
# take time mean of zonal mean
ncra ${exp}_zmean.nc ${exp}_tzmean.nc

# take time mean
echo time_mean
ncra ${exp}_all.nc ${exp}_tmean.nc

# extract variables
echo extract_variables
ncks -v ucomp ${exp}_all.nc ${exp}_u.nc
ncks -v vcomp ${exp}_all.nc ${exp}_v.nc
ncks -v temp ${exp}_all.nc ${exp}_t.nc

# remove file with all included
# move created files to folder for processed data
rm ${exp}_all.nc
mv ${exp}*.nc $outdir/