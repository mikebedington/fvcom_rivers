#!/usr/bin/env bash

# example script for using ncecat to generate monthly WRF output

for THIS_YEAR in 2005 2006 2007
do
for THIS_MONTH in 01 02 03 04 05 06 07 08 09 10 11 12 
do
ncecat -v T2,RAINNC,Times -d Time,1,8 /data/euryale6/scratch/pica/models/WRF/pml-uk-wrf/run/output/sst/${THIS_YEAR}/wrfout_d03_${THIS_YEAR}-${THIS_MONTH}*18:00:00 -O ${THIS_YEAR}_${THIS_MONTH}_data.nc
done
done
