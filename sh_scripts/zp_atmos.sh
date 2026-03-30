#!/bin/bash


#python run_scripts/telescope/zero_points_atmos.py --sigma_pwv_min=0.01 --sigma_pwv_max=0.3 --sigma_pwv_step=0.01 --nproc=16 --airmass_step=0.05 --outName=zp_atmos_pwv.hdf5 --airmass_max=2.5

#python run_scripts/telescope/zero_points_atmos.py --nproc=16 --airmass_step=0.05 --sigma_aerosol_min=0.001 --sigma_aerosol_max=0.02 --sigma_aerosol_step=0.001  --outName=zp_atmos_aerosol.hdf5

python run_scripts/telescope/zero_points_atmos.py --nproc=16 --airmass_step=0.05 --sigma_airmass_min=0.001 --sigma_airmass_max=0.02 --sigma_airmass_step=0.001  --outName=zp_atmos_airmass.hdf5

python run_scripts/telescope/zero_points_atmos.py --nproc=16 --airmass_step=0.05 --sigma_ozone_min=5 --sigma_ozone_max=50 --sigma_ozone_step=5  --outName=zp_atmos_ozone.hdf5
