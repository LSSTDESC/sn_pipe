#!/bin/bash


#python run_scripts/telescope/zero_points_atmos.py --bias_airmass_step=0.001 --bias_airmass_min=-0.08 --bias_airmass_max=0.08 --airmass_max=2.5 --airmass_step=0.1 --nsample=1 --outDir=../zp_atmos_bias --outName=zp_atmos_airmass.hdf5 --nproc=16

#python run_scripts/telescope/zero_points_atmos.py --bias_pwv_step=0.001 --bias_pwv_min=-0.08 --bias_pwv_max=0.08 --airmass_max=2.5 --airmass_step=0.1 --nsample=1 --outDir=../zp_atmos_bias --outName=zp_atmos_pwv.hdf5 --nproc=16

python run_scripts/telescope/zero_points_atmos.py --bias_ozone_step=0.001 --bias_ozone_min=-0.08 --bias_ozone_max=0.08 --airmass_max=2.5 --airmass_step=0.1 --nsample=1 --outDir=../zp_atmos_bias --outName=zp_atmos_ozone.hdf5 --nproc=16

python run_scripts/telescope/zero_points_atmos.py --bias_aerosol_step=0.001 --bias_aerosol_min=-0.08 --bias_aerosol_max=0.08 --airmass_max=2.5 --airmass_step=0.1 --nsample=1 --outDir=../zp_atmos_bias --outName=zp_atmos_aerosol.hdf5 --nproc=16
