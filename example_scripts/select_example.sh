#!/bin/bash

python for_batch/scripts/select/select_sn.py --dataDir=/sps/lsst/groups/cadence/LSST_SN_PhG/prod_simu/Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_airmass_newatm_lc_coadd  --outDir_pre=/sps/lsst/users/gris/Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_airmass_newatm_lc_coadd --dbList=list_OS_test_DDF.csv --scriptName=select_nsn.sh

python for_batch/scripts/select/select_sn.py --dataDir=/sps/lsst/groups/cadence/LSST_SN_PhG/prod_simu/Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_airmass_obs_coadd  --outDir_pre=/sps/lsst/users/gris/Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_airmass_obs_coadd --dbList=list_OS_test_DDF.csv --scriptName=select_nsn.sh
