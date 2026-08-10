#!/bin/bash

python for_batch/scripts/sim_to_fit/prodIt.py --dbList_DD=DD_fbs_5.0.0_extract.csv --outDir_DD=Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_airmass_newatm_lc_coadd --Observations_coadd=0 --mem=10Gb --tag_script=lc_coadd

python for_batch/scripts/sim_to_fit/prodIt.py --dbList_DD=DD_fbs_5.0.0_extract.csv --outDir_DD=Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_airmass_obs_coadd --Observations_coadd=1 --LC_coadd=0 --mem=10Gb --tag_script=obs_coadd
