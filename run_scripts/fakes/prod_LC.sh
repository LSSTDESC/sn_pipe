#!/bin/bash

# SNIa template
python run_scripts/fakes/loop_full_fast.py --error_model 0 --sn_type SN_IaT --sn_model nugent-sn1a --simus sn_cosmo --bluecutoff 0.0 --redcutoff 1.e5 --sn_version 1.2

# SNIa SALT2
python run_scripts/fakes/loop_full_fast.py --sn_type SN_Ia --sn_model salt2-extended --simus sn_cosmo --x1 0.2 --color -0.1 --sn_version 1.0 --error_model 1
