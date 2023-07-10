import numpy as np
import os

scr_m = 'python run_scripts/sim_to_fit/run_fake.py \
    --scriptName moon_run_b.sh --show_results 0 \
    --config_obs=input/DESC_cohesive_strategy/combi_obs.csv \
    --config_simu=input/DESC_cohesive_strategy/combi_simufit.csv \
    --SN_x1_type=unique --SN_color_type=unique --SN_z_type=uniform \
    --SN_z_min=0.01 --SN_z_max=0.9 \
    --SN_daymax_type=uniform --mooncompensate=1 --SN_NSNabsolute 1 \
    --MultiprocessingFit_nproc 12'
#scr_m = 'python run_scripts/sim_to_fit/run_test.py --show_results=0 --scriptName=moon_run.sh --config_obs=input/DESC_cohesive_strategy/combi_obs.csv --config_simu=input/DESC_cohesive_strategy/combi_simufit.csv'
os.system(scr_m)
