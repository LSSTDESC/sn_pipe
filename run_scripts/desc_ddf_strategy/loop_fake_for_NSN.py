import numpy as np
import os
scr_m = 'python run_scripts/sim_to_fit/run_fake.py \
        --scriptName moon_run_b.sh --show_results 0 \
        --config_obs=input/DESC_cohesive_strategy/combi_obs.csv \
        --config_simu=input/DESC_cohesive_strategy/combi_simufit.csv \
        --SN_NSNfactor 300 --SN_x1_type=random --SN_color_type=random \
        --SN_z_type=random --SN_daymax_type=random'
scr_m += ' --mooncompensate=1 --MultiprocessingFit_nproc 12'

dz = 0.1
zmin = 0.01
zmax = 1.1
zvals = np.arange(zmin, zmax, dz)

for io in range(len(zvals)-1):
    zzmin = np.round(zvals[io], 2)
    zzmax = np.round(zvals[io+1], 2)
    add_tag = '{}_{}'.format(np.round(zzmin, 2), np.round(zzmax, 2))
    scr = scr_m
    scr += ' --SN_z_min={}'.format(zzmin)
    scr += ' --SN_z_max={}'.format(zzmax)
    scr += ' --add_tag={}'.format(add_tag)
    print(scr)
    os.system(scr)
