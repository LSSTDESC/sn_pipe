import numpy as np
import os

for bluecutoff in [360.0, 370.0, 380.0]:
    bluecutoff = np.round(bluecutoff, 1)
    cmd_comm = 'python run_scripts/fakes/full_simulation_fit.py  --ebvofMW 0.0 --outDir_simu Output_Simu_{}_ebvofMW_0.0 --outDir_fit Output_Fit_{}_ebvofMW_0.0 --bluecutoff {}'.format(
        bluecutoff, bluecutoff, bluecutoff)

    cmd = cmd_comm
    cmd += ' --simulator sn_fast'
    os.system(cmd)

    cmd = cmd_comm
    cmd += ' --simulator sn_cosmo'
    os.system(cmd)
