import numpy as np
import os

simus = ['sn_cosmo']
ebvs = [0.0]
redcutoff = 800.0
blues = [360.0, 370.0, 380.0]
blues = [360.]
for bluecutoff in blues:
    bluecutoff = np.round(bluecutoff, 1)
    for ebv in ebvs:
        cmd_comm = 'python run_scripts/fakes/full_simulation_fit.py  --ebvofMW {} --outDir_simu Output_Simu_{}_{}_ebvofMW_{} --outDir_fit Output_Fit_{}_{}_ebvofMW_{} --bluecutoff {} --redcutoff {}'.format(
            ebv, bluecutoff, redcutoff, ebv, bluecutoff, redcutoff, ebv, bluecutoff, redcutoff)

        # fast simulator
        # for simu in
        for simu in simus:
            cmd = cmd_comm
            cmd += ' --simulator {}'.format(simu)
            os.system(cmd)

        """
        # sncosmo simulator
        cmd = cmd_comm
        cmd += ' --simulator sn_cosmo'
        os.system(cmd)
        """
