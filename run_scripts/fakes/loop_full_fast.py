import numpy as np
import os

simus = ['sn_fast', 'sn_cosmo']
#simus = ['sn_fast']
ebvs = [0.1]
redcutoff = 800.0
blues = [360.0, 370.0, 380.0]
blues = [380.]
fake_config = 'input/Fake_cadence/Fake_cadence.yaml'
x1 = 0.0
color = 0.0

for bluecutoff in blues:
    bluecutoff = np.round(bluecutoff, 1)
    for ebv in ebvs:
        cmd_comm = 'python run_scripts/fakes/full_simulation_fit.py  --ebvofMW {} --outDir_simu Output_Simu_{}_{}_ebvofMW_{} --outDir_fit Output_Fit_{}_{}_ebvofMW_{} --bluecutoff {} --redcutoff {} --fake_config {} --x1 {} --color {}'.format(
            ebv, bluecutoff, redcutoff, ebv, bluecutoff, redcutoff, ebv, bluecutoff, redcutoff, fake_config, x1, color)

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
