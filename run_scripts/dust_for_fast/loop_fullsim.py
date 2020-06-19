import os
import numpy as np

x1 = -2.0
color = 0.2
zmin = 0.01
zmax = 1.0
zstep = 0.01


x1 = np.round(x1, 2)
color = np.round(color, 2)

cmd = 'python run_scripts/fakes/full_simulation_fit.py'
cmd += ' --simulator sn_cosmo'
cmd += ' --x1 {}'.format(x1)
cmd += ' --color {}'.format(color)
cmd += ' --zmin {}'.format(zmin)
cmd += ' --zmax {}'.format(zmax)
cmd += ' --zstep {}'.format(zstep)

ebvals = list(np.arange(0.0, 0.06, 0.005))

for ebv in ebvals:
    comd = cmd
    comd += ' --ebvofMW {}'.format(ebv)
    print(comd)
    os.system(comd)
