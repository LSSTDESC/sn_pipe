import numpy as np
import os


def run(x1,color,simus,ebv,bluecutoff,redcutoff,error_model,fake_config,snrmin,nbef,naft):

    cutoff = '{}_{}'.format(bluecutoff,redcutoff)
    if error_model:
        cutoff = 'error_model'
    
    outDir_simu = 'Output_Simu_{}_ebvofMW_{}'.format(cutoff,ebv)
    outDir_fit = 'Output_Fit_{}_ebvofMW_{}_snrmin_{}'.format(cutoff,ebv,int(snrmin))
    
    cmd_comm = 'python run_scripts/fakes/full_simulation_fit.py  --ebvofMW {}'.format(ebv)
    cmd_comm += ' --outDir_simu {} --outDir_fit {}'.format(outDir_simu,outDir_fit)
    cmd_comm += ' --bluecutoff {} --redcutoff {}'.format(bluecutoff,redcutoff) 
    cmd_comm += ' --fake_config {} --x1 {} --color {} --error_model {}'.format(fake_config, x1, color,error_model)
    cmd_comm += ' --snrmin {} --nbef {} --naft {}'.format(snrmin,nbef,naft)

    for simu in simus:
            cmd = cmd_comm
            cmd += ' --simulator {}'.format(simu)
            os.system(cmd)
    

simus = ['sn_fast', 'sn_cosmo']
#simus = ['sn_cosmo']
ebvs = [0.0]
redcutoff = 800.0
blues = [360.0, 370.0, 380.0]
blues = [380.]
fake_config = 'input/Fake_cadence/Fake_cadence.yaml'
x1 = -2.0
color = 0.2
error_model = 1

snrmin = 5.
nbef = 4
naft = 5

#run(x1,color,simus,0.0,380.,800.,1,fake_config,snrmin,nbef,naft)

for bluecutoff in blues:
    run(x1,color,simus,0.0,bluecutoff,800.,0,fake_config,snrmin,nbef,naft)
