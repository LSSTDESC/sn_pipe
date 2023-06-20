import os
import glob
from sn_tools.sn_utils import multiproc


def loop_it(listDB, params, j=0, output_q=None):

    dirScen = params['dirScen']

    cmd = 'python run_scripts/fakes/make_scenario.py --configFile {}'.format(
        dirScen)
    for scen in listDB:
        cmd_ = '{}/{}'.format(cmd, scen.split('/')[-1])
        print(cmd_)
        os.system(cmd_)

    if output_q is not None:
        return output_q.put({j: 1})
    else:
        return 1


dirScen = 'input/DESC_cohesive_strategy'

list_scen = glob.glob('{}/DDF_DESC*.csv'.format(dirScen))

list_scen += ['{}/DDF_SCOC_pII.csv'.format(dirScen),
              '{}/DDF_Univ_SN.csv'.format(dirScen),
              '{}/DDF_Univ_WZ.csv'.format(dirScen)]

params = {}
params['dirScen'] = dirScen

multiproc(list_scen, params, loop_it, 8)

"""
cmd = 'python run_scripts/fakes/make_scenario.py --configFile {}'.format(
    dirScen)

for scen in list_scen:
    cmd_ = '{}/{}'.format(cmd, scen.split('/')[-1])
    print(cmd_)
    os.system(cmd_)
"""
