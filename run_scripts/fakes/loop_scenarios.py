import os
import glob
from sn_tools.sn_utils import multiproc
from optparse import OptionParser
import numpy as np
import pandas as pd


def loop_it(listDB, params, j=0, output_q=None):
    """
    Function to generate Fakes

    Parameters
    ----------
    listDB : list(str)
        list of db to process.
    params : dict
        parameters.
    j : int, optional
        internal tag for multiprocessing. The default is 0.
    output_q : multiprocessing queue, optional
        where to store the results. The default is None.

    Returns
    -------
    int
        just to say it is the end..

    """

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


parser = OptionParser(
    description='Script to generate fake data from scenarios.')

parser.add_option('--dirScen', type='str',
                  default='input/DESC_cohesive_strategy',
                  help='input dir[%default]')
parser.add_option('--budget_DD', type=float,
                  default=0.07,
                  help='DD budget [%default]')
parser.add_option('--configFile', type='str', default='scenarios.csv',
                  help='config file to use[%default]')

opts, args = parser.parse_args()


dirScen = opts.dirScen
budget = np.round(opts.budget_DD, 2)
configFile = opts.configFile

# load configs (ie scenarios)
df_config = pd.read_csv(configFile, comment='#')

df_config['name'] = dirScen+'/'+df_config['name'] + '_{}'.format(budget)+'.csv'


list_scen = df_config['name'].unique()

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
