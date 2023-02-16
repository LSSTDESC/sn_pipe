from optparse import OptionParser
import pandas as pd
import os
import numpy as np

def go(config, script, lvar):
    """
    Method to make and launch the script to generate fake observations

    Parameters
    ----------
    config: pandas df
      parameters for fake obs.
    script: str
      script to use for processing
    lvar: list(str)
      list of parameters of the script to use

    """

    cmd_scr = 'python {}'.format(script)

    for index, row in config.iterrows():
        cmd_ = cmd_scr
        for var in lvar:
            cmd_ += ' --{} {}'.format(var,row[var])
        print(cmd_)
        os.system(cmd_)



parser = OptionParser(description='Fake generation, simulation and fit')

parser.add_option("--configFile", type="str",default='for_batch/input/config_fakes.csv', help="parameters for the production [%default]")
parser.add_option("--action", type="str",default='generation', help="action to perform (generation, simulation, fit) [%default]")
parser.add_option("--genDir", type="str",default='/sps/lsst/users/gris/Fake_Observations', help="output dir of observations[%default]")
parser.add_option("--simuDir", type="str",default='/sps/lsst/users/gris/Fakes/Simu', help="output dir of simulations [%default]")
parser.add_option("--fitDir", type="str",default='/sps/lsst/users/gris/Fakes/Fit', help="output dir of fits [%default]")
parser.add_option("--x1sigma", type=int, default=0,help="shift of x1 parameter distribution[%default]")
parser.add_option("--colorsigma", type=int, default=0,help="shift of color parameter distribution[%default]")
parser.add_option("--mbsigma", type=int, default=0,help="shift of Mb value [%default]")
parser.add_option("--dbNames", type="str",
                  default='DD_0.90,DD_0.85,DD_0.80,DD_0.75,DD_0.70,DD_0.65,DD_0.60,DD_0.55,DD_0.50',
                  help="config name [%default]")
parser.add_option("--snType", type="str",default='allSN', help="SN type (faintSN, mediumSN, brightSN,allSN) [%default]")
parser.add_option("--nabs", type=int, default=1500,help="number of SN per redshift bin [%default]")

opts, args = parser.parse_args()

configFile = opts.configFile
action = opts.action
genDir = opts.genDir
simuDir = opts.simuDir
fitDir = opts.fitDir
x1sigma = opts.x1sigma
colorsigma = opts.colorsigma
mbsigma = opts.mbsigma
snType = opts.snType
nabs = opts.nabs

if opts.dbNames == '':
    dbNames = []
else:
    dbNames = opts.dbNames.split(',')

#read config parameters

params = pd.read_csv(configFile,comment='#')

if dbNames:
    idx = np.isin(params['dbName'],dbNames)
    params = params[idx]

if params.empty:
    action = 'nothing'


if action == 'nothing':
    print('no data to process')

if action == 'generation':
    params['outDir'] = genDir
    params['outName'] = params['dbName']
    go(params,'run_scripts/fakes/gen_fake_obs.py',
                  ['outName','Nvisits_g','Nvisits_r','Nvisits_i','Nvisits_z','Nvisits_y','bands','outDir'])

if action == 'simulation':
    params['outDir'] = simuDir
    params['dbDir'] = genDir
    params['snTypes'] = snType # also allSN
    params['nabs'] = nabs # 1500 for allSN
    params['nsnfactor'] = 1
    params['x1sigma'] = x1sigma
    params['colorsigma'] = colorsigma
    params['mbsigma'] = mbsigma
    go(params, 'for_batch/scripts/batch_fake_simu.py', ['dbName','outDir','dbDir','snTypes','nabs','nsnfactor','x1sigma','colorsigma','mbsigma'])

if action == 'fit':
    params['outDir'] = fitDir
    params['simuDir'] = simuDir
    params['mbcov_estimate'] = 0
    params['snTypes'] = snType
    go(params,'for_batch/scripts/batch_fake_fit.py', ['dbName','outDir','simuDir','mbcov_estimate','snTypes'])

