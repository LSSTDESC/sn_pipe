from optparse import OptionParser
from sn_tools.sn_io import make_dict_from_config
from sn_tools.sn_io import add_parser
import argparse
import copy
from sn_tools.sn_batchutils import BatchIt
import numpy as np


def batch_DDF(theDict, scriptref='run_scripts/sim_to_fit/run_sim_to_fit.py',
              time='30:00:00', mem='40G'):
    """

    Function to launch sim_to_fit for DD fields

     Parameters
     ----------
     theDict : dict
         script parameters.
     scriptref : str, optional
         script to use for production. The default is
             'run_scripts/sim_to_fit/run_sim_to_fit.py'.
     time : str, optional
           processing time per job. The default is '30:00:00'.
     mem : str, optional
           job mem. The default is '40G'.

    Returns
    -------
    None.

    """
    DD_list = theDict['DD_list'].split(',')
    dbName = theDict['dbName']
    outDir = theDict['OutputSimu_directory']

    procDict = copy.deepcopy(theDict)

    del procDict['DD_list']

    # procDict['nside'] = 128
    # procDict['fieldType'] = 'DD'

    procDict['Fitter_parnames'] = 'z,t0,x1,c,x0'
    procDict['OutputSimu_directory'] = '{}/{}'.format(outDir, dbName)
    procDict['OutputFit_directory'] = procDict['OutputSimu_directory']

    for fieldName in DD_list:
        procDict['fieldName'] = fieldName
        procName = 'DD_{}_{}'.format(dbName, fieldName)
        mybatch = BatchIt(processName=procName, time=time, mem=mem)
        for season in range(1, 11):
            procDict['ProductionIDSimu'] = 'SN_DD_{}_{}'.format(
                fieldName, season)
            procDict['Observations_season'] = season
            mybatch.add_batch(scriptref, procDict)

        # go for batch
        mybatch.go_batch()


def batch_WFD(theDict, scriptref='run_scripts/sim_to_fit/run_sim_to_fit.py',
              time='40:00:00', mem='40G'):
    """
    Function to launch sim_to_fit for WFD

    Parameters
    ----------
    theDict : dict
        script parameters.
    scriptref : str, optional
        script to use for production. The default is
            'run_scripts/sim_to_fit/run_sim_to_fit.py'.
    time : str, optional
        processing time per job. The default is '30:00:00'.
    mem : str, optional
        job mem. The default is '40G'.

    Returns
    -------
    None.

    """

    dbName = theDict['dbName']
    outDir = theDict['OutputSimu_directory']

    procDict = copy.deepcopy(theDict)

    del procDict['DD_list']
    # procDict['nside'] = 64
    # procDict['fieldType'] = 'WD'
    procDict['Fitter_parnames'] = 'z,t0,x1,c,x0'
    procDict['OutputSimu_directory'] = '{}/{}'.format(outDir, dbName)
    procDict['OutputFit_directory'] = procDict['OutputSimu_directory']

    deltaRA = 36.

    RAs = np.arange(0., 360., deltaRA)

    for RA in RAs[:-1]:
        RAmin = np.round(RA, 1)
        RAmax = RAmin+deltaRA
        RAmax = np.round(RAmax, 1)
        procName = 'WFD_{}_{}_{}'.format(dbName, RAmin, RAmax)
        mybatch = BatchIt(processName=procName, time=time, mem=mem)
        for seas in range(1, 11):
            procDict['ProductionIDSimu'] = 'SN_{}_{}'.format(procName, seas)
            procDict['Observations_season'] = seas
            mybatch.add_batch(scriptref, procDict)

        # go for batch
        mybatch.go_batch()


# get script parameters and put in a dict
path = 'for_batch/input/sn_prod'
confDict = make_dict_from_config(path, 'config_sn_prod.txt')

parser = argparse.ArgumentParser(
    description='Script to simulate, select, and Fit LC - full sky')

parser = OptionParser()

add_parser(parser, confDict)

opts, args = parser.parse_args()

# load the new values
procDict = {}
for key, vals in confDict.items():
    procDict[key] = eval('opts.{}'.format(key))


# this is for DDFs
if opts.fieldType == 'DD':
    batch_DDF(procDict)

# this is for WFD
if opts.fieldType == 'WFD':
    procDict['SN_z_max'] = 0.6
    batch_WFD(procDict)