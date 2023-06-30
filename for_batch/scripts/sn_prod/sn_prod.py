from optparse import OptionParser
from sn_tools.sn_io import make_dict_from_config
from sn_tools.sn_io import add_parser
import argparse
import copy
from sn_tools.sn_batchutils import BatchIt
import numpy as np
import pandas as pd


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
    reprocList = theDict['reprocList']
    sigmaInt = theDict['SN_sigmaInt']
    snrate = theDict['SN_z_rate']

    tag_list = pd.DataFrame()
    if 'None' not in reprocList:
        tag_list = pd.read_csv(reprocList)

    procDict = copy.deepcopy(theDict)

    del procDict['DD_list']
    del procDict['reprocList']
    # procDict['nside'] = 128
    # procDict['fieldType'] = 'DD'

    # procDict['Fitter_parnames'] = 'z,t0,x1,c,x0'
    tag_dir = '_spectroz'
    if 'z' in procDict['Fitter_parnames']:
        tag_dir = '_photz'
    procDict['OutputSimu_directory'] = '{}/{}/DDF{}'.format(outDir,
                                                            dbName, tag_dir)
    procDict['OutputFit_directory'] = procDict['OutputSimu_directory']
    # procDict['SN_NSNfactor'] = 30
    procDict['Pixelisation_nside'] = procDict['nside']

    for fieldName in DD_list:
        procDict['fieldName'] = fieldName
        procName = 'DD_{}_{}{}_{}_{}'.format(
            dbName, fieldName, tag_dir, np.round(sigmaInt, 2), snrate)
        mybatch = BatchIt(processName=procName, time=time, mem=mem)
        seasons = range(1, 11)
        if not tag_list.empty:
            idx = tag_list['ProductionID'] == procName
            sel = tag_list[idx]
            if len(sel) > 0:
                season_max = sel['season_max'].max()
                seasons = range(season_max, 11)

        for season in seasons:
            procDict['ProductionIDSimu'] = 'SN_{}_{}'.format(
                procName, season)
            procDict['Observations_season'] = season
            mybatch.add_batch(scriptref, procDict)

        # go for batch
        mybatch.go_batch()


def batch_WFD(theDict, scriptref='',
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
    reprocList = theDict['reprocList']
    sigmaInt = theDict['SN_sigmaInt']
    snrate = theDict['SN_z_rate']

    tag_list = pd.DataFrame()
    if 'None' not in reprocList:
        tag_list = pd.read_csv(reprocList)

    procDict = copy.deepcopy(theDict)

    del procDict['DD_list']
    del procDict['reprocList']
    # procDict['nside'] = 64
    # procDict['fieldType'] = 'WD'
    # procDict['Fitter_parnames'] = 'z,t0,x1,c,x0'
    tag_dir = '_spectroz'
    if 'z' in procDict['Fitter_parnames']:
        tag_dir = '_photz'
    procDict['OutputSimu_directory'] = '{}/{}/WFD{}'.format(outDir,
                                                            dbName, tag_dir)
    procDict['OutputFit_directory'] = procDict['OutputSimu_directory']
    # procDict['SN_NSNfactor'] = 30

    deltaRA = 10.

    RAs = np.arange(0., 360.+deltaRA, deltaRA)

    for RA in RAs[:-1]:
        RAmin = np.round(RA, 1)
        RAmax = RAmin+deltaRA
        RAmax = np.round(RAmax, 1)
        procName = 'WFD_{}_{}_{}{}_{}_{}'.format(
            dbName, RAmin, RAmax, tag_dir, np.round(sigmaInt, 2), snrate)
        sprocName = 'SN_WFD_{}_{}_{}{}'.format(dbName, RAmin, RAmax, tag_dir)
        mybatch = BatchIt(processName=procName, time=time, mem=mem)
        procDict['RAmin'] = RAmin
        procDict['RAmax'] = RAmax
        seasons = range(1, 11)
        if not tag_list.empty:
            idx = tag_list['ProductionID'] == sprocName
            sel = tag_list[idx]
            if len(sel) > 0:
                season_max = sel['season_max'].max()
                seasons = range(season_max, 11)

        for seas in seasons:
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
    procDict['SN_z_max'] = 0.7
    batch_WFD(procDict)
