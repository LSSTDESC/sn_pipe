from optparse import OptionParser
from sn_tools.sn_io import make_dict_from_config
from sn_tools.sn_io import add_parser, checkDir
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
    smearFlux = theDict['SN_smearFlux']

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
    # if 'z' in procDict['Fitter_parnames']:
    if procDict['Fitter_sigmaz'] >= 1.e-3:
        tag_dir = '_photz'
    procDict['OutputSimu_directory'] = '{}/{}/DDF{}'.format(outDir,
                                                            dbName, tag_dir)
    checkDir(procDict['OutputSimu_directory'])
    procDict['OutputFit_directory'] = procDict['OutputSimu_directory']
    # procDict['SN_NSNfactor'] = 30
    procDict['Pixelisation_nside'] = procDict['nside']

    simu_fromFile = procDict['simuParams_fromFile']
    simuParams_dir = procDict['simuParams_dir']
    procDict.pop('simuParams_fromFile')
    procDict.pop('simuParams_dir')

    for fieldName in DD_list:
        procDict['fieldName'] = fieldName
        procName = 'DD_{}_{}{}_{}_{}_{}'.format(
            dbName, fieldName, tag_dir, np.round(sigmaInt, 2),
            snrate, smearFlux)
        mybatch = BatchIt(processName=procName, time=time, mem=mem)
        seasons = range(1, 14)
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

            if simu_fromFile == 1:
                ffi = 'SN_simu_params_DDF_{}_season_{}.hdf5'.format(
                    dbName, season)
                outDir_simuparams = '{}_simuparams'.format(simuParams_dir)
                procDict['SN_simuFile'] = '{}/{}/DDF_spectroz/{}'.format(
                    outDir_simuparams, dbName, ffi)

            mybatch.add_batch(scriptref, procDict)

        # go for batch
        mybatch.go_batch()


def batch_WFD(theDict, scriptref='run_scripts/sim_to_fit/run_sim_to_fit.py',
              time='60:00:00', mem='40G', seas_min=1, seas_max=10,
              zmin=0.01, zmax=0.7, runMode='seasonal', splitobs=False,
              extend_rate=True):
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
    seas_min: int, optional
        min season of obs. The default is 1.
    seas_max: int, optional
        max season of obs. The default is 10.


    Returns
    -------
    None.

    """

    dbName = theDict['dbName']
    dbExtens = theDict['dbExtens']
    outDir = theDict['OutputSimu_directory']
    reprocList = theDict['reprocList']
    sigmaInt = theDict['SN_sigmaInt']
    snrate = theDict['SN_z_rate']
    zmax = np.round(zmax, 2)
    zmin = np.round(zmin, 2)
    smearFlux = theDict['SN_smearFlux']
    sat = theDict['saturation_effect']
    sat_psf = theDict['saturation_psf']
    ccdfullwell = theDict['saturation_ccdfullwell']

    tag_list = pd.DataFrame()
    if 'None' not in reprocList:
        tag_list = pd.read_csv(reprocList)

    procDict = copy.deepcopy(theDict)

    del procDict['DD_list']
    del procDict['reprocList']
    # procDict['nside'] = 64
    # procDict['fieldType'] = 'WD'
    # procDict['Fitter_parnames'] = 'z,t0,x1,c,x0'
    satb = 'nosat'
    if sat == 1:
        satb = 'sat_{}_{}'.format(sat_psf, int(ccdfullwell))
    tag_dir = '_spectroz_{}'.format(satb)
    # if 'z' in procDict['Fitter_parnames']:
    if procDict['Fitter_sigmaz'] >= 1.e-3:
        tag_dir = '_photz'

    procDict['OutputSimu_directory'] = '{}/{}/WFD{}'.format(outDir,
                                                            dbName, tag_dir)
    procDict['OutputFit_directory'] = procDict['OutputSimu_directory']
    # procDict['SN_NSNfactor'] = 30

    simu_fromFile = procDict['simuParams_fromFile']
    simuParams_dir = procDict['simuParams_dir']
    procDict.pop('simuParams_fromFile')
    procDict.pop('simuParams_dir')

    deltaRA = 10.

    RAs = np.arange(0., 360.+deltaRA, deltaRA)

    seasons = range(seas_min, seas_max+1)
    for RA in RAs[:-1]:
        RAmin = np.round(RA, 1)
        RAmax = RAmin+deltaRA
        RAmax = np.round(RAmax, 1)
        procName = 'WFD_{}_{}_{}{}_{}_{}'.format(
            dbName, RAmin, RAmax, tag_dir, np.round(sigmaInt, 2), snrate)
        procNamea = '{}_{}_{}_{}_{}'.format(
            procName, seas_min, seas_max, zmin, zmax)
        sprocName = 'SN_WFD_{}_{}_{}{}_{}'.format(dbName, RAmin, RAmax,
                                                  tag_dir, smearFlux)
        mybatch = BatchIt(processName=procNamea, time=time, mem=mem)
        procDict['RAmin'] = RAmin
        procDict['RAmax'] = RAmax
        procDict['SN_z_min'] = zmin
        procDict['SN_z_max'] = zmax
        procDict['SN_z_minsimu'] = zmin
        procDict['SN_z_maxsimu'] = zmax
        if extend_rate:
            procDict['SN_z_minsimu'] = 0.01
            procDict['SN_z_maxsimu'] = 0.8

        if dbExtens == 'npy' and splitobs:
            procDict['dbName'] = '{}_{}_{}'.format(dbName, RAmin, RAmax)

        if not tag_list.empty:
            idx = tag_list['ProductionID'] == sprocName
            sel = tag_list[idx]
            if len(sel) > 0:
                season_max = sel['season_max'].max()
                seasons = range(season_max, 11)

        if runMode == 'seasonal':

            for seas in seasons:
                tttag = 'SN_{}_{}_{}_{}'.format(procName, seas, zmin, zmax)
                procDict['ProductionIDSimu'] = tttag
                procDict['Observations_season'] = seas

                if simu_fromFile == 1:
                    suffix = '{}_{}'.format(zmin, zmax)
                    ffi = 'SN_simu_params_WFD_{}_season_{}_{}.hdf5'.format(
                        dbName, seas, suffix)
                    outDir_simuparams = '{}_simuparams'.format(simuParams_dir)
                    procDict['SN_simuFile'] = '{}/{}/WFD_spectroz/{}'.format(
                        outDir_simuparams, dbName, ffi)

                mybatch.add_batch(scriptref, procDict)

        else:

            tttag = 'SN_{}_{}_{}_{}_{}'.format(
                procName, seas_min, seas_max, zmin, zmax)
            procDict['ProductionIDSimu'] = tttag
            procDict['Observations_season'] = '{}-{}'.format(
                seas_min, seas_max)

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
    batch_DDF(procDict, mem='20Gb')

# this is for WFD
# procDict['simuParams_fromFile'] = opts.simuParams_fromFile

seasons = [(1, 5), (6, 10), (11, 14)]
seasons = [(1, 7), (7, 14)]
runMode = 'all_seasons'
if opts.fieldType == 'WFD':
    for seas in seasons:
        if opts.SN_z_max >= 0.4:
            batch_WFD(procDict,
                      seas_min=seas[0], seas_max=seas[1],
                      zmin=0.01, zmax=0.4,
                      mem='20Gb', runMode=runMode)
            batch_WFD(procDict,
                      seas_min=seas[0], seas_max=seas[1],
                      zmin=0.4, zmax=0.8,
                      mem='20Gb', runMode=runMode)
        else:
            batch_WFD(procDict,
                      seas_min=seas[0], seas_max=seas[1],
                      zmin=0.01, zmax=opts.SN_z_max,
                      mem='20Gb', runMode=runMode, extend_rate=False)

        """
        batch_WFD(procDict,
                  seas_min=seas[0], seas_max=seas[1],
                  zmin=0.6, zmax=0.8,
                  mem='20Gb', runMode=runMode)
        """
