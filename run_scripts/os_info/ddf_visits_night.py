#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 08:52:23 2025

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import numpy as np
import pandas as pd
import yaml
from sn_tools.sn_obs import season
from sn_tools.sn_io import checkDir


def process_night(grp):
    """
    Function to process an observing night

    Parameters
    ----------
    grp : pandas df
        Data to process.

    Returns
    -------
    res : pandas df
        Processed data.

    """

    bands = 'ugrizy'

    resdict = {}

    resdict['nvisits'] = [len(grp)]
    resdict['season'] = [int(grp['season'].mean())]
    resdict['moonPhase'] = [int(grp['moonPhase'].mean())]
    for b in bands:
        idx = grp['band'] == b
        sel = grp[idx]
        resdict[b] = [len(sel)]

    seq = []
    count = []
    grp = grp.sort_values(by='mjd')

    cb = 0
    for i, row in grp.iterrows():
        band = row['band']
        if len(seq) == 0:
            seq = [band]
        if band != seq[-1]:
            seq.append(band)
            count.append(cb)
            cb = 0
        cb += 1
    # add the last one
    count.append(cb)
    resdict['seq'] = [''.join(seq)]

    count = list(map(str, count))
    resdict['seq_visits'] = ['/'.join(count)]

    res = pd.DataFrame.from_dict(resdict)

    return res


def get_survey(survey, ddf_list=['DD:COSMOS']):
    """
    Function to build the expected survey

    Parameters
    ----------
    survey : dict
        config parameters.
    ddf_list : list(str), optional
        List of DDFs to process. The default is ['DD:COSMOS'].

    Returns
    -------
    res : pandas df
        output resu.

    """

    configFile = survey['fName']
    # load config
    df_conf = pd.read_csv(configFile, comment='#')

    print(df_conf)

    res = pd.DataFrame()
    for ddf in ddf_list:
        name = ddf.split(':')[-1]
        # get fitleType
        fieldType = survey[name].split(',')[0]
        idx = df_conf['target'] == fieldType
        rr = pd.DataFrame(df_conf[idx])
        rr['target_name'] = ddf
        rr['DD_type'] = fieldType
        res = pd.concat((res, rr))

    res['nvisits'] = res[list('ugrizy')].sum(axis=1)
    res = res.drop(columns=['target'])
    return res


def merge_exp_simu(dbDir, dbName, configDir, configName):
    """
    Function to merge simu data and expected os parameters

    Parameters
    ----------
    dbDir : str
        OS dir.
    dbName : str
        OS  name.
    configDir : str
        dir for config expected params.
    configName : str
        name of the expected config file.

    Returns
    -------
    ddf : pandas df
        merged dataframe.

    """

    # get expected values
    config = '{}/{}.yaml'.format(configDir, configName)
    ddf_exp = yaml.load(open(config), Loader=yaml.FullLoader)
    exp_survey = get_survey(ddf_exp, ddf_list=ddf_list)

    # get simu data
    fName = '{}/{}.npy'.format(dbDir, dbName)

    obs = np.load(fName)
    idx = np.in1d(obs['target_name'], ddf_list)
    obs = obs[idx]
    obs = season(obs, season_gap=20., mjdCol='mjd')

    obs = pd.DataFrame.from_records(obs)

    dd = obs.groupby(['target_name', 'night']).apply(
        lambda x: process_night(x)).reset_index()

    # merge dfs

    ddf = dd.merge(exp_survey, left_on=['target_name', 'season'], right_on=[
                   'target_name', 'season'], suffixes=['', '_exp'])

    # adjustements related to moonPhase
    idx = ddf['moonPhase'] <= 40
    sela = pd.DataFrame(ddf[idx])
    selb = pd.DataFrame(ddf[~idx])

    sela['y_exp'] = 0
    selb['u_exp'] = 0

    ddf_moon = pd.concat((sela, selb))

    r = []
    for b in 'ugrizy':
        r.append('{}_exp'.format(b))

    ddf_moon['nvisits_exp'] = ddf_moon[r].sum(axis=1)

    return ddf_moon


parser = OptionParser(
    description='Script to study DDF visits on a nightly basis from pointings')

parser.add_option("--dbDir", type="str",
                  default='../DB_Files',
                  help="file directory [%default]")
parser.add_option("--dbName", type="str",
                  default='desc_ddf_v4.2.1_10yrs',
                  help="OS name [%default]")
parser.add_option("--ddf_list", type="str",
                  default="DD:COSMOS,DD:ECDFS,DD:EDFS_a,DD:EDFS_b,DD:ELAISS1,DD:XMM_LSS",
                  help="list of ddf [%default]")
parser.add_option("--configDir", type="str",
                  default="input/scheduler",
                  help="input config dir [%default]")
parser.add_option("--configName", type="str",
                  default="ddf_desc_0.70_sn",
                  help="input config name [%default]")
parser.add_option("--outDir", type="str",
                  default="../ddf_visits_night",
                  help="output directory [%default]")


opts, args = parser.parse_args()
# Load parameters
dbDir = opts.dbDir
dbName = opts.dbName
ddf_list = opts.ddf_list.split(',')
configDir = opts.configDir
configName = opts.configName
outDir = opts.outDir

checkDir(outDir)

fName = '{}/{}.hdf5'.format(outDir, dbName)

res = merge_exp_simu(dbDir, dbName, configDir, configName)
print(res)
res.to_hdf(fName, key='ddf')
