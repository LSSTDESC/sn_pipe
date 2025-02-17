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

    return ddf


def analyze_simu_exp(data):

    # add new columns as diff

    for vv in ['nvisits', 'u', 'g', 'r', 'i', 'z', 'y']:
        data['diff_{}'.format(vv)] = data['{}_exp'.format(vv)] - data[vv]

    print(data)

    res_stat = data.groupby(['target_name', 'season']).apply(
        lambda x: stat_simu_exp(x))

    print(res_stat)


def stat_simu_exp(grp):

    dict_frac = {}
    nnights = len(grp)
    for vv in ['nvisits', 'u', 'g', 'r', 'i', 'z', 'y']:
        myvar = 'diff_{}'.format(vv)
        idxa = grp[myvar] >= 1
        idxb = grp[myvar] <= -1
        idxc = np.abs(grp[myvar]) < 0.5
        dict_frac['{}_missing'.format(vv)] = [len(grp[idxa])/nnights]
        dict_frac['{}_excess'.format(vv)] = [len(grp[idxb])/nnights]
        dict_frac['{}_perfect'.format(vv)] = [len(grp[idxc])/nnights]

    rr = pd.DataFrame.from_dict(dict_frac)

    return rr


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

opts, args = parser.parse_args()
# Load parameters
dbDir = opts.dbDir
dbName = opts.dbName
ddf_list = opts.ddf_list.split(',')
configDir = opts.configDir
configName = opts.configName

fName = 'test_merge.hdf5'
"""
res = merge_exp_simu(dbDir, dbName, configDir, configName)
print(res)
res.to_hdf('test_merge.hdf5', key='ddf')
"""

data = pd.read_hdf(fName)

analyze_simu_exp(data)
