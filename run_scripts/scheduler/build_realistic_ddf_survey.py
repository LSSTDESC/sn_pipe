#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 15:24:13 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sn_tools.sn_obs import season


def load_data(dirFiles):
    """
    Fonction to load files of dirFiles

    Parameters
    ----------
    dirFiles : str
        location dir of the files.

    Returns
    -------
    ddf : pandas df
        data from the files.

    """

    fis = glob.glob('{}/*.hdf5'.format(dirFiles))

    ddf = pd.DataFrame()
    for fi in fis:
        dd = pd.read_hdf(fi)
        ddf = pd.concat((ddf, dd))

    return ddf


def get_survey(field, survey, field_type='UD'):
    """
    Function to grab the survey corresponding to a field

    Parameters
    ----------
    field : str
        field name.
    survey : pandas df
        survey.
    field_type : str, optional
        Type of field. The default is 'UD'.

    Returns
    -------
    dfb : pandas df
        survey corresponding to the field.

    """

    idx = survey['target'] == field_type
    dfb = pd.DataFrame(survey[idx])
    dfb['target'] = field

    return dfb


def process(ddf_scheduler, survey, field_type):
    """
    Function to process the targets

    Parameters
    ----------
    ddf_scheduler : pandas df
        observations.
    survey : pandas df
        survey to consider.
    field_type : dict
        field types.

    Returns
    -------
    res : pandas df
        output schedule.

    """

    targets = ddf_scheduler['target'].unique()

    res = pd.DataFrame()
    for tt in targets:
        print('processing', tt)
        idx = ddf_scheduler['target'] == tt
        target_sched = ddf_scheduler[idx]
        # get corresponding survey
        target_survey = get_survey(tt, survey, field_type[tt])
        # process
        dd = process_target(target_sched, target_survey)

        res = pd.concat((res, dd))

    return res


def process_target(target_sched, target_survey):
    """
    Function to process a target

    Parameters
    ----------
    target_sched : pandas df
        Data to process
    target_survey : pandas df
        survey to use.

    Returns
    -------
    target_sched : pandas df
        processed target data.

    """

    # grab seasons
    seas = season(target_sched.to_records(index=False), mjdCol='mjd')

    target_sched = pd.DataFrame.from_records(seas)

    # match seasons with number of visits
    target_sched = target_sched.merge(target_survey,
                                      left_on=['target', 'season'],
                                      right_on=['target', 'season'],
                                      suffixes=['', ''])

    if 'index' in target_sched.columns:
        target_sched = target_sched.drop(['index'])
    # select target
    target_sched = select_target(target_sched)

    return target_sched


def select_target(target_sched):
    """
    Function to select the target

    Parameters
    ----------
    target_sched : pandas df
        Data to process.

    Returns
    -------
    tt : pandas df
        selected df.

    """

    # select obs with sufficient observing time
    bands = 'ugrizy'
    target_sched['obs_time_survey [h]'] = target_sched[list(bands)].sum(axis=1)
    target_sched['obs_time_survey [h]'] *= 30./3600.
    # print(target_sched.columns)
    idx = target_sched['obs_duration [h]'] >= target_sched['obs_time_survey [h]']
    sel_target = pd.DataFrame(target_sched[idx])
    print(len(sel_target)/len(target_sched))
    if 'index' in sel_target.columns:
        sel_target = sel_target.drop(['index'])
    # reduce season length to 180 days
    tt = sel_target.groupby(['target', 'season']).apply(
        lambda x: reduce_season_length(x)).reset_index(drop=True)

    return tt


def reduce_season_length(grp, mjdCol='mjd', sl_max=200.):
    """
    Function to reduce the number of observations acdcording to season length

    Parameters
    ----------
    grp : pandas df
        Data to process.
    mjdCol : str, optional
        col name to estimate season length. The default is 'mjd'.
    sl_max : float, optional
        max season length. The default is 180..

    Returns
    -------
    res : pandas df
        obs corresponding to the reduced season length.

    """

    # get season length
    mjd_min = grp[mjdCol].min()
    mjd_max = grp[mjdCol].max()

    season_length = mjd_max-mjd_min

    if season_length < sl_max:
        res = pd.DataFrame(grp)
    else:
        mjd_season = mjd_min+sl_max
        idx = grp['mjd'] <= mjd_season
        res = pd.DataFrame(grp[idx])

    return res


parser = OptionParser(description='Script to build a realistic ddf survey')

parser.add_option('--dirFiles', type=str,
                  default='../ddf_scheduler',
                  help='Location dir of the ddf summary file [%default]')
parser.add_option('--ddf_rubin_scheduler_data', type=str,
                  default='../../rubin_sim_data/scheduler/ddf_grid.npz',
                  help='Location dir of the ddf summary file [%default]')
parser.add_option("--mjd_min", type=int, default=60980,
                  help="survey start [%default]")
parser.add_option("--survey", type=str, default='input/scheduler/ddf_desc_0.70_sn.csv',
                  help="survey to implement [%default]")
parser.add_option("--udf", type=str, default='COSMOS,XMM_LSS',
                  help="ultra-deep fields [%default]")
parser.add_option("--ddf", type=str, default='ELAISS1,ECDFS,EDFS_a,EDFS_b',
                  help="deep fields [%default]")

opts, args = parser.parse_args()

dirFiles = opts.dirFiles
ddf_rubin_scheduler_data = opts.ddf_rubin_scheduler_data
mjd_min = opts.mjd_min
survey = opts.survey
udfs = opts.udf.split(',')
ddfs = opts.ddf.split(',')

# set a dict for field types
field_type = {}

for vv in udfs:
    field_type[vv] = 'UD'

for vv in ddfs:
    field_type[vv] = 'DF'


# load data

ddf_scheduler = load_data(dirFiles)

# load survey
survey_df = pd.read_csv(survey, comment='#', index_col=False)

res_survey = process(ddf_scheduler, survey_df, field_type)

print(res_survey.columns)


outName = '{}.hdf5'.format(survey.split('/')[-1].split('.csv')[0])
res_survey.to_hdf(outName, key='ddf')
"""
# load rubin scheduler data
ddf_data = np.load(ddf_rubin_scheduler_data)
ddf_grid = pd.DataFrame.from_records(ddf_data["ddf_grid"].copy())
ddf_data.close()

print(ddf_grid.columns)
idx = ddf_grid['mjd'] >= mjd_min
ddf_grid = ddf_grid[idx]
ddf_grid['night'] = ddf_grid['mjd']-mjd_min+1
ddf_grid['night'] = ddf_grid['night'].astype(int)

fig, ax = plt.subplots()
ax.plot(ddf_grid['night'], ddf_grid['COSMOS_m5_g'], 'k.')

plt.show()
"""
