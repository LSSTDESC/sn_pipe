#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 09:06:51 2025

@author: philippe.gris@clermont.in2p3.fr
"""

from sn_tools.sn_obs import season
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from optparse import OptionParser


def get_seasons(obs, mjdCol='mjd'):
    """
    Function to estimate the season for each field

    Parameters
    ----------
    obs : numpy array
        Data to process.
    mjdCol : str, optional
        column name for season estimation. The default is 'mjd'.

    Returns
    -------
    obs_season : numpy array
        Original array+season col.

    """

    targets = np.unique(obs['target_name'])

    obs_season = None
    for target in targets:
        idx = obs['target_name'] == target
        sel = obs[idx]
        selb = season(sel, mjdCol=mjdCol)
        if obs_season is None:
            obs_season = selb
        else:
            obs_season = np.concatenate((obs_season, selb))

    return obs_season


def stat_night(grp, mjdCol='mjd'):
    """
    Function to estimate the stat per night

    Parameters
    ----------
    grp : pandas df
        Data to process.
    mjdCol : str, optional
        column name. The default is 'mjd'.

    Returns
    -------
    pandas df
        stat for the night.

    """

    dd = {}
    for b in 'ugrizy':
        idx = grp['band'] == b
        dd[b] = [len(grp[idx])]

    dd[mjdCol] = [grp[mjdCol].mean()]
    dd['season'] = [int(grp['season'].mean())]

    return pd.DataFrame.from_dict(dd)


def stat_season(grp, mjdCol='mjd'):
    """
    Function to grab stat per season (cadence, gaps, ...)

    Parameters
    ----------
    grp : pandas df
        Data to process.
    mjdCol : str, optional
        mjd col name. The default is 'mjd'.

    Returns
    -------
    pandas df
        df with stat.

    """

    grp = grp.sort_values(by=[mjdCol])

    diff = grp[mjdCol].diff()
    cad = diff.mean()
    gap_max = diff.max()
    gap_min = diff.min()

    dict_out = {}
    dict_out['cad'] = [cad]
    dict_out['gap_max'] = [gap_max]
    dict_out['gap_min'] = [gap_min]
    dict_out['season_length'] = [grp[mjdCol].max()-grp[mjdCol].min()]

    if 'u' in grp.columns:
        nv = 0
        for b in 'ugrizy':
            dict_out[b] = [grp[b].sum()]
            nv += grp[b].sum()

        dict_out['nvisits'] = [nv]
    dict_out['nnight'] = [len(grp)]
    return pd.DataFrame.from_dict(dict_out)


def ana_observation(obs, mjd0=60980.):

    # get seasons

    obs_season = get_seasons(obs, mjdCol='mjd')
    print(obs_season.dtype.names)

    # add night number
    df = pd.DataFrame.from_records(obs_season)
    df['night'] = df['mjd']-mjd0
    df['night'] = df['night'].astype(int)
    df['night'] += 1
    df['season'] = df['season'].astype(int)

    # get stat per night
    rstat_night = df.groupby(['target_name', 'night']).apply(
        lambda x: stat_night(x)).reset_index()

    print(rstat_night)

    rstat = rstat_night.groupby(['target_name', 'season']).apply(
        lambda x: stat_season(x)).reset_index()

    print(rstat.groupby(['target_name'])['nvisits'].sum().reset_index())

    print(rstat['nvisits'].sum())
    print(rstat[['target_name', 'season', 'season_length', 'cad']])

    plot_res(rstat, vary='season_length')
    plot_res(rstat, vary='cad', legy='cadence [day-1]')
    plt.show()
    """
    idx = obs_season['target_name'] == 'DD:COSMOS'
    plt.plot(obs_season[idx]["mjd"]-mjd0,
             obs_season[idx]["season"], 'ko', alpha=.1)
    plt.xlabel("night")
    plt.ylabel("season")
    plt.title("All DDFs")

    plt.show()
    """


def plot_res(rstat, varx='season', legx='season', vary='season_length', legy='season_length'):

    targets = rstat['target_name'].unique()

    fig, ax = plt.subplots()
    for tt in targets:
        idx = rstat['target_name'] == tt
        sel = rstat[idx]
        ax.plot(sel[varx], sel[vary], label=tt)

    ax.grid(visible=True)
    ax.set_xlabel(r'{}'.format(legx))
    ax.set_ylabel(r'{}'.format(legy))
    ax.legend()


def ana_ddf_grid(obs, mjdCol='mjd', mjd0=60980.):
    """
    Function to analyze a file in rubin_scheduler

    Parameters
    ----------
    obs : array
        Data to process.
    mjdCol : str, optional
        MJD col name. The default is 'mjd'.
    mjd0 : float, optional
        MJD start of the survey. The default is 60980..

    Returns
    -------
    None.

    """

    df = pd.DataFrame.from_records(obs)
    print('rrr', df[mjdCol])
    df['night'] = df[mjdCol]-mjd0
    df['night'] = df['night'].astype(int)
    df['night'] += 1

    df = df.fillna(-1)
    print(df)
    idx = df['COSMOS_airmass'] >= 1
    idx &= df['COSMOS_airmass'] <= 3.
    seldf = df[idx]

    sel_seas = season(seldf.to_records(index=False), mjdCol=mjdCol)
    sel_seas = pd.DataFrame.from_records(sel_seas)
    print(sel_seas)

    rstat = sel_seas.groupby(['season']).apply(lambda x: stat(x)).reset_index()

    print(rstat)

    print(test)
    fig, ax = plt.subplots()

    # ax.plot(df['mjd'], df['COSMOS_airmass'], 'k.')
    idx = df['COSMOS_airmass'] >= 1
    idx &= df['COSMOS_airmass'] <= 3.
    ax.plot(df[idx]['mjd']-mjd0, df[idx]['COSMOS_airmass'], 'k.')
    # ax.hist(df[idx]['COSMOS_airmass'], histtype='step')

    plt.show()


def make_full_survey(obs):

    bands = 'ugrizy'
    r = []
    for i, row in obs.iterrows():
        for b in bands:
            for nvisit in range(row[b]):
                r.append((row['target'], row['mjd'], b))

    res = np.rec.fromrecords(r, names=['target_name', 'mjd', 'band'])

    return res


parser = OptionParser(
    description='Script to analyze the file produced for the LSST scheduler')

parser.add_option('--dirFiles', type=str,
                  default='../desc_ddf_deep_rolling',
                  help='Location dir of the ddf obs file [%default]')
parser.add_option("--ddf_survey", type=str, default='ddf_desc_0.70_sn',
                  help="survey to analyze [%default]")
parser.add_option("--mjd_min", type=int, default=60980,
                  help="survey start [%default]")

opts, args = parser.parse_args()

dirFiles = opts.dirFiles
ddf_survey = opts.ddf_survey
mjd0 = opts.mjd_min

# ddf_grid
"""
fName = '../../rubin_sim_data/scheduler/ddf_grid.npz'

ddf_data = np.load(fName)
ddf_grid = ddf_data["ddf_grid"].copy()
ddf_data.close()
print(ddf_grid.dtype.names)
indx = np.where(ddf_grid["mjd"] < mjd0)[0].max()
ddf_grid = ddf_grid[indx:]
ana_ddf_grid(ddf_grid, mjd0=mjd0)

"""
# observations
fName = '{}/{}.npy'.format(dirFiles, ddf_survey)

obs = np.load(fName)

"""
fName = 'ddf_desc_0.70_sn.hdf5'
obs = pd.read_hdf(fName)

obs = obs.drop(columns=['season'])
obs['target_name'] = obs['target']

obs = make_full_survey(obs)
"""
ana_observation(obs, mjd0=mjd0)
