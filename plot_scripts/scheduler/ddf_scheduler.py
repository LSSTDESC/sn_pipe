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


def get_seasons(obs, mjdCol='mjd'):

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

    dd = {}
    for b in 'ugrizy':
        idx = grp['band'] == b
        dd[b] = [len(grp[idx])]

    dd[mjdCol] = [grp[mjdCol].mean()]
    dd['season'] = [int(grp['season'].mean())]

    return pd.DataFrame.from_dict(dd)


def stat(grp, mjdCol='mjd'):

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

    df = pd.DataFrame.from_records(obs_season)
    df['night'] = df['mjd']-mjd0
    df['night'] = df['night'].astype(int)
    df['night'] += 1
    df['season'] = df['season'].astype(int)

    rstat_night = df.groupby(['target_name', 'night']).apply(
        lambda x: stat_night(x)).reset_index()

    print(rstat_night)

    rstat = rstat_night.groupby(['target_name', 'season']).apply(
        lambda x: stat(x)).reset_index()

    print(rstat)

    print(rstat.groupby(['target_name'])['nvisits'].sum().reset_index())
    """
    idx = obs_season['target_name'] == 'DD:COSMOS'
    plt.plot(obs_season[idx]["mjd"]-mjd0,
             obs_season[idx]["season"], 'ko', alpha=.1)
    plt.xlabel("night")
    plt.ylabel("season")
    plt.title("All DDFs")

    plt.show()
    """


def ana_ddf_grid(obs, mjdCol='mjd', mjd0=60980.):

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


mjd0 = 60980.0

# ddf_grid

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
fName = 'notebooks/observations_scheduler_orig.npy'

obs = np.load(fName)
ana_observation(obs)
"""
