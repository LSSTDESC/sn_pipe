from sn_tools.sn_obs import load_obs
import numpy as np
import pandas as pd
from sn_tools.sn_obs import season
import matplotlib.pyplot as plt


def filter_allocation(obs, Nvisits=800, outName='WL_req.csv'):
    """
    Function to estimate filter allocation corresponding to obs

    Parameters
    ----------
    obs : array
        data to process.
    Nvisits : int, optional
        Total number of visits. The default is 800.
    outName : str, optional
        output file name. The default is 'WL_req.csv'.

    Returns
    -------
    None.

    """

    ntot = len(obs)
    r = []
    for b in 'ugrizy':
        idx = obs['filter'] == b
        sel = obs[idx]
        b_alloc = len(sel)/ntot
        nv = np.round(b_alloc*Nvisits, 0)
        r.append((b, np.round(100.*b_alloc), nv))
        print(b, np.round(100.*b_alloc, 1), int(nv))

    df = pd.DataFrame(r, columns=['band', 'filter_allocation', 'Nvisits'])

    df.to_csv('WL_req.csv', index=False)


def get_DDF(obs):
    """
    Function to grab DDFs

    Parameters
    ----------
    obs : array
        all observations.

    Returns
    -------
    obs_DD : array
        DDF observations.

    """

    ido = np.core.defchararray.find(
        obs['note'].astype(str), 'DD')
    idob = np.ma.asarray(list(map(lambda x: bool(x+1), ido)))
    obs_DD = obs[idob]

    return obs_DD


def add_season(obs_DD):
    """
    method to estimate season on obs

    Parameters
    ----------
    obs_DD : array
        observations.

    Returns
    -------
    obs_season : array
        initial observations + season info (col).

    """

    obs_season = None
    for dd in np.unique(obs_DD['note']):
        idx = obs_DD['note'] == dd
        sel = obs_DD[idx]
        sel_DD = season(sel)
        if obs_season is None:
            obs_season = sel_DD
        else:
            obs_season = np.concatenate((sel_DD, obs_season))

    return obs_season


def stat_m5(obs):

    obs_c = pd.DataFrame.from_records(obs)

    # plot_hist_band(obs_c, 'airmass')
    # plot_hist_band(obs_c, 'fiveSigmaDepth')

    idx = obs_c['airmass'] <= 2.
    obs_c = obs_c[idx]
    res = obs_c.groupby(['note', 'season', 'filter'])[[
        'fiveSigmaDepth', 'airmass']].median().reset_index()

    plot(res)

    print(res)

    print(test)


def plot(res):

    for dd in res['note'].unique():
        idx = res['note'] == dd
        sela = res[idx]
        bands = sela['filter'].unique()
        fig, ax = plt.subplots()
        fig.suptitle(dd)
        for b in bands:
            io = sela['filter'] == b
            selb = sela[io]
            ax.plot(selb['season'], selb['fiveSigmaDepth'],
                    label='${}$-band'.format(b))
            print(dd, b, selb['fiveSigmaDepth'].median())
        ax.grid()
        ax.legend(loc='best')
        ax.set_xlabel(r'season')
        ax.set_ylabel(r'5$\sigma$ depth [mag]')
    plt.show()


def plot_hist_band(resa, var):

    idf = resa['note'] == 'DD:XMM_LSS'
    res = resa[idf]

    for dd in res['note'].unique():
        idx = res['note'] == dd
        sela = res[idx]
        bands = sela['filter'].unique()

        for b in bands:
            fig, ax = plt.subplots()
            fig.suptitle('{} - ${}$-band'.format(dd, b))
            io = sela['filter'] == b
            selb = sela[io]
            ax.hist(selb[var], histtype='step')

        ax.set_xlabel(r'{}'.format(var))
        ax.set_ylabel(r'Number of Entries')
        plt.show()


def get_m5(obs_season):

    stat_m5(obs_season)

    fig, ax = plt.subplots()
    idx = obs_season['note'] == 'DD:COSMOS'
    sel = obs_season[idx]
    vvar = 'observationStartMJD'
    # vvar = 'night'
    vary = 'fiveSigmaDepth'
    vary = 'sky'
    vary = 'seeingFwhmEff'
    vary = 'airmass'
    for b in 'ugrizy':
        ido = sel['filter'] == b
        selb = sel[ido]
        ax.plot(selb[vvar], selb[vary],
                linestyle='None', marker='o', label='${}$-band'.format(b))
        idf = selb['airmass'] <= 2.0
        print(b, np.median(selb['fiveSigmaDepth']),
              np.median(selb[idf]['fiveSigmaDepth']))
        """
        figb, axb = plt.subplots()
        for night in np.unique(sel['night']):
            print('night', night)
            io = sel['night'] == night
            ssel = sel[io]
            axb.plot(ssel[vvar], ssel[vary], 'ko')
            plt.show()
        """
    ax.legend()
    plt.show()

    # estimate m5 per DD and per season

    obs_df = pd.DataFrame.from_records(obs_season)
    idx = obs_df['airmass'] <= 1.2
    obs_df = obs_df[idx]
    var = 'fiveSigmaDepth'
    tt = obs_df.groupby(['note', 'season', 'filter']).apply(
        lambda x: pd.DataFrame({var: [x[var].median()]})).reset_index()

    for dd in np.unique(tt['note']):
        idx = tt['note'] == dd
        selp = tt[idx]
        fig, ax = plt.subplots()
        fig.suptitle('{}'.format(dd))
        for b in np.unique(selp['filter']):
            idxb = selp['filter'] == b
            selb = selp[idxb]
            print('aoo', selb)
            ax.plot(selb['season'], selb['fiveSigmaDepth'],
                    linestyle='None', marker='o', label='${}$-band'.format(b))

        ax.legend()
    plt.show()

    print(tt)


def cadence_sl(obs):

    print(obs['night'])

    df = pd.DataFrame.from_records(obs)

    res = df.groupby(['note', 'season', 'night'])[
        'observationStartMJD'].mean().reset_index()

    resb = res.groupby(['note', 'season'])[
        'observationStartMJD'].diff().reset_index()

    print(resb)


dbDir = '../DB_Files'
dbName = 'baseline_v3.0_10yrs'
dbExtens = 'db'

"""
dbName = 'draft_connected_v2.99_10yrs'
dbName = 'ddf_early_deep_slf0.20_f10.60_f20.80_v2.1_10yrs'
dbExtens = 'npy'
"""

# load observations
observations = load_obs(dbDir, dbName, dbExtens)

# select DDF and get seasons
obs_season = add_season(get_DDF(observations))

# filter_allocation(observations)

# get_m5(observations)

cadence_sl(obs_season)
