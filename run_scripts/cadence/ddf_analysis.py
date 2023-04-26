import numpy as np
import pandas as pd
from sn_tools.sn_obs import season
from sn_tools.sn_stacker import CoaddStacker
import numpy.lib.recfunctions as rf
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def load_DDF(dbDir, dbName, DDList=['COSMOS', 'ECDFS',
                                    'EDFS, a', 'EDFS, b',
                                    'ELAISS1', 'XMM-LSS']):
    """
    Method to load DDFs

    Parameters
    ----------
    dbDir : str
        location dir of the database.
    dbName : str
        db name (OS) to load.
    DDList : list(str), optional
        list of DDFs to consider. The default is ['COSMOS', 'ECDFS',
                                                  'EDFS, a', 'EDFS, b',
                                                  'ELAISS1', 'XMM-LSS'].

    Returns
    -------
    data : array
        DDF observations.

    """

    fullPath = '{}/{}'.format(dbDir, dbName)
    tt = np.load(fullPath)

    print(np.unique(tt['note']))
    data = None
    for field in DDList:
        idx = tt['note'] == 'DD:{}'.format(field)
        if data is None:
            data = tt[idx]
        else:
            data = np.concatenate((data, tt[idx]))

    return data


def filter_alloc(grp):
    """
    Function to estimate the number of visits per obs. night

    Parameters
    ----------
    grp : pandas df
        data to process.

    Returns
    -------
    pandas df
        resulting df.

    """

    filters = grp['filter'].unique()

    dictout = {}
    Nvisits_moon = 0
    Nvisits_no_moon = 0
    Nvisits = 0
    for b in 'ugrizy':
        idx = grp['filter'] == b
        sel = grp[idx]
        if len(sel) > 0:
            nv = np.sum(sel['exptime'])/30.
            dictout[b] = [nv]
            Nvisits += nv

    io = grp['filter'] == 'u'
    if len(grp[io]) == 0:
        Nvisits_moon = Nvisits
        Nvisits_no_moon = 0
    else:
        Nvisits_moon = 0.
        Nvisits_no_moon = Nvisits

    dictout['Nvisits_no_moon'] = Nvisits_no_moon
    dictout['Nvisits_moon'] = Nvisits_moon

    return pd.DataFrame.from_dict(dictout)


def cadence(grp):
    """
    Function to estimate the cadence

    Parameters
    ----------
    grp : pandas df
        data to process.

    Returns
    -------
    pandas df
        df with cadence estimation.

    """

    dictout = {}
    for b in 'ugrizy':

        idx = grp['filter'] == b
        sel = grp[idx]
        sel = sel.sort_values(by=['mjd'])
        dictout[b] = [sel['mjd'].diff().median()]

    return pd.DataFrame.from_dict(dictout)


def stackIt(data):
    """
    Function to stack data (per night/band)

    Parameters
    ----------
    data : numpy array
        data to stack.

    Returns
    -------
    dd : pandas df
        Stacked data.

    """

    stacker = CoaddStacker(col_sum=['exptime', 'numExposures'],
                           col_mean=['mjd', 'ra', 'dec', 'season',
                                     'airmass', 'fiveSigmaDepth'],
                           col_median=['sky', 'moonPhase'],
                           col_group=['filter', 'night'],
                           col_coadd=['fiveSigmaDepth', 'exptime'])

    dd = None

    for field in np.unique(data['note']):
        idx = data['note'] == field
        sel = data[idx]
        datab = stacker._run(sel)
        datab = rf.append_fields(
            datab, 'note', [field]*len(datab), dtypes='<U11')
        if dd is None:
            dd = datab
        else:
            dd = np.concatenate((dd, datab))

    return dd


def plot_cadence(df, varx, vary, fig=None, ax=None, label=''):
    """
    Function to make plots

    Parameters
    ----------
    df : pandas df
        data to plot.
    varx : str
        x-axis variable.
    vary : str
        y-axis var.
    fig : matplotlib figure, optional
        Figure for the plot. The default is None.
    ax : matplotlib axis, optional
        acis for the figure. The default is None.
    label : str, optional
        label. The default is ''.

    Returns
    -------
    None.

    """

    if fig is None:
        fig, ax = plt.subplots()

    ax.plot(df[varx], df[vary], label=label)


dbDir = '../DB_Files'
dbName = 'baseline_v2.0_10yrs.npy'

# load data
data = load_DDF(dbDir, dbName)

# get seasons

data_seas = None
for field in np.unique(data['note']):
    idx = data['note'] == field
    sel = data[idx]
    selb = season(sel, mjdCol='mjd')
    if data_seas is None:
        data_seas = selb
    else:
        data_seas = np.concatenate((data_seas, selb))

data_seas = rf.append_fields(data_seas, 'filter', data_seas['band'])

data_season = stackIt(data_seas)

print(data_season)

df = pd.DataFrame.from_records(data_season)


dd = df.groupby(['note', 'night', 'season']).apply(
    lambda x: filter_alloc(x)).reset_index()

for bb in ['Nvisits_no_moon', 'Nvisits_moon']:
    idx = dd[bb] > 0

    gg = dd[idx].groupby(['note'])[[bb]].median().reset_index()
    print(gg)


ddb = df.groupby(['note', 'season']).apply(
    lambda x: cadence(x)).reset_index()
print(ddb)

# ddb.to_hdf()

# plot cadences
for field in ddb['note'].unique():
    idx = ddb['note'] == field
    sel = ddb[idx]
    fig, ax = plt.subplots(figsize=(10, 9))
    fig.suptitle(field)
    for b in 'ugrizy':
        plot_cadence(sel, 'season', b, fig=fig, ax=ax, label=b)

    ax.grid()
    ax.set_xlabel('season')
    ax.set_ylabel('cadence [night]')
    ax.legend()

plt.show()
