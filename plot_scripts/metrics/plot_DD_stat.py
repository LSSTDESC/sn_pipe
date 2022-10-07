import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sn_plotter_metrics.utils import Infos, Simu
import numpy.lib.recfunctions as rf


def flat_this(grp, cols=['filter_alloc', 'filter_frac']):

    dictout = {}

    for vv in cols:
        dictout[vv] = sum(grp[vv].to_list(), [])

    return pd.DataFrame.from_dict(dictout)


def plot_vs_OS(data, varx='family', vary='time_budget', legy='Time Budget [%]', title='', fig=None, ax=None, label='', color='k', marker='.'):

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    fig.suptitle(title)
    fig.subplots_adjust(bottom=0.20)

    ll = ''
    if label != '':
        ll = data['field'].unique()
    ax.plot(data[varx], data[vary], color=color,
            marker=marker, label='{}'.format(ll))

    ax.grid()
    ax.tick_params(axis='x', labelrotation=20., labelsize=12)
    for tick in ax.xaxis.get_majorticklabels():
        tick.set_horizontalalignment("right")

    ax.legend()
    ax.set_ylabel(legy)


def plot_vs_OS_dual(data, varx='family', vary=['time_budget'], legy=['Time Budget [%]'], title='', fig=None, ax=None, color='k', marker='.'):

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8), ncols=1, nrows=len(vary))

    fig.suptitle(title)
    fig.subplots_adjust(bottom=0.150, hspace=0.02)

    lsize = 17
    for io, vv in enumerate(vary):
        ax[io].plot(data[varx], data[vv], color=color,
                    marker=marker)
        ax[io].grid()
        ax[io].tick_params(axis='x', labelrotation=20., labelsize=lsize)
        for tick in ax[io].xaxis.get_majorticklabels():
            tick.set_horizontalalignment("right")

        ax[io].set_ylabel(legy[io], size=lsize)
        ax[io].tick_params(axis='y', labelsize=lsize)
        if io == 0:
            ax[io].get_xaxis().set_ticklabels([])

    plotDir = '../../Bureau/ddf_fbs_2.1'
    plotName = '{}/cad_sl_{}.png'.format(plotDir, title)
    fig.savefig(plotName)


def plot_hist_OS(data, by='family', what='cadence'):

    fig, ax = plt.subplots()
    for fam in np.unique(data['family']):
        idx = data['family'] == fam
        sel = data[idx]
        sel = sel.dropna()
        ax.hist(sel[what], histtype='step', bins=range(1, 25))
        ll = sel[what].to_list()
        p = np.percentile(ll, 50)
        pm = np.percentile(ll, 16)
        pp = np.percentile(ll, 84)
        print(fam, len(np.unique(sel['dbName'])), pm, p, pp)


def plot_series(df, title='', varx='family', what=['time_budget', 'field'], leg=['Time budget [%]', 'DD Field']):

    for i, vv in enumerate(what):
        plot_vs_OS(df, varx=varx, vary=vv, legy=leg[i], title=title)


def plot_series_median(df, title='', varx='family', what=['time_budget', 'field'], leg=['Time budget [%]', 'DD Field']):

    df = df.groupby(varx)[what].median().reset_index()
    for i, vv in enumerate(what):
        plot_vs_OS(df, varx=varx, vary=vv, legy=leg[i], title=title)


def plot_series_median_fields(df, title='', varx='family', what=['time_budget', 'field'], leg=['Time budget [%]', 'DD Field']):

    df = df.groupby([varx, 'field'])[what].median().reset_index()

    for io, field in enumerate(np.unique(df['field'])):
        idx = df['field'] == field
        sel = df[idx]
        plot_vs_OS_dual(sel, varx=varx, vary=what,
                        legy=leg, title=field)
    # ax.grid()


def plot_night(df, dbName, field):

    print(df.columns)
    idx = df['dbName'] == dbName
    idx &= df['field'] == field
    sel = df[idx]

    for season in sel['season'].unique():
        ids = sel['season'] == season
        sels = sel[ids]
        fig, ax = plt.subplots()
        r = get_list(sels['filter_alloc'])
        rb = get_list(sels['filter_frac'])
        rt = []
        for i, val in enumerate(r):
            rt.append((val, rb[i]))
        tab = np.rec.fromrecords(rt, names=['filter_alloc', 'filter_frac'])
        print('hhh', np.sum(tab['filter_frac']))
        idx = tab['filter_frac'] > 0.02
        tab = tab[idx]
        ax.plot(tab['filter_alloc'], tab['filter_frac'])

    plt.show()


def get_list(tt):

    r = []
    for vv in tt:
        for v in vv:
            r.append(v)
    return r


configFile = 'config_DD.csv'
list_to_process = pd.read_csv(configFile, comment='#')
simu_list = []

for i, row in list_to_process.iterrows():
    print('hello', row)
    simu_list.append(Simu(row['simuType'], row['simuNum'],
                          row['dirFile'], row['dbList'], row['nside']))

for ip, vv in enumerate(simu_list):
    toprocess = Infos(vv, ip).resdf
print(toprocess)

configGroup = 'DD_fbs_2.1_plot.csv'
dfgroup = pd.read_csv(configGroup, comment='#')


df = pd.read_hdf('Summary_DD_orig.hdf5')

df = df.merge(toprocess[['dbName', 'family']],
              left_on=['dbName'], right_on=['dbName'])
df['time_budget'] *= 100.
df = df.merge(dfgroup[['dbName', 'group']],
              left_on=['dbName'], right_on=['dbName'])

df['family'] = df['group']
# strip db Name
df['family'] = df['family'].str.split('_v2.1_10yrs', expand=True)[0]

# uniformity of DD names

torep = dict(zip(['ECDFS', 'EDFS, a', 'EDFS, b', 'EDFS_a', 'EDFS_b', 'XMM_LSS'], [
    'CDFS', 'EDFSa', 'EDFSb', 'EDFSa', 'EDFSb', 'XMM-LSS']))

for key, vals in torep.items():
    df['field'] = df['field'].str.replace(
        key, vals)
df['field'] = df['field'].str.split(':', expand=True)[1]
print(df.columns)

# time-budget and fields vs OS family
# plot_series(df)

#plot_hist_OS(df, what='cadence_median')
# plt.show()
"""
toplot = ['season_length', 'cadence_median']
leg = ['Season length [days]', 'Median cadence [days]']

plot_series_median_fields(df, what=toplot, leg=leg)
plt.show()
"""

flat = df.groupby(['dbName', 'field', 'family', 'season']).apply(
    lambda x: flat_this(x, cols=['filter_alloc', 'filter_frac'])).reset_index()

print('fflat', type(flat))

flat = flat.groupby(['dbName', 'field', 'family', 'filter_alloc', 'season'])[
    'filter_frac'].median().reset_index()


#idx = flat['family'] == 'ddf_deep'
"""
toplot = ['filter_frac']
leg = ['Median obs. night frac']

# for family in np.unique(flat['family']):
for family in ['early_deep']:
    idx = flat['family'] == family
    idx &= flat['field'] == 'COSMOS'
    idx &= flat['filter_frac'] > 0.05
    #idx &= np.abs(flat['season']-1) < 1.e-5
    sel = flat[idx]

    print(sel['season'])

    print(len(sel['filter_alloc']), len(np.unique(sel['filter_alloc'])))
    tit = '{} - COSMOS'.format(family)
    plot_series(sel, title=tit, varx='filter_alloc', what=toplot, leg=leg)
"""

plot_night(
    df, dbName='ddf_early_deep_slf0.20_f10.60_f20.80_v2.1_10yrs', field='COSMOS')

plt.show()
