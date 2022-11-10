import pandas as pd
from sn_plotter_metrics import plt
import numpy as np
from sn_plotter_metrics.utils import Infos, Simu
import numpy.lib.recfunctions as rf


def summary_plots(df):
    df['time_budget'] *= 100.
    df['time_budget_field'] *= 100.
    plot_series(df)
    df['time_budget_rel'] = df['time_budget_field']/df['time_budget']
    df['time_budget_rel'] *= 100.
    plot_series_fields(df)
    df_noseas = df.groupby(['dbName', 'field', 'family'])[
        'Nfc'].sum().reset_index()
    df_noseas['overhead'] = df_noseas['Nfc'] * \
        2./60  # 2min overhead per filter swap
    plot_series_fields(df_noseas, what=['Nfc', 'overhead'], leg=[
        'Number of filter changes', 'Overhead (filter changes) [h]'])
    df_fi = df.groupby(['dbName', 'family'])['Nfc'].sum().reset_index()
    df_fi['overhead'] = df_fi['Nfc']*2./60  # 2min overhead per filter swap
    plot_series(df_fi, what=['Nfc', 'overhead'], leg=[
        'Number of filter changes', 'Overhead (filter changes) [h]'])
    # plot_hist_OS(df, what='cadence_median')


def flat_this(grp, cols=['filter_alloc', 'filter_frac']):

    dictout = {}

    for vv in cols:
        dictout[vv] = sum(grp[vv].to_list(), [])

    return pd.DataFrame.from_dict(dictout)


def plot_vs_OS(data, varx='family', vary='time_budget', legy='Time Budget [%]', title='', fig=None, ax=None, label='', color='k', marker='.', ls='solid', mfc='k'):

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    fig.suptitle(title)
    fig.subplots_adjust(bottom=0.20)

    # ll = ''
    # if label != '':
    #    ll = data['field'].unique()
    ax.plot(data[varx], data[vary], color=color,
            marker=marker, label='{}'.format(label), linestyle=ls, mfc=mfc)

    ax.grid()
    ax.tick_params(axis='x', labelrotation=20., labelsize=12)
    for tick in ax.xaxis.get_majorticklabels():
        tick.set_horizontalalignment("right")

    if label != '':
        ax.legend()
    ax.set_ylabel(legy)


def plot_vs_OS_dual(data, varx='family', vary=['time_budget'], legy=['Time Budget [%]'], title='', fig=None, ax=None, color='k', marker='.', ls='solid'):

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


def plot_series_fields(df, title='', varx='family', what=['time_budget_field', 'time_budget_rel'], leg=['Field Time budget [%]', 'Relative Field Time budget [%]']):

    ls = ['solid', 'dotted', 'dashed', 'dashdot']*2
    marker = ['.', 's', 'o', '^', 'P', 'h']
    colors = ['k', 'r', 'b', 'm', 'g', 'c']
    for i, vv in enumerate(what):
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.subplots_adjust(top=0.90)
        for io, field in enumerate(np.unique(df['field'])):
            idx = df['field'] == field
            sel = df[idx]
            print('aoooouuu', field)
            plot_vs_OS(sel, varx=varx, vary=vv,
                       legy=leg[i], title=title, fig=fig, ax=ax, ls=ls[io], label='{}'.format(field), marker=marker[io], mfc='None', color=colors[io])
        ax.legend(bbox_to_anchor=(0.5, 1.17), ncol=3,
                  frameon=False, loc='upper center')
        ax.grid()


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


def plot_indiv(data, dbName, field='COSMOS', fig=None, ax=None, xvars=['season', 'season'], xlab=['Season', 'Season'], yvars=['season_length', 'cadence_mean'], ylab=['Season length [days]', 'Mean Cadence [days]'], label='', color='k', marker='.', mfc='k'):

    print('hhhhhh', data.columns)

    if fig is None:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
        fig.suptitle('{} pointings'.format(field))
        fig.subplots_adjust(hspace=0.02)
    idx = data['field'] == field
    sel = data[idx]
    # dbName = sel['dbName'].unique()[0]
    # field = sel['field'].unique()[0]
    # fig.suptitle('{} \n {} pointings'.format(dbName, field))

    for io, vv in enumerate(xvars):
        ax[io].plot(sel[vv], sel[yvars[io]], label=label,
                    marker=marker, mfc=mfc, color=color)
        ax[io].set_ylabel(ylab[io])
        if io == 0:
            ax[io].get_xaxis().set_ticklabels([])
        if io == 1:
            ax[1].set_xlabel('Season')
        ax[io].grid()


def get_list(tt):

    r = []
    for vv in tt:
        for v in vv:
            r.append(v)
    return r


def plot_field(df, field='COSMOS'):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(14, 8))
    fig.suptitle('{} pointings'.format(field))
    fig.subplots_adjust(hspace=0.02, right=0.75)
    colors = ['k', 'r', 'b', 'g']
    for dbName in df['dbName'].unique():
        idx = df['dbName'] == dbName
        sel = df[idx]
        family = sel['family'].unique()[0]
        marker = sel['marker'].unique()[0]
        color = sel['color'].unique()[0]
        plot_indiv(sel, dbName, field=field, fig=fig, ax=ax,
                   label=family, marker=marker, color=color, mfc='None')

    ax[0].legend(bbox_to_anchor=(1., 0.5), ncol=1, frameon=False)
    for io in range(2):
        ax[io].grid()


def plot_filter_alloc(flat, family, field):

    toplot = ['filter_frac']
    leg = ['Median obs. night frac']
    idx = flat['family'] == family
    idx &= flat['field'] == field
    idx &= flat['filter_frac'] > 0.05
    # idx &= np.abs(flat['season']-1) < 1.e-5
    sel = flat[idx]

    tit = '{} - {}'.format(family, field)
    plot_series(sel, title=tit, varx='filter_alloc', what=toplot, leg=leg)


configGroup = 'DD_fbs_2.99_plot.csv'
dfgroup = pd.read_csv(configGroup, comment='#')

df = pd.read_hdf('Summary_DD_pointings_v2.99.hdf5')

df = df.merge(dfgroup[['dbName', 'group', 'marker', 'color']],
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

# summary plots
# summary_plots(df)
# plt.show()

# plots per field

for field in df['field'].unique():
    # field = 'COSMOS'
    plot_field(df, field=field)
plt.show()

# Medians over season
"""
toplot = ['season_length', 'cadence_median']
leg = ['Season length [days]', 'Median cadence [days]']

plot_series_median_fields(df, what=toplot, leg=leg)
plt.show()
"""

flat = df.groupby(['dbName', 'field', 'family', 'season']).apply(
    lambda x: flat_this(x, cols=['filter_alloc', 'filter_frac'])).reset_index()

flat = flat.groupby(['dbName', 'field', 'family', 'filter_alloc', 'season'])[
    'filter_frac'].median().reset_index()


plot_filter_alloc(flat, 'dd6', 'COSMOS')
plt.show()

"""

plot_night(
    df, dbName='dd6_v2.99_10yrs', field='COSMOS')

plt.show()
"""
