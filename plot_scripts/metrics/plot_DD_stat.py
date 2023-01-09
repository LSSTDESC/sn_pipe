import pandas as pd
from sn_plotter_metrics import plt
import numpy as np
from sn_plotter_metrics.utils import get_dist
from optparse import OptionParser


def zcomp_frac(grp, frac=0.95):

    selfi = get_dist(grp)
    nmax = np.max(selfi['nsn'])
    idx = selfi['nsn'] <= frac*nmax

    dist_cut = np.min(selfi[idx]['dist'])
    idd = selfi['dist'] <= dist_cut

    zcomp = np.median(selfi[idd]['zcomp'])
    nsn = np.sum(grp['nsn'])

    return pd.DataFrame({'nsn': [nsn], 'zcomp': [zcomp]})


def sel(df, season=1, field='COSMOS'):

    idx = df['season'] == season
    idx &= df['field'] == field

    return df[idx]


def merge_with_pointing(metric, pointings):

    metric_season = metric.groupby(['dbName', 'fieldname', 'season']).apply(
        lambda x: zcomp_frac(x)).reset_index()

    # metric_season = metric.groupby(['dbName', 'fieldname', 'season']).agg({'nsn': 'sum',
    #                                                                       'zcomp': 'median',
    #                                                                       }).reset_index()

    print(metric_season[['season', 'nsn']])
    print(sel(pointings))
    metric_merged = pointings.merge(metric_season, left_on=['dbName', 'field', 'season'], right_on=[
        'dbName', 'fieldname', 'season'], how='outer')

    print(sel(metric_merged))
    io = metric_merged['season'] > 0
    metric_merged = metric_merged[io]

    ii = metric_merged['field'].isna()
    metric_merged = metric_merged[~ii]
    # print(test)
    print('before', metric_merged['field'].unique(),
          metric_merged['fieldname'].unique())
    metric_merged = metric_merged.fillna(0.)
    print('after', metric_merged['field'].unique(),
          metric_merged['fieldname'].unique())

    return metric_merged


def load_metric(dirFile, dbNames, metricName,
                fieldType, fieldNames, nside):

    from sn_plotter_metrics.utils import MetricValues
    metric = MetricValues(dirFile, dbNames, metricName,
                          fieldType, fieldNames, nside).data
    var, varz = 'nsn', 'zcomp'
    idx = metric[var] > 0.
    idx &= metric[varz] > 0.

    metricPlot = metric[idx]
    bad = metric[~idx]
    bad['nsn'] = 0.
    bad['zcomp'] = 0.
    metricPlot = pd.concat((metricPlot, bad))
    return metricPlot


def complete_pointing(df, dfgroup):
    """
    function to merge two df, make some cleaning, ...

    Parameters
    ---------------
    df: pandas df
      first pandas df
    dfgroup: pandas df
      second pandas df

    Returns
    ------------
    modified merged df

    """
    df = df.merge(dfgroup[['dbName', 'group', 'marker', 'color']],
                  left_on=['dbName'], right_on=['dbName'])

    df['family'] = df['group']

    # strip db Name
    df['family'] = df['family'].str.split('_v2.99_10yrs', expand=True)[0]

    # uniformity of DD names

    torep = dict(zip(['ECDFS', 'EDFS, a', 'EDFS, b', 'EDFS_a', 'EDFS_b', 'XMM_LSS'], [
        'CDFS', 'EDFSa', 'EDFSb', 'EDFSa', 'EDFSb', 'XMM-LSS']))

    for key, vals in torep.items():
        df['field'] = df['field'].str.replace(
            key, vals)
    df['field'] = df['field'].str.split(':', expand=True)[1]

    return df


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


def plot_indiv(data, dbName, fig=None, ax=None, xvars=['season', 'season'], xlab=['Season', 'Season'], yvars=['season_length', 'cadence_mean'], ylab=['Season length [days]', 'Mean Cadence [days]'], label='', color='k', marker='.', mfc='k'):

    print('hhhhhh', data.columns)

    if fig is None:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
        fig.suptitle('{} pointings'.format(field))
        fig.subplots_adjust(hspace=0.02)

    # dbName = sel['dbName'].unique()[0]
    # field = sel['field'].unique()[0]
    # fig.suptitle('{} \n {} pointings'.format(dbName, field))

    for io, vv in enumerate(xvars):
        ax[io].plot(data[vv], data[yvars[io]], label=label,
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


def plot_field(df, xvars=['season', 'season'], xlab=['Season', 'Season'], yvars=['season_length', 'cadence_mean'], ylab=['Season length [days]', 'Mean Cadence [days]'], title=''):

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(14, 8))
    fig.suptitle(title)
    fig.subplots_adjust(hspace=0.02, right=0.75)
    colors = ['k', 'r', 'b', 'g']
    for dbName in df['dbName'].unique():
        idx = df['dbName'] == dbName
        sel = df[idx]
        family = sel['family'].unique()[0]
        marker = sel['marker'].unique()[0]
        color = sel['color'].unique()[0]
        plot_indiv(sel, dbName, fig=fig, ax=ax, xvars=xvars, xlab=xlab, yvars=yvars, ylab=ylab,
                   label=family, marker=marker, color=color, mfc='None')

    ax[1].legend(bbox_to_anchor=(1., 1.), ncol=1, frameon=False)

    ax[0].grid()
    ax[1].grid()

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


def plot_cumsum(selb, title='', xvar='zcomp', xleg='$z_{complete}$',
                yvar='nsn', yleg='$N_{SN}$', ascending=False):

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle(title)
    fig.subplots_adjust(right=0.75)
    from scipy.interpolate import interp1d
    # fig.suptitle(title)
    fig.subplots_adjust(bottom=0.20)
    for dbName in selb['dbName'].unique():
        idx = selb['dbName'] == dbName
        selp = selb[idx]
        family = selp['family'].unique()[0]
        marker = selp['marker'].unique()[0]
        color = selp['color'].unique()[0]
        selp = selp.sort_values(by=[xvar], ascending=ascending)
        cumulnorm = np.cumsum(selp[yvar])/np.sum(selp[yvar])
        ax.plot(selp[xvar], cumulnorm, marker=marker,
                color=color, mfc='None', label=family)
        interp = interp1d(
            cumulnorm, selp[xvar], bounds_error=False, fill_value=0.)
        zcomp = interp(0.95)
        io = selp[xvar] >= zcomp
        print('zcomp', np.median(selp[io][xvar]))
    ax.grid()
    ax.invert_xaxis()
    ax.legend(bbox_to_anchor=(1.4, 0.8), ncol=1, frameon=False)
    ax.set_xlabel(xleg)
    ax.set_ylabel(yleg)


parser = OptionParser(
    description='Display correlation plots between (NSN,zlim) metric results for DD fields and the budget')
parser.add_option("--dirFile", type="str",
                  default='../MetricOutput_DD_new_128_gnomonic_circular',
                  help="file directory [%default]")
parser.add_option("--nside", type="int", default=128,
                  help="nside for healpixels [%default]")
parser.add_option("--fieldType", type="str", default='DD',
                  help="field type - DD, WFD, Fake [%default]")
parser.add_option("--dbList", type="str", default='List.csv',
                  help="list of cadences to display[%default]")
parser.add_option("--fieldNames", type="str", default='COSMOS,CDFS,XMM-LSS,ELAISS1,EDFSa,EDFSb,EDFS',
                  help="fields to process [%default]")
parser.add_option("--metric", type="str", default='NSNY',
                  help="metric name [%default]")
parser.add_option("--pointingFile", type="str", default='Summary_DD_pointings.hdf5',
                  help="pointing file name [%default]")
parser.add_option("--configGroup", type="str", default='DD_fbs_2.99_plot.csv',
                  help="pointing file name [%default]")
parser.add_option("--addMetric", type=int, default=0,
                  help="to add metric correlation plots [%default]")
parser.add_option("--plotSummary", type=int, default=0,
                  help="to draw summary plots [%default]")


opts, args = parser.parse_args()
# Load parameters
dirFile = opts.dirFile
dbList = opts.dbList
nside = opts.nside
fieldType = opts.fieldType
metricName = opts.metric
fieldNames = opts.fieldNames.split(',')
pointingFile = opts.pointingFile
configGroup = opts.configGroup
addMetric = opts.addMetric
plotSummary = opts.plotSummary

dfgroup = pd.read_csv(configGroup, comment='#')  # load list of db+plot infos
df = pd.read_hdf(pointingFile)  # load pointing data
df = complete_pointing(df, dfgroup)  # merge pointing data+plot data

metric = pd.DataFrame()
if addMetric:
    # load metric data here
    dbNames = df['dbName'].unique()
    metric = load_metric(dirFile, dbNames, metricName,
                         fieldType, fieldNames, nside)
    metric = merge_with_pointing(metric, df)

print('ahhh', metric)

# summary plots
if plotSummary:
    summary_plots(df)
    plt.show()

# plots per field

# for field in df['field'].unique():
for field in ['COSMOS']:
    idx = df['field'] == field
    sel = df[idx]
    plot_field(sel, title='{} pointings'.format(field))
    if addMetric:
        print(metric.columns)
        idc = metric['field'] == field
        selm = metric[idc]

        #
        plot_field(selm, yvars=['nsn', 'zcomp'], ylab=[
                   'N$_{SN}$', '$z_{complete}$'], title='{} metrics'.format(field))
        """
        selm['time_budget_field_season'] *= 100.
        plot_field(selm, yvars=['time_budget_field_season', 'zcomp'], ylab=[
                   'Time budget [%]', '$z_{complete}$'], title='{} metrics'.format(field))
        plot_field(selm, yvars=['time_budget_field_season', 'nsn'], ylab=[
                   'Time budget [%]', 'N$_{SN}$'], title='{} metrics'.format(field))
        selmm = selm.groupby(['dbName'])['nsn'].sum().reset_index()
        selmm = selmm.rename(columns={'nsn': 'nsn_season'})
        selb = selm.merge(selmm, left_on=['dbName'], right_on=['dbName'])
        selb['nsn_frac'] = selb['nsn']/selb['nsn_season']
        plot_field(selb, yvars=['time_budget_field_season', 'nsn_frac'], ylab=[
                   'Time budget [%]', 'N$_{SN}$ frac'], title='{} metrics'.format(field))
        
        """
        """
        idx = selm['zcomp'] > 0
        plot_cumsum(selm[idx], title=field, xvar='zcomp', xleg='$z_{complete}$',
                    yvar='nsn', yleg='$N_{SN}$ frac', ascending=False)
        """

"""
print(metric['field'].unique())
metric_field = metric.groupby(['dbName', 'field', 'family', 'marker', 'color']).agg({'nsn': 'sum',
                                                                                     'zcomp': 'median',
                                                                                     }).reset_index()
# plot_field(metric_field, xvars=['fieldname', 'fieldname'], xlab=['', ''], yvars=['nsn', 'zcomp'], ylab=[
#    'N$_{SN}$', '$z_{complete}$'], title='{} metrics'.format(field))
metric_field['fieldname'] = metric_field['field']
plot_series_fields(metric_field, title='', varx='family', what=[
                   'nsn', 'zcomp'], leg=['N$_{SN}$', '$z_{complete}$'])
"""
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
