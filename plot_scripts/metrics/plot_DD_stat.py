import pandas as pd
from sn_plotter_metrics import plt
import numpy as np
from sn_plotter_metrics.utils import get_dist
from sn_plotter_metrics.plot4metric import plot_night, plot_series
from sn_plotter_metrics.plot4metric import plot_series_fields, plot_filter_alloc
from sn_plotter_metrics.plot4metric import plot_field, plot_cumsum
from optparse import OptionParser


def zcomp_frac(grp, frac=0.95):
    """
    Function to estimate metric values from NSN distribution

    Parameters
    ----------
    grp : pandas df
        data to process.
    frac : float, optional
        NSN frac to estimate nsn, zcomp. The default is 0.95.

    Returns
    -------
    pandas df
        metric data with the frac selection.

    """

    selfi = get_dist(grp)
    nmax = np.max(selfi['nsn'])
    idx = selfi['nsn'] <= frac*nmax

    dist_cut = np.min(selfi[idx]['dist'])
    idd = selfi['dist'] <= dist_cut

    zcomp = np.median(selfi[idd]['zcomp'])
    nsn = np.sum(grp['nsn'])

    return pd.DataFrame({'nsn': [nsn], 'zcomp': [zcomp]})


def sel(df, season=1, field='COSMOS'):
    """
    Function to select a field from data

    Parameters
    ----------
    df : pandas df
        data to process
    season : int, optional
        Season of observation. The default is 1.
    field : str, optional
        Name of the field to consider. The default is 'COSMOS'.

    Returns
    -------
    pandas df
        data corresponding to (field,season)

    """
    idx = df['season'] == season
    idx &= df['field'] == field

    return df[idx]


def merge_with_pointing(metric, pointings):
    """
    Function to merge metric and pointing data

    Parameters
    ----------
    metric : pandas df
        metric data
    pointings : pandas df
        pointings data

    Returns
    -------
    metric_merged : pandas df
        metric+pointings data

    """

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
    """
    Function to load metric values

    Parameters
    ----------
    dirFile : str
        location dir of the files
    dbNames : list(str)
        list of db to load
    metricName : str
        metric to consider
    fieldType : str
        type of field (DD or WFD)
    fieldNames : list(str)
        list of fields to consider
    nside : int
        healpix nside parameter

    Returns
    -------
    metricPlot : pandas df
        metric data

    """

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
    df = df.merge(dfgroup[['dbName', 'family', 'marker', 'color']],
                  left_on=['dbName'], right_on=['dbName'])

    # df['family'] = df['group']

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
    """
    Function to plot a set of OS parameters (pointings)

    Parameters
    ----------
    df : pandas df
        data to plot

    Returns
    -------
    None.

    """
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
    """
    Function to flatten some df columns

    Parameters
    ----------
    grp : pandas df
        data to process
    cols : list(str), optional
        list of cols to flatten. The default is ['filter_alloc', 'filter_frac'].

    Returns
    -------
    pandas df
        data with flattened cols

    """

    dictout = {}

    for vv in cols:
        dictout[vv] = sum(grp[vv].to_list(), [])

    return pd.DataFrame.from_dict(dictout)


parser = OptionParser(
    description='Display correlation plots between (NSN,zlim) \
    metric results for DD fields and the budget')
parser.add_option("--dirFile", type="str",
                  default='../MetricOutput_DD_new_128_gnomonic_circular',
                  help="file directory [%default]")
parser.add_option("--nside", type="int", default=128,
                  help="nside for healpixels [%default]")
parser.add_option("--fieldType", type="str", default='DD',
                  help="field type - DD, WFD, Fake [%default]")
parser.add_option("--dbList", type="str", default='List.csv',
                  help="list of cadences to display[%default]")
parser.add_option("--fieldNames", type="str", default='COSMOS,CDFS,\
                  XMM-LSS,ELAISS1,EDFSa,EDFSb,EDFS',
                  help="fields to process [%default]")
parser.add_option("--metric", type="str", default='NSNY',
                  help="metric name [%default]")
parser.add_option("--pointingFile", type="str",
                  default='Summary_DD_pointings.hdf5',
                  help="pointing file name [%default]")
parser.add_option("--configGroup", type="str", default='DD_fbs_2.99_plot.csv',
                  help="pointing file name [%default]")
parser.add_option("--addMetric", type=int, default=0,
                  help="to add metric correlation plots [%default]")
parser.add_option("--plotSummary", type=int, default=0,
                  help="to draw summary plots [%default]")
parser.add_option("--dbName_night", type=str, default='baseline_v3.0_10yrs',
                  help="dbName for night plot stat [%default]")
parser.add_option("--fieldName_night", type=str, default='COSMOS',
                  help="field for night plot stat [%default]")
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
dbName_night = opts.dbName_night
fieldName_night = opts.fieldName_night

dfgroup = pd.read_csv(configGroup, comment='#')  # load list of db+plot infos
df = pd.read_hdf(pointingFile)  # load pointing data
print('hhh', df)
print('bbb', dfgroup)
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

        idx = selm['zcomp'] > 0
        plot_cumsum(selm[idx], title=field, xvar='zcomp', xleg='$z_{complete}$',
                    yvar='nsn', yleg='$N_{SN}$ frac', ascending=False)


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
leg = ['Median season length [days]', 'Median cadence [days]']

dfb = df.groupby(['family', 'field'])[toplot].median().reset_index()
plot_series_fields(dfb, what=toplot, leg=leg)
plt.show()
"""

# this is to plot fraction of filter alloc per night

flat = df.groupby(['dbName', 'field', 'family', 'season']).apply(
    lambda x: flat_this(x, cols=['filter_alloc', 'filter_frac'])).reset_index()

flat = flat.groupby(['dbName', 'field', 'family', 'filter_alloc', 'season'])[
    'filter_frac'].median().reset_index()

print(flat)
idx = dfgroup['dbName'] == dbName_night
family = dfgroup[idx]['family'].to_list()[0]
plot_filter_alloc(flat, family, fieldName_night)

# plot_night(
#    df, dbName=dbName_night, field=fieldName_night)

plt.show()
