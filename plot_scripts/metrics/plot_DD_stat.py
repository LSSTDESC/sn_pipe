import pandas as pd
from sn_plotter_metrics import plt
import numpy as np
from sn_plotter_metrics.utils import get_dist
from sn_plotter_metrics.plot4metric import plot_night, plot_series
from sn_plotter_metrics.plot4metric import plot_series_fields, plot_filter_alloc
from sn_plotter_metrics.plot4metric import plot_field, plot_cumsum
from sn_tools.sn_utils import clean_level
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
    df = df.merge(dfgroup[['dbName', 'dbName_plot', 'marker', 'color', 'ls']],
                  left_on=['dbName'], right_on=['dbName'])

    # df['family'] = df['group']

    # strip db Name
    # df['family'] = df['family'].str.split('_v2.99_10yrs', expand=True)[0]

    # uniformity of DD names

    torep = dict(zip(['ECDFS', 'EDFS, a', 'EDFS, b', 'EDFS_a', 'EDFS_b', 'XMM_LSS'], [
        'CDFS', 'EDFSa', 'EDFSb', 'EDFSa', 'EDFSb', 'XMM-LSS']))

    """
    for key, vals in torep.items():
        df['field'] = df['field'].str.replace(
            key, vals)
    df['field'] = df['field'].str.split(':', expand=True)[1]
    """

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
    tt = df.groupby(['dbName', 'dbName_plot'])['nvisits'].mean().reset_index()
    plot_series(tt, what=['nvisits'], leg=['$N_{visits}$'])
    df['time_budget_rel'] = df['time_budget_field']/df['time_budget']
    df['time_budget_rel'] *= 100.
    df_noseas = df.groupby(['dbName', 'field', 'dbName_plot'])[
        'Nfc'].sum().reset_index()
    df_noseas['overhead'] = df_noseas['Nfc'] * \
        2./60  # 2min overhead per filter swap
    plot_series_fields(df_noseas, what=['Nfc', 'overhead'], leg=[
        'Number of filter changes', 'Overhead (filter changes) [h]'])
    df_fi = df.groupby(['dbName', 'dbName_plot'])['Nfc'].sum().reset_index()
    df_fi['overhead'] = df_fi['Nfc']*2./60  # 2min overhead per filter swap
    plot_series(df_fi, what=['Nfc', 'overhead'], leg=[
        'Number of filter changes', 'Overhead (filter changes) [h]'])
    # plot_hist_OS(df, what='cadence_median')

    tf = df.groupby(['dbName', 'dbName_plot', 'field']).apply(
        lambda x: nv_f(x), include_groups=False).reset_index()

    plot_series_fields(tf, what=['nvisits_field'],
                       leg=['$N_{visits}^{field}$'])

    tb = tf.groupby(['dbName', 'dbName_plot'])[
        'nvisits_field'].sum().reset_index(name='nvisits')

    print(tb)

    plot_series(tb, what=['nvisits'], leg=['$N_{visits}^{DDF}$'])


def nv_f(grp):
    """
    function to estimate the number of visits

    Parameters
    ----------
    grp : pandas df
        Data to process.

    Returns
    -------
    res : pandas df
        result.

    """

    bands = list('ugrizy')
    nvisits = grp[bands].sum(axis=1).to_list()

    res = pd.DataFrame([np.sum(nvisits)], columns=['nvisits_field'])

    return res


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


def plot_relative_depth(dfb):
    """
    Function to plot the relative depth of the UD vs DD

    Parameters
    ----------
    df : pandas df
        Data to process.

    Returns
    -------
    None.

    """

    df = pd.DataFrame(dfb)
    bands = list('ugrizy')
    df['nvisits'] = df[bands].sum(axis=1).to_list()

    dbNames = df['dbName'].unique()

    dft = pd.DataFrame()
    for dbName in dbNames:
        idx = df['dbName'] == dbName
        sel = df[idx]
        dfc = get_seasons(sel)
        dfc['dbName'] = dbName
        dft = pd.concat((dft, dfc))
    """
    dfc = df.groupby(['dbName']).apply(
        lambda x: get_relative_depth(x)).reset_index()
    """
    print(dft)

    dfr = dft.groupby(['dbName']).apply(
        lambda x: get_relative_depth(x, df)).reset_index()

    print(dfr)

    dfr = dfr.merge(df[['dbName', 'dbName_plot']], left_on=[
                    'dbName'], right_on=['dbName'], suffixes=['', ''])

    dfr = dfr.sort_values(by=['depth'])
    fig, ax = plt.subplots(figsize=(12, 10))

    ax.plot(dfr['dbName_plot'], dfr['depth'], color='k',
            linestyle='solid', marker='o', mfc='None')

    ax.set_ylabel(
        '$\\frac{N_{visits}^{UD,UD seasons}}{N_{visits}^{ELAISS1,10 seasons}}$')
    ax.grid(visible=True)

    ax.tick_params(axis='x', labelrotation=20., labelsize=12)
    for tick in ax.xaxis.get_majorticklabels():
        tick.set_horizontalalignment("right")


def get_seasons(grp):
    """
    function to get seasons correspondig to UDs per field

    Parameters
    ----------
    grp : pandas df
        Data to process.

    Returns
    -------
    dd : pandas df
        output data.

    """

    dd = grp.groupby(['field']).apply(
        lambda x: get_high_season(x), include_groups=False).reset_index()

    return dd


def get_high_season(grp):
    """
    Function to get high seasons

    Parameters
    ----------
    grp : pandas df
        Data to process.

    Returns
    -------
    pandas df
        list of seasons.

    """

    # take the last seasons as ref
    seaslist = range(6, 10)
    idx = grp['season'].isin(seaslist)

    nvisits = grp[idx]['nvisits'].mean()

    # grab the seasons that are above nvisits

    idx = grp['nvisits'] >= 2.5*nvisits

    seas_res = {}
    rr = grp[idx]['season'].unique().tolist()
    rr = list(map(int, rr))
    rr = list(map(str, rr))
    seas_res['seasons'] = [','.join(rr)]

    return pd.DataFrame.from_dict(seas_res)


def get_relative_depth(grp, data, fields=['COSMOS'], fieldref='ELAISS1'):
    """
    Function to estimate the relative depth

    Parameters
    ----------
    grp : pandas df
        Data to process.
    data : pandas df
        Original data.
    fields : list(str), optional
        List of UD fields. The default is ['COSMOS'].
    fieldref : str, optional
        reference DD field for nvisits(10 years). The default is 'ELAISS1'.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    # print('database', grp.name)
    if 'desc_ddf' in grp.name:
        fields += ['XMM-LSS']

    idx = grp['field'].isin(fields)

    sel = grp[idx]

    seasons = sel['seasons'].to_list()[0]
    ll = seasons.split(',')
    # print('alll', ll)
    ll = list(map(int, ll))

    idxb = data['field'].isin(fields)
    idxb &= data['season'].isin(ll)
    idxb &= data['dbName'] == grp.name

    nvisits = data[idxb]['nvisits'].sum()

    # print('hohoho', data['field'].unique())
    idxc = data['field'] == fieldref
    idxc &= data['dbName'] == grp.name
    selc = data[idxc]

    nvisits_ref = selc['nvisits'].sum()

    # print('aoo', nvisits, nvisits_ref)
    res = [nvisits/nvisits_ref]

    return pd.DataFrame(res, columns=['depth'])


def load_data(dirFile, dbList):
    """
    Function to load data

    Parameters
    ----------
    dirFile : str
        dir files.
    dbList : list(str)
        db list.

    Returns
    -------
    data : pandas df
        output data.

    """

    data = pd.DataFrame()
    for dbName in dbList:
        fName = '{}/{}/Summary_DD_pointings.hdf5'.format(dirFile, dbName)
        print(fName)
        df_ = pd.read_hdf(fName)
        # print(df_.columns)
        df_['dbName'] = dbName

        data = pd.concat((data, df_))

    return data


def identify_ud_scenario(grp, colName='nvisits_field', nseason=5, thresh=3000):
    """
    Function to identify ud scenarios

    Parameters
    ----------
    grp : pandas df
        Data to process.
    colName : str, optional
        column of interest. The default is 'nvisits_field'.
    nseason : int, optional
        number of seasons to use to estimate <Nvisits>. The default is 5.
    thresh : float, optional
        threshold for a season to be considered as a ud season. The default is 3000.

    Returns
    -------
    res : pandas df
        Result.

    """

    # take the nseason lowest seasons for nvisits_fields

    grp = grp.sort_values(by=[colName])
    mysel = grp[colName][:nseason]
    mean_visits = mysel.mean()
    rms_visits = mysel.std()
    median_visits = mysel.median()

    """
    res = pd.DataFrame([mean_visits], columns=['mean_visits'])
    res['median_visits'] = median_visits
    res['rms_visits'] = rms_visits
    """

    diff_visits = grp['nvisits_field']-mean_visits
    idx = diff_visits >= thresh

    res_ud = grp[idx]

    nseason_ud = len(res_ud)
    theseas = '---'
    if len(res_ud) > 0:
        res_ud = res_ud.sort_values(by=['season'])
        res_ud['season'] = res_ud['season'].astype(int)
        seasons_ud = res_ud['season'].to_list()
        seasons_ud = list(map(str, seasons_ud))
        theseas = ','.join(seasons_ud)

    res = pd.DataFrame([nseason_ud], columns=['nseason_ud'])
    res['seasons_ud'] = theseas

    return res


def get_ud_scenario(df, thresh=3000):
    """
    Function to analyse the ud scenario

    Parameters
    ----------
    df : pandas df
        Data to process.
    thresh : float, optional
        threshold (nvisits) to identify ud seasons. The default is 3000.

    Returns
    -------
    None.

    """

    # grab the number if visits per field/season
    tt = df.groupby(['dbName', 'field', 'season']).apply(
        lambda x: nv_f(x), include_groups=True).reset_index()

    # get which ud scenario it is
    bb = tt.groupby(['dbName', 'field']).apply(
        lambda x: identify_ud_scenario(x, thresh=thresh),
        include_groups=False).reset_index()

    res = bb.groupby(['dbName']).apply(
        lambda x: ana_ud_scenario(x), include_groups=False).reset_index()

    bb = bb.merge(res, left_on=['dbName'], right_on=[
                  'dbName'], suffixes=['', ''])

    fi = bb.groupby(['dbName']).apply(lambda x: reformat(x),
                                      include_groups=False).reset_index()

    return fi


def ana_ud_scenario(grp):
    """
    Function to analyze dbName/field ud scenario

    Parameters
    ----------
    grp : pandas df
        Data to process.

    Returns
    -------
    res : pandas df
        result.

    """

    nseason_ud = grp['nseason_ud'].sum()

    idx = grp['nseason_ud'] > 0
    sel = grp[idx]

    nfield_ud = len(sel['field'].unique())

    res = pd.DataFrame([nseason_ud], columns=['ns_ud'])
    res['nf_ud'] = [nfield_ud]

    return res


def reformat(grp,
             fields=['COSMOS', 'XMM-LSS',
                     'ELAISS1', 'CDFS', 'EDFS_a', 'EDFS_b']):
    """
    Function to re-format the df

    Parameters
    ----------
    grp : pandas df
        Data to process.
    fields : list(str), optional
        DDf list. The default is 
        ['COSMOS', 'XMM-LSS','ELAISS1', 'CDFS', 'EDFS_a', 'EDFS_b'].

    Returns
    -------
    res : pandas df
        re-formatted df.

    """

    dd = {}
    dd['ns_ud'] = [grp['ns_ud'].unique()[0]]
    dd['nf_ud'] = [grp['nf_ud'].unique()[0]]

    print(grp['field'].tolist())

    rrb = []
    for i, field in enumerate(fields):
        idx = grp['field'] == field
        sel = grp[idx]
        rrb.append(sel['seasons_ud'].to_list()[0])
        seasons = sel['seasons_ud'].to_list()[0]
        dd[field] = [seasons]
    # dd['/'.join(fields)] = ['/'.join(rrb)]

    res = pd.DataFrame.from_dict(dd)

    return res


parser = OptionParser(
    description='OS analysis plots from pointings')
parser.add_option("--dirFile", type="str",
                  default='../summary_DD_pointings',
                  help="file directory [%default]")
parser.add_option("--nside", type="int", default=128,
                  help="nside for healpixels [%default]")
parser.add_option("--fieldType", type="str", default='DD',
                  help="field type - DD, WFD, Fake [%default]")
"""
parser.add_option("--dbList", type="str", default='List.csv',
                  help="list of cadences to display[%default]")
"""
parser.add_option("--fieldNames", type="str", default='COSMOS,CDFS,XMM-LSS,ELAISS1,EDFS_a,EDFS_b',
                  help="fields to process [%default]")
parser.add_option("--metric", type="str", default='NSNY',
                  help="metric name [%default]")
parser.add_option("--pointingFile", type="str",
                  default='Summary_DD_pointings.hdf5',
                  help="pointing file name [%default]")
parser.add_option("--config", type="str", default='DD_fbs_2.99_plot.csv',
                  help="pointing file name [%default]")
parser.add_option("--addMetric", type=int, default=0,
                  help="to add metric correlation plots [%default]")
parser.add_option("--dbName_night", type=str, default='baseline_v3.0_10yrs',
                  help="dbName for night plot stat [%default]")
parser.add_option("--fieldName_night", type=str, default='COSMOS',
                  help="field for night plot stat [%default]")
parser.add_option("--plots", type=str,
                  default='summary,field_cad_seasonlength,field_nvisits,\
                          field_nvisits_band,relative_depth,filter_alloc,\
                          field_dithering_season,field_dithering_night,\
                          field_weather,get_ud_scenario',
                  help="plots to draw [%default]")


opts, args = parser.parse_args()
# Load parameters
dirFile = opts.dirFile
# dbList = opts.dbList
nside = opts.nside
fieldType = opts.fieldType
metricName = opts.metric
fieldNames = opts.fieldNames.split(',')
pointingFile = opts.pointingFile
config = opts.config
addMetric = opts.addMetric
dbName_night = opts.dbName_night
fieldName_night = opts.fieldName_night
plots = opts.plots.split(',')

df_conf = pd.read_csv(config, comment='#')  # load list of db+plot infos
# df = pd.read_hdf(pointingFile)  # load pointing data
df = load_data(dirFile, df_conf['dbName'].to_list())

df = complete_pointing(df, df_conf)  # merge pointing data+plot data

metric = pd.DataFrame()
if addMetric:
    # load metric data here
    dbNames = df['dbName'].unique()
    metric = load_metric(dirFile, dbNames, metricName,
                         fieldType, fieldNames, nside)
    metric = merge_with_pointing(metric, df)

if 'relative_depth' in plots:
    plot_relative_depth(df)

idx = df['field'].isin(fieldNames)
df = df[idx]

# summary plots
if 'summary' in plots:
    summary_plots(df)

# plots per field

# for field in df['field'].unique():
fields = df['field'].unique()

bands = list('ugrizy')
df['nvisits'] = df[bands].sum(axis=1).to_list()
prefix = 'N$_{visits}$'
for field in fields:
    idx = df['field'] == field
    sel = df[idx]
    if 'field_cad_seasonlength' in plots:
        plot_field(sel, title='{} pointings'.format(field))
    if 'field_nvisits' in plots:
        plot_field(sel, xvars=['season', 'season'],
                   xlab=['season', 'season'],
                   yvars=['nvisits', 'gap_5_10'],
                   ylab=['N$_{visits}$', 'N$_{gaps}^{5-10}$'],
                   title='{} pointings'.format(field))
    if 'field_dithering_season' in plots:
        plot_field(sel, xvars=['season', 'season'],
                   xlab=['season', 'season'],
                   yvars=['RA_std', 'Dec_std'],
                   ylab=['std(RA)', 'std(Dec)'],
                   title='{} pointings'.format(field))
    if 'field_dithering_night' in plots:
        plot_field(sel, xvars=['season', 'season'],
                   xlab=['season', 'season'],
                   yvars=['RA_mean_std_night', 'Dec_mean_std_night'],
                   ylab=['<std(RA)>$_{night}$', '<std(Dec)>$_{night}$'],
                   title='{} pointings'.format(field))
    if 'field_weather' in plots:
        plot_field(sel, xvars=['season', 'season'],
                   xlab=['season', 'season'],
                   yvars=['m5_z_mean', 'm5_z_std'],
                   ylab=['$<m_5^z>$', '$std(m_5^z)$'],
                   title='{} pointings'.format(field))
        plot_field(sel, xvars=['season', 'season'],
                   xlab=['season', 'season'],
                   yvars=['m5_y_mean', 'm5_y_std'],
                   ylab=['$<m_5^y>$', '$std(m_5^y)$'],
                   title='{} pointings'.format(field))
    if 'field_nvisits_band' in plots:
        for b in [['u', 'g'], ['r', 'i'], ['z', 'y']]:
            ylab = list(map(lambda x: prefix + '$^'+x+'$', b))
            plot_field(sel, xvars=['season', 'season'],
                       xlab=['season', 'season'],
                       yvars=b,
                       ylab=ylab,
                       title='{} pointings'.format(field))

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

# this is to plot fraction of filter alloc per night - for one OS only
if 'filter_alloc' in plots:
    flat = df.groupby(['dbName', 'dbName_plot', 'field', 'season']).apply(
        lambda x: flat_this(x, cols=['filter_alloc', 'filter_frac']),
        include_groups=False).reset_index()

    flat = flat.groupby(['dbName', 'dbName_plot', 'field', 'filter_alloc', 'season'])[
        'filter_frac'].median().reset_index()

    idx = df_conf['dbName'] == dbName_night

    family = df_conf[idx]['dbName_plot'].to_list()[0]

    plot_filter_alloc(flat, family, fieldName_night)

    # plot_night(
    #    df, dbName=dbName_night, field=fieldName_night)

if 'get_ud_scenario' in plots:
    res = get_ud_scenario(df)
    res = clean_level(res)
    res.to_csv('ddf_ud_scenario.csv', index=False)

if len(plots) > 0:
    plt.show()
