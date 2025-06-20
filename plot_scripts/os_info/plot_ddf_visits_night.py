#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:46:22 2025

@author: philippe.gris@clermont.in2p3.fr
"""
import pandas as pd
from optparse import OptionParser
from sn_plotter_os_info.ddf_visits_night import analyze_simu_exp
from sn_plotter_analysis.sn_analyser_tools import clean_level
import matplotlib.pyplot as plt
from sn_plotter_metrics.plot4metric import plot_vs_OS
from sn_tools.sn_utils import multiproc
import numpy as np
from sn_plotter_os_info.ddf_visits_night import plot_stat_visits_vs_exp
import operator as op


def ana_seq_multi(toproc, params, j=0, output_q=None):
    """
    Analysis function using multiprocessing

    Parameters
    ----------
    toproc : list(str)
        List of OS to process.
    params : dict
        parameters.
    j : int, optional
        internal tag for multiprocessing. The default is 0.
    output_q : multiprocessing queue, optional
        Where to put the data. The default is None.

    Returns
    -------
    pandas df
        Analyzed data.

    """

    timescale = params['timescale']
    df = params['data']

    idx = df['dbName'].isin(toproc)

    sel = pd.DataFrame(df[idx])

    del df

    res = ana_seq(sel, timescale)

    if output_q is not None:
        return output_q.put({j: res})
    else:
        return res


def ana_seq(df, timescale='year'):
    """
    Function to analyze DDF sequences

    Parameters
    ----------
    df : pandas df
        Data to process.
    timescale : str, optional
        Timescale. The default is 'year'.

    Returns
    -------
    dfd : pandas df
        Output data.

    """

    ccols_m = ['target_name', timescale, 'dbName']
    ccols = ccols_m+['seq_tot']

    dfb = df.groupby(ccols)[ccols].apply(
        lambda x: get_nvisits(x)).reset_index()
    dfb = clean_level(dfb)

    bands = 'ugrizy'
    colsb = ccols+list(bands)
    # dfb = dfb.merge(df[colsb], left_on=ccols,right_on=ccols,suffixes=['',''])

    for b in bands:
        ccols = ccols_m+[b]+['seq_tot']
        ccob = 'nnights_{}'.format(b)
        dfe = df.groupby(ccols)[ccols].apply(
            lambda x: get_nvisits_band(x, b, ccob)).reset_index()

        dfe = clean_level(dfe)

        dfb = dfb.merge(dfe, left_on=ccols_m+['seq_tot'],
                        right_on=ccols_m+['seq_tot'], suffixes=['', ''])

    ccols = ccols_m+['night']

    dfc = df.groupby(ccols_m)[ccols].apply(
        lambda x: get_nnights(x)).reset_index()
    dfc = clean_level(dfc)
    dfd = dfb.merge(dfc, left_on=ccols_m, right_on=ccols_m, suffixes=['', ''])

    dfd['seq_frac'] = 100.*dfd['nnights']/dfd['nnights_year']

    return dfd


def get_nvisits(grp, thevar='nnights'):
    """
    Function to estimate the number of nights corresponding to a DDF sequence

    Parameters
    ----------
    grp : pandas df
        Data to process.
    thevar : str, optional
        output col name. The default is 'nnights'.

    Returns
    -------
    res : pandas df
        output data.

    """

    dd = {}

    dd[thevar] = [len(grp)]

    res = pd.DataFrame.from_dict(dd)

    res[thevar] = res[thevar].astype(int)
    return res


def get_nvisits_band(grp, thevar, thevar_name='nnights'):
    """
    Function to get the number of visits per band


    Parameters
    ----------
    grp : pandas df
        Data to process.
    thevar : str
        col to process.
    thevar_name : str, optional
        col output name. The default is 'nnights'.

    Returns
    -------
    res : pandas df
        output result.

    """

    dd = {}

    idx = grp[thevar] > 0
    sel = grp[idx]

    dd[thevar_name] = [len(sel)]

    res = pd.DataFrame.from_dict(dd)

    res[thevar_name] = res[thevar_name].astype(int)
    return res


def get_nnights(grp, thevar='nnights_year'):
    """
    Function to estimate the total number of nights

    Parameters
    ----------
    grp : pandas df
        Data to process.
    thevar : str, optional
        output col name. The default is 'nnights_year'.

    Returns
    -------
    res : pandas df
        Result.

    """

    dd = {}
    nights = grp['night'].unique()
    dd[thevar] = [len(nights)]

    res = pd.DataFrame.from_dict(dd)

    res[thevar] = res[thevar].astype(int)

    return res


def plot_seq_frac(data, dbName, field='COSMOS', season=1,
                  what='seq_frac', legy='sequence fraction [%]'):
    """
    Funtion to plot DDF sequence fraction

    Parameters
    ----------
    data : pandas df
        Data to process.
    dbName : str
        OS name.
    field : str, optional
        Field considered. The default is 'COSMOS'.
    season : int, optional
        season of interest. The default is 1.
    what : str, optional
        What to plot. The default is 'seq_frac'.
    legy : str, optional
        y-axis legend. The default is 'sequence fraction [%]'.

    Returns
    -------
    None.

    """

    idx = data['target_name'] == field
    idx &= data['year'] == season
    sel = data[idx]

    figtit = dbName
    figtit += '\n {} - year {}'.format(field, season)

    plot_vs_OS(sel, varx='seq_tot',
               vary=what,
               legy=legy,
               title=figtit, fig=None, ax=None,
               label='', color='k', marker='.', ls='solid', mfc='k', mec='k')

    selb = sel.sort_values(by=[what], ascending=False)
    print(selb[['seq_tot', what]][:2])


def plot_all(ro, dbName, field='DD:COSMOS', season=3):
    """
    Function to plot a serie of results

    Parameters
    ----------
    ro : pandas df
        Data to process.
    dbName : str
        Db name.
    field : str, optional
        DDF field name. The default is 'DD:COSMOS'.
    season : int, optional
        season/year to show. The default is 3.

    Returns
    -------
    None.

    """

    plot_seq_frac(ro, dbName, field=field, season=season)

    idx = ro['y'] > 0
    """
    plot_seq_frac(ro[idx], dbName, field=field, season=season,
                  what='nnights', legy='Number of nights')
    """
    plot_seq_frac(ro[idx], dbName, field=field, season=season,
                  what='seq_frac')

    idx = ro['u'] > 0
    """
    plot_seq_frac(ro[idx], dbName, field=field, season=season,
                  what='nnights', legy='Number of nights')
    """
    plot_seq_frac(ro[idx], dbName, field=field, season=season,
                  what='seq_frac')

    idx = ro['u'] == 0
    idx &= ro['y'] == 0
    """
    plot_seq_frac(ro[idx], dbName, field=field, season=season,
                  what='nnights', legy='Number of nights')
    """
    plot_seq_frac(ro[idx], dbName, field=field, season=season,
                  what='seq_frac')


def calc_summary(grp, col='y'):
    """
    Function to extract some result

    Parameters
    ----------
    grp : pandas df
        Data to process.
    col : str, optional
        col name to select. The default is 'y'.

    Returns
    -------
    rr : pandas df
        output result.

    """

    idx = grp[col] > 0
    sel = grp[idx]
    selb = sel.sort_values(by=['seq_frac'], ascending=False)

    rr = selb[['seq_tot', 'seq_frac', col]][:1]
    rr['nvisits_{}'.format(col)] = selb[col].sum()
    nnights_band = selb['nnights_{}'.format(col)].sum()
    rr['frac_{}'.format(col)] = sel['seq_frac'].sum()
    # correct to get the fraction of seq corresponding to the band
    nnights_year = selb['nnights_year'][:1]
    rr['seq_frac'] *= nnights_year/nnights_band

    rr = rr.rename(columns={'seq_tot': 'seq_tot_{}'.format(col),
                            'seq_frac': 'seq_frac_{}'.format(col)})
    return rr


def summary_seq(grp):
    """
    function to estimate summary results

    Parameters
    ----------
    grp : pandas df
        Data to process.

    Returns
    -------
    res : pandas df
        output data.

    """

    rr = calc_summary(grp, 'y')

    bb = calc_summary(grp, 'u')

    res = rr.merge(bb, how='cross')

    return res


def plot_summary(data, field='DD:COSMOS',
                 varx='year', labx='year',
                 vary='seq_tot_y', laby='',
                 figtit='DD:COSMOS',
                 df_config=pd.DataFrame()):
    """
    Summary plot

    Parameters
    ----------
    data : pandas df
        Data to process.
    field : str, optional
        Field type. The default is 'DD:COSMOS'.
    varx : str, optional
        x-axis variable. The default is 'year'.
    labx : str, optional
        x-axis label. The default is 'year'.
    vary : str, optional
        y-axis variable. The default is 'seq_tot_y'.
    laby : str, optional
        y-axis label. The default is ''.
    figtit : str, optional
        Figure title. The default is 'DD:COSMOS'.
    df_config : pandas df, optional
        config for the plot. The default is pd.DataFrame().

    Returns
    -------
    None.

    """

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.suptitle(figtit)
    fig.subplots_adjust(right=0.75)

    idx = data['target_name'] == field

    sel = data[idx]

    dbNames = sel['dbName'].unique()

    for dbName in dbNames:
        io = sel['dbName'] == dbName
        selb = sel[io]
        idxb = df_config['dbName'] == dbName
        selp = df_config[idxb]
        ls = selp['ls'].values[0]
        marker = selp['marker'].values[0]
        color = selp['color'].values[0]
        dbNameb = selp['dbName_plot'].values[0]
        ax.plot(selb[varx], selb[vary],
                ls=ls, marker=marker, color=color, mfc='None', label=dbNameb)

    ax.grid(visible=True)
    ax.set_xlabel(r'{}'.format(labx))
    ax.set_ylabel(r'{}'.format(laby))
    if laby == '':
        ax.tick_params(axis='y', labelrotation=20, labelsize=10)

    ax.legend(loc='upper center',
              bbox_to_anchor=(1.20, 0.7),
              ncol=1, fontsize=12, frameon=False)

    # plt.tight_layout()


def plot_fields(dft, config):
    """
    Function to make a set of plots

    Parameters
    ----------
    dft : pandas df
        Data to plot.
    config : str
        config file name for the plot.

    Returns
    -------
    None.

    """

    # load config for the plot
    df_config = pd.read_csv(config, comment='#')

    field = 'DD:COSMOS'
    plot_summary(dft, field=field, df_config=df_config)
    plot_summary(dft, field=field, vary='seq_frac_y',
                 laby='Fraction of nights [%]', df_config=df_config)
    plot_summary(dft, field=field, vary='nvisits_y',
                 laby='$N_{visits}^{y}$', df_config=df_config)

    plt.show()
    """
    field = 'DD:XMM_LSS'
    plot_summary(dft, field=field, df_config=df_config,figtit=field)
    plot_summary(dft, field=field,figtit=field, vary='seq_frac_y',
                 laby='Fraction of nights [%]', df_config=df_config)
    plot_summary(dft, field=field,figtit=field, vary='nvisits_y',
                 laby='$N_{visits}^{y}$', df_config=df_config)

    """


def plot_stat(dft, config):

    # load config for the plot
    df_config = pd.read_csv(config, comment='#')

    field = 'DD:COSMOS'
    figtot = field
    figtit = figtot + '\n $\\frac{N_{visits}^{obs}}{N_{visits}^{exp}}$==1'
    plot_summary(dft, field=field, figtit=figtit, df_config=df_config, vary='frac_equal',
                 laby='Fraction of nights [%]')
    figtit = figtot + '\n $\\frac{N_{visits}^{obs}}{N_{visits}^{exp}}$>1'
    plot_summary(dft, field=field, figtit=figtit, vary='frac_plus',
                 laby='Fraction of nights [%]', df_config=df_config)
    figtit = figtot + '\n $\\frac{N_{visits}^{obs}}{N_{visits}^{exp}}$<1'
    plot_summary(dft, field=field, figtit=figtit, vary='frac_minus',
                 laby='Fraction of nights [%]', df_config=df_config)

    plt.tight_layout()
    plt.show()


def get_stats(df_summary, df_orig):
    """
    Funtion to get the number of nights corresponding to a sequence
    (ie nnights with the sequence, nnights with nvisits < n_sequence, 
     and nnights with nvisits > n_sequence)

    Parameters
    ----------
    df_summary : pandas df
        Data to process.
    df_orig : pandas df
        Data to process.

    Returns
    -------
    rr : pandas df
        Result.

    """

    rr = df_summary.groupby(['dbName', 'target_name', 'year']).apply(
        lambda x: get_stat_indiv(x, df_orig), include_groups=False).reset_index()

    return rr


def analysis_sequences(df_summary, df_orig, target_name='DD:COSMOS', year=3):
    """
    Function to analyze sequences for each field/season

    Parameters
    ----------
    df_summary : pandas df
        Summary results.
    df_orig : pandas df
        original results.
    target_name : str, optional
        field name. The default is 'DD:COSMOS'.
    year : int, optional
        year. The default is 3.
    Returns
    -------
    None.

    """

    idx = df_summary['target_name'] == target_name
    idx &= df_summary['year'] == year

    rr = df_summary[idx].groupby(['dbName', 'target_name', 'year']).apply(
        lambda x: get_ratios(x, df_orig), include_groups=False)

    plot_stat_visits_vs_exp(rr, op.ge, '>', bins=np.arange(1.0, 2.5, 0.01))
    plot_stat_visits_vs_exp(rr, op.le, '<', bins=np.arange(0.0, 1.1, 0.01))

    plt.show()


def get_ratios(grp, df_orig, band='y'):
    """
    Function to extract the ratios of the number of visits per band wrt ref

    Parameters
    ----------
    grp : pandas df
        Data to process.
    df_orig : pandas df
        Data to extract info from.
    band : str, optional
        band considered. The default is 'y'.

    Returns
    -------
    sel_test : pandas df
        The result.

    """

    dbName = grp.name[0]
    target_name = grp.name[1]
    year = grp.name[2]

    idx = df_orig['dbName'] == dbName
    idx &= df_orig['target_name'] == target_name
    idx &= df_orig['year'] == year
    idx &= df_orig[band] > 0

    sel_orig = pd.DataFrame(df_orig[idx])

    del df_orig

    nnights_y = len(sel_orig)

    b_ref = get_ref_sequence(grp, band)

    b_ref = clean_level(b_ref)
    sel_test = sel_orig.merge(b_ref, how='cross')

    sel_test['diff_nvisits'] = sel_test['nvisits']-sel_test['nvisits_ref']
    bands = 'grizy'
    for b in bands:
        sel_test['ratio_{}'.format(
            b)] = sel_test[b]/sel_test['{}_ref'.format(b)]

    sel_test = clean_level(sel_test)
    return sel_test


def get_ref_sequence(grp, band):
    """
    Function to extract the reference sequence as a df

    Parameters
    ----------
    grp : pandas df
        Data to process.
    band : str
        band to consider for the ref sequence.

    Returns
    -------
    b_ref : pandas df
        Reference df sequence.

    """

    seq = grp['seq_tot_{}'.format(band)].values[0]

    spl = seq.split('-')

    bands = [sp[-1] for sp in spl]
    bands_ref = list(map(lambda el: el+'_ref', bands))
    nv_ref = list(map(int, [sp[:-1] for sp in spl]))
    nv_ref = list(map(lambda el: [el], nv_ref))
    dd = dict(zip(bands_ref, nv_ref))
    b_ref = pd.DataFrame.from_dict(dd)

    b_ref['nvisits_ref'] = b_ref[bands_ref].sum(axis=1)

    return b_ref


def get_stat_indiv(grp, df_orig, band='y'):
    """
    Function to grab the number of nights corresponding to a sequence 

    Parameters
    ----------
    grp : pandas df
        Data to process.
    df_orig : pandas df
        Data to process.
    band : str, optional
        filter for the sequence. The default is 'y'.

    Returns
    -------
    res : pandas df
        Result.

    """

    print(grp.name, grp.name[0])

    dbName = grp.name[0]
    target_name = grp.name[1]
    year = grp.name[2]

    idx = df_orig['dbName'] == dbName
    idx &= df_orig['target_name'] == target_name
    idx &= df_orig['year'] == year
    idx &= df_orig[band] > 0

    sel_orig = pd.DataFrame(df_orig[idx])

    del df_orig

    b_ref = get_ref_sequence(grp, band)

    b_ref = clean_level(b_ref)
    sel_test = sel_orig.merge(b_ref, how='cross')

    sel_test['diff_nvisits'] = sel_test['nvisits']-sel_test['nvisits_ref']
    bands = 'grizy'
    for b in bands:
        sel_test['ratio_{}'.format(
            b)] = sel_test[b]/sel_test['{}_ref'.format(b)]

    # three types of nights
    dd = {}
    dd['frac_equal'] = [100.*get_val(sel_test, 'diff_nvisits', op.eq, 0)]
    dd['frac_plus'] = [100.*get_val(sel_test, 'diff_nvisits', op.gt, 0)]
    dd['frac_minus'] = [100.*get_val(sel_test, 'diff_nvisits', op.lt, 0)]

    res = pd.DataFrame.from_dict(dd)

    return res


def get_val(df, col, op, selvalue):
    """
    Function to estimate values

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    col : TYPE
        DESCRIPTION.
    op : TYPE
        DESCRIPTION.
    selvalue : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    idx = op(df[col], selvalue)
    sel = df[idx]

    return len(sel)/len(df)


"""
    seq = grp['seq_tot_{}'.format(band)].values[0]

    print(seq)

    spl = seq.split('-')

    bands = [sp[-1] for sp in spl]
    bands_ref = list(map(lambda el: el+'_ref', bands))
    nv_ref = list(map(int, [sp[:-1] for sp in spl]))
    nv_ref = list(map(lambda el: [el], nv_ref))
    dd = dict(zip(bands_ref, nv_ref))
    print(bands, nv_ref)
    b_ref = pd.DataFrame.from_dict(dd)
    print(b_ref)

    print(sel_orig.columns)
    print(sel_orig[['seq_tot', 'nvisits']])

    nnights_y = len(sel_orig)
    # select the sequence and nvisits_ref
    ida = sel_orig['seq_tot'] == seq
    print('allo', len(sel_orig[ida])/nnights_y)
    nvisits_ref = sel_orig[ida]['nvisits'].mean()

    # grab the nights with a higher/lower number of visits
    idp = sel_orig['nvisits'] > nvisits_ref
    sel_test = sel_orig[idp]

    sel_test = sel_test.merge(b_ref, how='cross')

    sel_test['diff_nvisits'] = nvisits_ref-sel_orig['nvisits']

    for b in bands:
        sel_test['ratio_{}'.format(
            b)] = sel_test['{}_ref'.format(b)]/sel_test[b]

    plot_stat_visits_vs_exp(sel_test, op.lt, '<')

    plt.show()

    print('plus', len(sel_orig[idp])/nnights_y,
          np.mean(sel_orig[idp]['nvisits']/nvisits_ref))

    idm = sel_orig['nvisits'] < nvisits_ref
    print('minus', len(sel_orig[idm])/nnights_y)

    print(test)
"""

parser = OptionParser(
    description='Script to analyse DDF visits on a nightly basis from pointings')

parser.add_option("--dbDir", type="str",
                  default='../ddf_visits_night',
                  help="file directory [%default]")
parser.add_option("--dbName", type="str",
                  default='desc_ddf_v4.2.1_10yrs',
                  help="OS name [%default]")
parser.add_option("--dbList", type="str",
                  default='dbList.csv',
                  help="dbList to process [%default]")
parser.add_option("--nproc", type=int,
                  default=8,
                  help="number of procs for multiprocessing [%default]")
parser.add_option("--configplot", type=str,
                  default='config_ana_selplot.csv',
                  help="configuration for the plot [%default]")

opts, args = parser.parse_args()

# Load parameters
dbDir = opts.dbDir
dbName = opts.dbName
dbList = opts.dbList
nproc = opts.nproc
config = opts.configplot

# load dbNames
df_db = pd.read_csv(dbList, comment='#')

data = pd.DataFrame()
for i, row in df_db.iterrows():
    fName = '{}/{}.hdf5'.format(dbDir, row['dbName'])

    dat_ = pd.read_hdf(fName)
    data = pd.concat((data, dat_))

print(data)

# ro = ana_seq(data)
dbNames = data['dbName'].unique().tolist()

params = {}

params['timescale'] = 'year'
params['data'] = data

ro = multiproc(dbNames, params, ana_seq_multi, nproc)


print(ro.columns)

"""
plot_all(ro,dbName,field='DD:COSMOS',season=1)

plt.show()
"""
dft = ro.groupby(['dbName', 'target_name', 'year']).apply(
    lambda x: summary_seq(x)).reset_index()

print(dft.columns)


# plots here
# plot_fields(dft, config)

print(dft)

rr = get_stats(dft, data)

print(rr.columns)

plot_stat(rr, config)

# analysis_sequences(dft, data)

# analyze_simu_exp(data)
