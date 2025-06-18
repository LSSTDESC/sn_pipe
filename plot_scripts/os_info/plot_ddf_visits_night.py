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

    ccols_m = ['target_name', timescale,'dbName']
    ccols = ccols_m+['seq_tot']

    dfb = df.groupby(ccols)[ccols].apply(
        lambda x: get_nvisits(x)).reset_index()
    dfb = clean_level(dfb)
    
    
    bands = 'ugrizy'
    colsb = ccols+list(bands)
    #dfb = dfb.merge(df[colsb], left_on=ccols,right_on=ccols,suffixes=['',''])
    
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

    dd[thevar_name] = [grp[thevar].mean()]

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


def plot_all(ro,dbName,field='DD:COSMOS',season=3):
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
    
    idx = ro['u'] ==0
    idx &= ro['y'] == 0
    """
    plot_seq_frac(ro[idx], dbName, field=field, season=season,
                  what='nnights', legy='Number of nights')
    """
    plot_seq_frac(ro[idx], dbName, field=field, season=season,
                  what='seq_frac')
    
def calc_summary(grp,col='y'):
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
    
    rr = selb[['seq_tot', 'seq_frac',col]][:1]
    rr['nvisits_{}'.format(col)] = sel[col].sum()
    rr['frac_{}'.format(col)] = sel['seq_frac'].sum()
    
    rr = rr.rename(columns={'seq_tot':'seq_tot_{}'.format(col),
                    'seq_frac':'seq_frac_{}'.format(col)})
    return rr

def summary_seq(grp):
    """
    function to estimate summery results

    Parameters
    ----------
    grp : pandas df
        Data to process.

    Returns
    -------
    res : pandas df
        output data.

    """
   
    rr = calc_summary(grp,'y')
   
    bb = calc_summary(grp,'u')
    
    res = rr.merge(bb,how='cross')
    
    return res
    
 
def plot_summary(data,field='DD:COSMOS'):
    
    
    fig, ax = plt.subplots(figsize=(12,8))
    
    idx = data['target_name'] == field
    
    sel = data[idx]

    dbNames = sel['dbName'].unique()

    for dbName in dbNames:
        io = sel['dbName'] == dbName
        selb = sel[io]
        ax.plot(selb['year'],selb['seq_tot_y'])
    
    ax.grid(visible=True)
    
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
                  help="dbList to process[%default]")

opts, args = parser.parse_args()

# Load parameters
dbDir = opts.dbDir
dbName = opts.dbName
dbList = opts.dbList

# load dbNames
df_db = pd.read_csv(dbList, comment='#')

data = pd.DataFrame()
for i, row in df_db.iterrows():
    fName = '{}/{}.hdf5'.format(dbDir, row['dbName'])

    dat_ = pd.read_hdf(fName)
    data = pd.concat((data,dat_))

print(data)

ro = ana_seq(data)

print(ro.columns)

"""
plot_all(ro,dbName,field='DD:COSMOS',season=1)

plt.show()
"""
dft = ro.groupby(['dbName','target_name','year']).apply(lambda x:summary_seq(x)).reset_index()

plot_summary(dft)
print(dft)




plt.show()
# analyze_simu_exp(data)
