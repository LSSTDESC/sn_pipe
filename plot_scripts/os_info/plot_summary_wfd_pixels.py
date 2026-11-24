#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 10:19:22 2026

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import pandas as pd
from sn_plotter_os_info.os_info_util import load_sel_dust
from sn_tools.sn_utils import multiproc
#import matplotlib.pyplot as plt
from sn_plotter_analysis import plt
import numpy as np

def nvisits_survey(grp,timescale='year'):
    """
    total number of visits

    Parameters
    ----------
    grp : pandas df
        Data to process.
    timescale : str, optional
        time scale. The default is 'year'.

    Returns
    -------
    res : TYPE
        DESCRIPTION.

    """
    
    nvisits = [grp['nvisits'].sum()]
    
    res = pd.DataFrame(nvisits,columns=['nvisits_10yrs'])
    
    return res

def nvisits_multiproc(dbNames,params,j=0,output_q=None):
    """
    nvisits estimation using multiprocessing

    Parameters
    ----------
    dbNames : list(str)
        list of OS to process.
    params : dict
        parameters.
    j : int, optional
        internal int for multiprocessing. The default is 0.
    output_q : multiprocessing queue, optional
        where to store the data. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    data = params['data']
    
    idx = data['dbName'].isin(dbNames)
    
    sel = data[idx]
    
    cols = ['dbName','healpixID']
    res = sel.groupby(cols).apply(lambda x:nvisits_survey(x),
                                 include_groups=False).reset_index()
    
    if output_q is not None:
        return output_q.put({j: res})
    else:
        return res
    
def stat(grp,col='nvisits_10yrs'):
    """
    Function to grab some stat

    Parameters
    ----------
    grp : pandas df
        Data to process.
    col : str, optional
        column for the stat. The default is 'nvisits_10yrs'.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    idx = grp['nvisits_10yrs'] >= 400
    idx &= grp['nvisits_10yrs'] <= 1200
    
    grp = grp[idx]
    dd = {}
    
    dd['{}_mean'.format(col)] = [grp[col].mean()]
    dd['{}_std'.format(col)] = [grp[col].std()]
    dd['{}_med'.format(col)] = [grp[col].median()]
    
    return pd.DataFrame.from_dict(dd)

def plot_vs(data,varx,legx,vary,legy,errvary):
    """
    plot function

    Parameters
    ----------
    data : pandas df
        Data to plot.
    varx : str
        x-axis variable.
    legx : str
        x-axis legend.
    vary : str
        y-axis variable.
    legy : str
        y-axis legend.
    errvary : str
        y-axis err variable.

    Returns
    -------
    None.

    """
    
    
    fig, ax = plt.subplots(figsize=(14,10))
    fig.subplots_adjust(top=0.99,bottom=0.17)
    
    yerr='None'
    
    if errvary!='':
        yerr = data[errvary]
        
    ax.errorbar(data[varx],data[vary],yerr=yerr,color='k',marker='o')
    
    
    if legx != '':
        ax.set_xlabel(legx)
    else:
        #ax.tick_params(axis='x', labelrotation=20., labelsize=12)
        plt.setp(ax.get_xticklabels(), rotation=30,ha="right", 
                 rotation_mode="anchor", fontsize=12)

    if legy != '':
        ax.set_ylabel(legy)
        
    ax.grid(visible=True)
    
def plot_nvisits(df):
    """
    Function to plot the number of visits after 10 yrs

    Parameters
    ----------
    df : pandas df
        Data to plot.

    Returns
    -------
    None.

    """
    
    params = {}
    params['data'] = df
    
    nvisits = multiproc(df['dbName'].unique(),
                        params,nvisits_multiproc,nproc=nproc)
    
    res = nvisits.groupby(['dbName']).apply(lambda x:stat(x),
                                        include_groups=False).reset_index()
    res['dbName_red'] = res['dbName'].str.split('_v5.3.7').str.get(0)
    print(res.columns)
    plot_vs(res,'dbName_red','','nvisits_10yrs_mean',
            '$N_{visits}$ [10 yrs]','nvisits_10yrs_std')

def slices(data,var='pixDec',varb='Dec',
           varsum=['nvisits',
                   'nvisits_u','nvisits_g',
                   'nvisits_r','nvisits_i',
                   'nvisits_z','nvisits_y',],bins=np.arange(-90.,90.,10.)):
    
    
    group = data.groupby(pd.cut(data[var], bins),observed=False)
    
    res = pd.DataFrame()
    for name,gr in group:
        dd = {}
        dd['{}_min'.format(varb)] = [float(name.left)]
        dd['{}_max'.format(varb)] = [float(name.right)]
        for vv in varsum:
            dd[vv] = [int(gr[vv].sum())]
            
        rr = pd.DataFrame.from_dict(dd)
            
        res = pd.concat((res,rr))
    
    return res

parser = OptionParser(description='Script to plot pixel level OS infos')

parser.add_option('--dbList', type=str, default='list_db.csv',
                  help='dbName to process [%default]')
parser.add_option('--dbDir', type=str, default='../wfd_pixels_nov2026',
                  help='dbDir of the OS to process [%default]')
parser.add_option('--nside', type=int, default=64,
                  help='healpix nside parameter [%default]')
parser.add_option('--ebvofMW_max', type=float, default=0.25,
                  help='E(B-V) max [%default]')
parser.add_option('--nproc', type=int, default=8,
                  help='nproc for multiprocessing [%default]')
parser.add_option('--plots', type=str, default='nvisits_10yrs,slices',
                  help='plots to display [%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbList = opts.dbList
nside = opts.nside
ebvofMW_max = opts.ebvofMW_max
nproc = opts.nproc
plots = opts.plots.split(',')

dbNames = pd.read_csv(dbList,comment='#')['dbName'].to_list()

df = load_sel_dust(dbDir, dbNames, 'WFD','WFD',nside=nside,
                   ebvofMW_max=ebvofMW_max)
idx = df['year'] <= 10
idx &= df['year'] >=1
    
df = df[idx]
    

print(df.columns)

"""
cols = ['dbName','healpixID']
nvisits = df.groupby(cols).apply(lambda x:nvisits_survey(x),
                                 include_groups=False).reset_index()
"""

if 'nvisits_10yrs' in plots:
    plot_nvisits(df)

col = 'nvisits'
varsum = [col]
for b in 'ugrizy':
    varsum.append('{}_{}'.format(col,b))

if 'slices' in plots:
    res = df.groupby(['dbName','year']).apply(lambda x: slices(x,
                                                               varsum=varsum),
                                              include_groups=False).reset_index()
    
    plot_slices_year(res)
    

plt.show()