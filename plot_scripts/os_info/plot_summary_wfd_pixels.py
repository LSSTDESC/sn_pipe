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
from sn_plotter_analysis import plt,filtercolors
import numpy as np
import healpy as hp
from sn_tools.sn_obs import season
from sn_plotter_tools.plot_tools import plot_xy

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
    
    res = df.groupby(['dbName']).apply(lambda x:stat(x),
                                        include_groups=False).reset_index()
    res['dbName_red'] = res['dbName'].str.split('_v5.3.7').str.get(0)
    print(res.columns)
    plot_vs(res,'dbName_red','','nvisits_10yrs_mean',
            '$N_{visits}$ [10 yrs]','nvisits_10yrs_std')

def slices(data,var='pixDec',varb='Dec',
           varsum=['nvisits',
                   'nvisits_u','nvisits_g',
                   'nvisits_r','nvisits_i',
                   'nvisits_z','nvisits_y',
                   'density'], 
           bins=np.arange(-90.,90.,10.)):
    
    
    group = data.groupby(pd.cut(data[var], bins),observed=False)
    nvisits_sum = data['nvisits'].sum()
    nvisits_med = data['nvisits'].median()
    nvisits_mean = data['nvisits'].mean()
    
    idx = data['pixDec'] <= 10
    nvisits_min = data[idx]['nvisits'].min()
    
    
    res = pd.DataFrame()
    for name,gr in group:
        dd = {}
        dd['{}_min'.format(varb)] = [float(name.left)]
        dd['{}_max'.format(varb)] = [float(name.right)]
        for vv in varsum:
            #dd[vv] = [int(gr[vv].sum())]
            dd[vv] = [gr[vv].mean()]
            dd['{}_std'.format(vv)] = [gr[vv].std()]
            dd['diff_{}'.format(vv)] = [gr[vv].median()/nvisits_med]
            """
            dd['frac_{}'.format(vv)] = [100.*gr[vv].sum()/nvisits_sum]
            dd['diff_{}'.format(vv)] = [gr[vv].median()-nvisits_med]
            dd['doff_{}'.format(vv)] = [gr[vv].mean()-nvisits_mean]
            #dd['daff_{}'.format(vv)] = [gr[vv].sum()-nvisits_min]
            """
        rr = pd.DataFrame.from_dict(dd)
            
        res = pd.concat((res,rr))
    
    """
    res['Dec'] = 0.5*(res['Dec_min']+res['Dec_max'])
    res = res.sort_values(by=['Dec'])
    
    for vv in varsum:
        res['diff_{}'.format(vv)]=res[vv].diff()
        norm = res['{}_std'.format(vv)]/np.sqrt(len(res))
        res['diff_std_{}'.format(vv)]=res[vv].diff()/norm
    """
    return res

def plot_slices_db(res,xvar='Dec_min',xleg='Dec [deg]',yvar='nvisits',
                    yleg='$N_{visits}$',yerr=None):
    
    dbNames = res['dbName'].unique()
    
    years = range(1, 13)
    marks = ['o', 's', 'P', 'v', '^', 'p']*2
    lsb = ['solid']*6+['dashed']*6
    clrs = ['k']*2+['r']*2+['g']*2+['b']*2+['m']*2+['orange']*2

    mm = dict(zip(years, marks))
    ll = dict(zip(years, lsb))
    ccol = dict(zip(years, clrs))
   
    for dbName in dbNames:
        idx = res['dbName'] == dbName
        
        sel = res[idx]
        
        years = sel['year'].unique()
        dbName_red = dbName.split('_v5.3.7')[0]
        fig, ax = plt.subplots(figsize=(14,8))
        fig.suptitle(dbName_red)
        fig.subplots_adjust(right=0.85)
        
        for y in years:
            
            idxb = sel['year'] == y
            
            selb = sel[idxb]
            
            ls = ll[y]
            color = ccol[y]
            marker = mm[y]
            
            #selb['rat_u'] = selb['nvisits_u']/selb['nvisits']
            #yvar = 'rat_u'
            plot_xy(selb,xvar=xvar,xleg=xleg,yvar=yvar,
                    yerr=yerr,
                    yleg=yleg,fig=fig,ax=ax,
                    label='year {}'.format(y),
                    color=color,
                    marker=marker,msize=10,
                    linestyle=ls,figtit='',markevery=3)
            
        
        ax.set_xlabel(r'{}'.format(xleg))
        ax.set_ylabel(r'{}'.format(yleg))
        ax.grid(visible=True)
        ax.legend(bbox_to_anchor=(1.0, 0.7),
                  ncol=1, fontsize=15, frameon=False)
        
def plot_dec_slices_db(df,dbName='baseline_v5.3.7_10yrs',
                       year=1,
                       bins=np.arange(-80,30,2)):
    
    idx = df['dbName'] == dbName
    idx &= df['year'] == year
    #df['density'] = df['nvisits']/pixSize
    
    group = df.groupby(pd.cut(df['pixDec'], bins),observed=False)
   
    for name,gr in group:
        fig, ax = plt.subplots()
        fig.suptitle(name)
        ax.hist(gr['density'],histtype='step',bins=40)
        
        plt.show()
   
def survey_spikes(grp,thresh=20,varx='diff_std_density'):
    
    dd = {}
    idx = np.abs(grp[varx]) >= thresh
    
    dd['nspikes'] = [len(grp[idx])]
    
    return pd.DataFrame.from_dict(dd)
    
    
def dec_extension(grp,thresh=20,varx='diff_nvisits'):

    idx = grp[varx] >= thresh
    
    sel = grp[idx]
    
    nn = grp.name
    
    nstripes = 0
    stripe_depth_max=0
    stripe_depth_min=0
    
    if len(sel) > 0:
        
        ro = []
    
        selseas = season(sel.to_records(index=False),
                         season_gap=4,mjdCol='Dec_min')
        
        selseas = pd.DataFrame.from_records(selseas)
    
        print(grp.name)
    
        dec_extens = selseas['season'].unique()
        
        for dec in dec_extens:
            idxb = selseas['season'] == dec
            
            rr = selseas[idxb]
            
            rr = rr.sort_values(by=['Dec_min'])
            
            x_min = rr['Dec_min'].min()
            x_max = rr['Dec_min'].max()
            
            x_length = x_max-x_min
            
            if x_length> 0:
                ro.append((dec,x_length,x_min,x_max))
            
    
        df = pd.DataFrame(ro,columns=['id_stripe','stripe_depth','dec_min','dec_max'])
        print('summary',df)
    
    
        nstripes = len(df)
        stripe_depth_max = df['stripe_depth'].max()
        stripe_depth_min = df['stripe_depth'].min()
        
    rr = [(nstripes,stripe_depth_max,stripe_depth_min)]

    res = pd.DataFrame(rr,
                       columns=['nstripes','stripe_depth_max','stripe_depth_min'])

    return res

def plot_summary_slices(df, config,xvar='year',xleg='year',
                        yvar='nstripes',yleg='$N_{dec}^{stripes}$'):
    
    dbNames = df['dbName'].unique()
    
    fig, ax = plt.subplots(figsize=(12,8))
    fig.subplots_adjust(right=0.75)
    
    idx = df['nstripes'] <=1
    """
    idx &= df['stripe_depth_max'] > 0.
    idx &= df['stripe_depth_min'] > 0.
    """
    print(df[idx][['dbName','year']])
    
    #idxb = df['dbName'] =='three_stripes_kr_v5.3.7_10yrs'
    idxb = df['dbName'] =='baseline_v5.3.7_10yrs'
    print(df[idxb][['dbName','year','nstripes',
                    'stripe_depth_max','stripe_depth_min']])
    
    for dbName in dbNames:
        
        idx = df['dbName'] == dbName
        
        sel = df[idx]
        
        idxb = config['dbName'] == dbName
        selconf = config[idxb]
        
        ls = selconf['ls'].values[0]
        mm = selconf['marker'].values[0]
        color = selconf['color'].values[0]
        dbName_red = selconf['dbName_plot'].values[0]
        
        plot_xy(sel,xvar=xvar,xleg=xleg,
                yvar=yvar,yleg=yleg,
                fig=fig,ax=ax,label=dbName_red,
                color=color,marker=mm,linestyle=ls)
        
        
    ax.grid(visible=True)
    ax.set_xlabel(r'{}'.format(xleg))
    ax.set_ylabel(r'{}'.format(yleg))
    
   
    ax.legend(bbox_to_anchor=(1.0, 0.7),
              ncol=1, fontsize=12, frameon=False)    
        
    
    
    
    


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
parser.add_option('--config', type=str,
                  default='config_ana_selplot_fbs_5.3_nov2026.csv',
                  help='plots to display [%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbList = opts.dbList
nside = opts.nside
ebvofMW_max = opts.ebvofMW_max
nproc = opts.nproc
plots = opts.plots.split(',')
config = opts.config

pixSize = hp.nside2pixarea(nside, degrees=True)

dbNames = pd.read_csv(dbList,comment='#')['dbName'].to_list()

df = load_sel_dust(dbDir, dbNames, 'WFD','WFD',nside=nside,
                   ebvofMW_max=ebvofMW_max)
idx = df['year'] <= 10
idx &= df['year'] >=1
#remove hot spots per year
idx &= df['nvisits'] <= 400
#idx &= df['cadence'] <= 4

df = df[idx]
    
df['density'] = df['nvisits']/pixSize

print(df.columns)

"""
cols = ['dbName','healpixID']
nvisits = df.groupby(cols).apply(lambda x:nvisits_survey(x),
                                 include_groups=False).reset_index()
"""
params = {}
params['data'] = df

nvisits = multiproc(df['dbName'].unique(),
                    params,nvisits_multiproc,nproc=nproc)
cols = ['dbName','healpixID']
df = df.merge(nvisits,left_on=cols,right_on=cols)

#select pixels with at 400<nvisits_10yrs < 1200

idx = df['nvisits_10yrs'] >= 400
idx &= df['nvisits_10yrs'] <= 1200

df = df[idx]

if 'nvisits_10yrs' in plots:
    plot_nvisits(df)

col = 'nvisits'
varsum = [col]
for b in 'ugrizy':
    varsum.append('{}_{}'.format(col,b))
varsum.append('density')

if 'slices' in plots:
    dec_min=-80
    dec_max = 30.
    delta_dec = 2.
    bins = np.arange(dec_min,dec_max+delta_dec,delta_dec)
    """
    dbName = 'three_stripes_kr_v5.3.7_10yrs'
    plot_dec_slices_db(df,dbName=dbName,year=2,bins=bins)
    
    """
    cols = ['dbName','year']
    res = df.groupby(cols).apply(lambda x: slices(x,varsum=varsum,bins=bins),
                                              include_groups=False).reset_index()
    print(res.columns)
    plot_slices_db(res,yvar='diff_nvisits',yleg='$\Delta N_{visits}$')
    
    resb = res.groupby(cols).apply(lambda x: dec_extension(x,thresh=1.1),
                                              include_groups=False).reset_index()
    
    print(resb)
    conf = pd.read_csv(config,comment='#')
    plot_summary_slices(resb,conf)
    """
    plot_summary_slices(resb,conf,yvar='stripe_depth_max',yleg='depth max')
    plot_summary_slices(resb,conf,yvar='stripe_depth_min',yleg='depth min')
    """
    """
    resb = res.groupby(cols).apply(lambda x: survey_spikes(x,thresh=1),
                                              include_groups=False).reset_index()
    
    print(resb)
    """
plt.show()