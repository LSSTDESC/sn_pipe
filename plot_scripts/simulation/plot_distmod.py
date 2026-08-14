#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 10:23:19 2026

@author: philippe.gris@clermont.in2p3.fr
"""
import glob
from optparse import OptionParser
import pandas as pd
from sn_analysis.sn_selection import selection_criteria
import numpy as np
import matplotlib.pyplot as plt
from sn_tools.sn_io import checkDir
from sn_tools.sn_utils import clean_level

def process_config(fDir,obs_coadd,lc_coadd,dbName,runType,
                   sellist,config,outDir):
    
    from sn_tools.sn_utils import multiproc
    
    ll = get_files(fDir, obs_coadd, lc_coadd, dbName, runType, config)
    
    print(len(ll))
    
    params = {}
    
    params['sellist'] = sellist
    
    
    df = multiproc(ll,params,process_multiproc,nproc=8)
    
    outName = '{}/z_summary_{}_{}_{}.hdf5'.format(outDir,config,
                                                  obs_coadd,lc_coadd)
    
    df.to_hdf(outName,key='sn')

def get_files(fDir,obs_coadd,lc_coadd,dbName,runType,config):
    
    if config == 'confe':
        fDir = '{}/z_*_{}_{}'.format(fDir,obs_coadd,lc_coadd)
    else:
        fDir = '{}/{}_z_*_{}_{}'.format(fDir,config,obs_coadd,lc_coadd)
    
    fis = glob.glob(fDir)
    
    list_files = []
    for fi in fis:
        fName = '{}/{}/{}/*.hdf5'.format(fi,dbName,runType)
        fisb = glob.glob(fName)
        list_files += list(fisb)
    
    return list_files

def process_multiproc(toproc, params, j=0, output_q=None):
    
    sellist = params['sellist']
    
    df = pd.DataFrame()
    for pp in toproc:
        rr = pd.read_hdf(pp)
        rr = complete_data(rr,sellist)
        df = pd.concat((df,rr))
    
    if output_q is not None:
        return output_q.put({j: df})
    else:
        return df

def complete_data(df,sellist=[]):
    
    
    from sn_analysis.sn_tools import complete_df
    from sn_analysis.sn_selection import select
 
    df['sigma_color'] = np.sqrt(df['Cov_colorcolor'])
 
    df = complete_df(df)
 
    if sellist:
        df = pd.DataFrame(select(df,list_sel=sellist))
 
    for vv in ['x1','color']:
        vdiff = '{}_fit'.format(vv)
        vsigma = 'sigma_{}'.format(vv)
        df['pull_{}'.format(vv)] = (df[vv]-df[vdiff])/df[vsigma]
        df['diff_{}'.format(vv)] = (df[vv]-df[vdiff])
    
    return df

def calc(grp):
    
    grp['c_mu'] = 1./grp['sigma_mu']**2
    dd = {}
    dd['diff_mu'] = [grp['diff_mu'].mean()]
    dd['sigma_mu'] = [(np.sum(grp['c_mu']))**-0.5]
    
    
    res = pd.DataFrame.from_dict(dd)
    
    return res

def plot(df,xvar='z',yvar='diff_mu',yvar_err='sigma_mu',figtit=''):
    
    fig, ax = plt.subplots(figsize=(12,8))

    fig.suptitle(figtit)
    seasons = df['season'].unique()
    
    for seas in seasons:
        idx = df['season'] == seas
        sel = df[idx]
        plot_indiv(sel,xvar,yvar,yvar_err,fig,ax,leg='season {}'.format(seas))
        
    ax.legend()

def plot_indiv(df,xvar,yvar,yvar_err,fig=None,ax=None,leg=''):
    
    if fig is None:
        fig, ax = plt.subplots()
        
    df = df.sort_values(by=[xvar])
    ax.errorbar(df[xvar],df[yvar],yerr=df[yvar_err],label=leg)
    
def binIt(grp,xvar='z',yvar='diff_mu',
                      yvar_err='sigma_mu',bins=np.arange(0.,1.1,0.01)):
    
    from sn_analysis.sn_calc_plot import bin_it_weighted
    rr = bin_it_weighted(grp,xvar=xvar,yvar=yvar,
                          yvar_err=yvar_err,bins=bins)
    
    rr = pd.DataFrame(rr)
    
    rr = rr.reset_index()

    rr = clean_level(rr)
    
    return rr

def shape_data(confs,obs_coadd,lc_coadd,outDir):
    
    bins = np.arange(0.,1.1,0.05)
    
    ccols = ['z','season']
    dd = {}
    for conf in confs:
        for i in range(len(obs_coadd)):
            fName = '{}/z_summary_{}_{}_{}.hdf5'.format(outDir,conf,
                                                        obs_coadd[i],lc_coadd[i])
    
            df = pd.read_hdf(fName)
            
            dfb = df.groupby(ccols).apply(lambda x: calc(x),include_groups=False).reset_index()
            figtit = '{} - obs_coadd {} - lc_coadd {}'.format(conf,
                                                              obs_coadd[i],
                                                              lc_coadd[i])
            dfb = pd.DataFrame(dfb)
            dfb = dfb.reset_index()

            dfb = clean_level(dfb)
            
            dfc = dfb.groupby(['season']).apply(lambda x: binIt(x,
                                                                xvar='z',yvar='diff_mu',
                                  yvar_err='sigma_mu',bins=bins),include_groups=False).reset_index()
            dictName = '{}_{}_{}'.format(conf,obs_coadd[i],lc_coadd[i])
            dd[dictName] = dfc
            
    return dd

def load_conf(theDir,conf,obs_coadd,lc_coadd,season):
    
    fName = '{}/z_summary_{}_{}_{}.hdf5'.format(theDir,conf,obs_coadd,lc_coadd)
    
    df = pd.read_hdf(fName)
    
    idx = df['season'] == season
    
    return pd.DataFrame(df[idx])
    
def load_conf_from_dict(dd,conf,obs_coadd,lc_coadd,season):
    
    key = '{}_{}_{}'.format(conf,obs_coadd,lc_coadd)
    
    print('loading',key)
    df = dd[key]
    
    idx = df['season'] == season
    
    return pd.DataFrame(df[idx])
    
def load_conf_orig(fDir,dbName,runType,z,conf,obs_coadd,lc_coadd,season):
    
    if conf== 'confe':
        fDir = '{}/z_{}_{}_{}'.format(fDir,z,obs_coadd,lc_coadd)
    else:
        fDir = '{}/{}_z_{}_{}_{}'.format(fDir,conf,z,obs_coadd,lc_coadd)
    
    fName = '{}/{}/{}/SN_*_{}.hdf5'.format(fDir,dbName,runType,season)
    
    fis = glob.glob(fName)
    df = pd.read_hdf(fis[0])
    
    return df

parser = OptionParser('script to plot LSST SN surveys')
"""
parser.add_option('--dataDir', type=str,
                  default='../sn_summary_surveys',
                  help='data directory [%default]')
"""
parser.add_option('--action', type=str, default='plot',
                  help='what to do (process,plot) [%default]')

opts, args = parser.parse_args()

pp = vars(opts)


#selection criteria
sellist = selection_criteria()['G10_JLA']


fDir = '../prod_single'

obs_coadd = 1
lc_coadd = 0
dbName = 'baseline_v5.3.0_10yrs'
runType  ='DDF_spectroz'

outDir = '../summary_single/{}/{}'.format(dbName,runType)
checkDir(outDir)

obs_coadd = [1,0,0]
lc_coadd = [0,1,2]

confs= ['confe','confd']

if pp['action'] == 'process':
    for conf in confs:
        for i in range(len(obs_coadd)):
            dfa = process_config(fDir, obs_coadd[i], lc_coadd[i], 
                                 dbName, runType,sellist,conf,outDir)

if pp['action'] == 'plot':
    # reshape data
    dd = shape_data(confs, obs_coadd, lc_coadd, outDir)
    """
    for key,vals in dd.items():
        #print(vals.columns)
        plot(vals,xvar='z',yvar='diff_mu_weighted_mean',
             yvar_err='diff_mu_sigma',figtit=key)
    """

    
    configs = ['confe','confe','confe']
    obs_coadds = [1,0,0]
    lc_coadds = [0,1,2]
    
    season = 5
    
    fig, ax = plt.subplots(figsize=(12,8))

    for i in range(len(configs)):
        dfa = load_conf_from_dict(dd,configs[i],obs_coadds[i],lc_coadds[i],season)
        plot_indiv(dfa,xvar='z',yvar='diff_mu_weighted_mean',
                   yvar_err='diff_mu_sigma',fig=fig,ax=ax)
    
    
    plt.show()
    
if pp['action'] == 'compare':

    configs = ['confe','confe','confe']
    obs_coadds = [1,0,0]
    lc_coadds = [0,1,2]
    season = 5
    z = 0.8
    
    for i in range(len(configs)):
        df = load_conf(outDir,configs[i],obs_coadds[i],lc_coadds[i],season)
        idx = df['z'] == z
        dfa  =df[idx]
        print(dfa['SNID'],configs[i],obs_coadds[i],lc_coadds[i])
    
if pp['action'] == 'compare_orig':
    
    configs = ['confe','confe','confe']
    obs_coadds = [1,0,0]
    lc_coadds = [0,1,2]
    season = 5
    z = 0.8
    
    fig, ax = plt.subplots()
    dd = {}
    for i in range(len(configs)):
        confdis = '{}_{}_{}'.format(configs[i],obs_coadds[i],lc_coadds[i])
        
        df = load_conf_orig(fDir,dbName,runType,z,
                            configs[i],obs_coadds[i],lc_coadds[i],season)
        df = complete_data(df)
        
        dd['conf_{}'.format(i)] = df
        """        
        thevar = 'pull_color'
        idx = np.abs(df[thevar]) <= 5
        
        sel = df[idx]
        ax.hist(sel[thevar],histtype='step',label=confdis)

        print(confdis,len(sel),sel[thevar].mean(),sel[thevar].std())
        """
    """
    ax.legend()
    plt.show()
    """
    
    ddb = dd['conf_0'].merge(dd['conf_2'],left_on=['SNID'],right_on=['SNID'])
    
    print(ddb[['SNID','x1_x','x1_y','color_x','color_y']])
    
"""
dfa = pd.read_hdf('test.hdf5')
print(dfa)

dfb = dfa.groupby(['z','season']).apply(lambda x: calc(x)).reset_index()

print(dfb)

plot(dfb)

plt.show()

"""