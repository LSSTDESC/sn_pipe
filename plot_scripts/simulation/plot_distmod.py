#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 10:23:19 2026

@author: philippe.gris@clermont.in2p3.fr
"""
import glob
import pandas as pd
from sn_analysis.sn_selection import selection_criteria
import numpy as np
import matplotlib.pyplot as plt
from sn_tools.sn_io import checkDir

def process_config(fDir,obs_coadd,lc_coadd,dbName,runType,sellist,config,outDir):
    
    fDir = '{}/*_{}_{}'.format(fDir,obs_coadd,lc_coadd)
    
    fis = glob.glob(fDir)
    
    print(fis,len(fis))    
    
    df = pd.DataFrame()
    
    for fi in fis:
        fName = '{}/{}/{}/*.hdf5'.format(fi,dbName,runType)
        fisb = glob.glob(fName)
        
        for fib in fisb:
            rr = pd.read_hdf(fib)
            rr = complete_data(rr,sellist)
            df = pd.concat((df,rr))
            
    outName = '{}/z_summary_{}_{}_{}.hdf5'.format(outDir,config,obs_coadd,lc_coadd)
    
    df.to_hdf(outName,key='sn')

def complete_data(df,sellist):
    
    
    from sn_analysis.sn_tools import complete_df
    from sn_analysis.sn_selection import select
 
    df['sigma_color'] = np.sqrt(df['Cov_colorcolor'])
 
    df = complete_df(df)
 
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

def plot(df):
    
    fig, ax = plt.subplots()


    seasons = df['season'].unique()
    
    for seas in seasons:
        idx = df['season'] == seas
        sel = df[idx]
        plot_indiv(sel,fig,ax,leg='season {}'.format(seas))
        
    ax.legend()

def plot_indiv(df,fig=None,ax=None,leg=''):
    
    if fig is None:
        fig, ax = plt.subplots()
        
    df = df.sort_values(by=['z'])
    ax.errorbar(df['z'],df['diff_mu'],yerr=df['sigma_mu'],label=leg)
    

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

conf = 'confe'
"""
for i in range(len(obs_coadd)):
    dfa = process_config(fDir, obs_coadd[i], lc_coadd[i], 
                         dbName, runType,sellist,conf,outDir)
"""
for i in range(len(obs_coadd)):
    fName = '{}/z_summary_{}_{}_{}.hdf5'.format(outDir,conf,obs_coadd[i],lc_coadd[i])
    
    df = pd.read_hdf(fName)

    dfb = df.groupby(['z','season']).apply(lambda x: calc(x),include_groups=False).reset_index()
    plot(dfb)
    

plt.show()

"""
dfa = pd.read_hdf('test.hdf5')
print(dfa)

dfb = dfa.groupby(['z','season']).apply(lambda x: calc(x)).reset_index()

print(dfb)

plot(dfb)

plt.show()

"""