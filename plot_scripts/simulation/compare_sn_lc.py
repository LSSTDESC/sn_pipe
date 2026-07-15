#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 11:11:44 2026

@author: philippe.gris@clermont.in2p3.fr
"""
import glob
from astropy.table import Table, vstack
from sn_tools.sn_io import Read_LightCurve, get_meta
import pandas as pd
import numpy as np
from sn_plotter_simu.visuLC import lc_sn
import matplotlib.pyplot as plt
from sn_analysis.sn_tools import complete_df
from sn_analysis.sn_selection import selection_criteria,select

def get_metadata(theDir):
    
    #grab simu files

    file_path = '{}/Simu*'.format(theDir)

    fis = glob.glob(file_path)

    meta = Table()
    for fi in fis:
        fName = fi.split('/')[-1]
        tt = get_meta('None',fName,theDir)
        meta = vstack([meta,tt])
    
    return meta

def get_sndata(fDir,fName,sellist):
    
    df =pd.read_hdf('{}/{}'.format(fDir,fName))
    df['sigma_color'] = np.sqrt(df['Cov_colorcolor'])
    
    df = complete_df(df)
    
    df = select(df,list_sel=sellist)
    
    for vv in ['x1','color']:
        df['pull_{}'.format(vv)] = (df[vv]-df['{}_fit'.format(vv)])/df['sigma_{}'.format(vv)]
        
    print(df.columns)
    
    return df

def get_info(meta,snid,sn_data):
    
    idx = meta['SNID'] == snid
    
    sel_meta = meta[idx]
    
    lcDir = sel_meta['lc_dir'].value[0]
    lcName = sel_meta['lc_fileName'].value[0]
    
    #grab LC
    lc_plus_sn = lc_sn(lcDir,lcName,sn_data)
        
    lc_plus_sn.get_infos(snid)
    
    return lc_plus_sn
    
def get_info_sn(sndata,snid,ccols=['SNID','x1','color','z','daymax',
                                   'x1_fit','color_fit',
                                   'sigma_x1','sigma_color',
                                   'diff_mu','sigma_mu',
                                   'pull_x1','pull_color','chisq_red',
                                   't0_fit','sigma_t0']):
    
    idx = sndata['SNID'] == snid
    
    sel = sndata[idx]
    
    return pd.DataFrame(sel[ccols])

def plot_hist(df_tot,var=['pull_x1_x','pull_x1_y']):
    
    fig, ax = plt.subplots()
    vv = '_'.join(var[0].split('_')[:-1])
    fig.suptitle(vv)
    idx = np.abs(df_tot['pull_x1_y']) < 1.e6
    #idx &= np.abs(df_tot['pull_x1_x']) < 5.
    sel = df_tot[idx]
    
    print('effi',len(df_tot),len(sel))

    for vv in var:
        ax.hist(sel[vv],histtype='step',bins=30)
        print('stat',vv,sel[vv].mean(),sel[vv].std(),sel[vv].median())
   
def compare_lc(lc_sn_a,lc_sn_b):
    
    lc_a = lc_sn_a.lc_plot['lc']
    lc_b = lc_sn_b.lc_plot['lc']
    
    
    df_a = lc_a[['filter','time','flux','fluxerr','zp','airmass']].to_pandas()
    df_b = lc_b[['filter','time','flux','fluxerr','zp','airmass']].to_pandas()
    
    df_a = df_a.round({'time':6})
    df_b = df_b.round({'time':6})
    
    df_c = df_a.merge(df_b,left_on=['filter','time','airmass'],
                      right_on=['filter','time','airmass'])
    
    
    print(df_c)
    
    df_c['fluxerr_ratio'] = df_c['fluxerr_x']/df_c['fluxerr_y']
    df_c['delta_zp'] = df_c['zp_x']-df_c['zp_y']
    
    fig, ax = plt.subplots()
    
    var = 'delta_zp'
    idx = df_c['flux_x'] > 0. 
    idx &= df_c['flux_y'] > 0.
    ax.hist(df_c[idx][var],histtype='step',bins=40)
    
    
    for b in df_c['filter'].unique():
        idx = df_c['filter'] == b
        idx &= df_c['flux_x'] > 0. 
        idx &= df_c['flux_y'] > 0. 
        sel = df_c[idx]
        
        print(b,sel['delta_zp'].mean(),
              sel['delta_zp'].median(),sel['delta_zp'].std())
        
def plot_lc_feature(lc, varx='flux',vary='sigma_m5',fig=None,ax=None,marker='o',color='r'):
    
    if fig is None:
        fig, ax = plt.subplots()

    ax.plot(lc[varx],lc[vary],marker=marker,color=color,linestyle='None')
    

dira = '../test_LC_confe/baseline_v5.3.0_10yrs/DDF_spectroz/'
dirb = '../test_LC_confd/baseline_v5.3.0_10yrs/DDF_spectroz/'

snFile_a = 'SN_SN_DD_baseline_v5.3.0_10yrs_-2.0_0.2_1.hdf5'
snFile_b = 'SN_SN_DD_baseline_v5.3.0_10yrs_-2.0_0.2_1.hdf5'

sellist = selection_criteria()['G10_JLA']
print(sellist)

"""
import operator as op
sellist.append(('chisq_red',op.le,20))
sellist.append(('sigma_t0',op.le,0.5))
"""
print(sellist)
sndata_a = get_sndata(dira,snFile_a,sellist)
sndata_b = get_sndata(dirb,snFile_b,sellist)

print('alors',len(sndata_a),len(sndata_b))

meta_a = get_metadata(dira)
meta_b = get_metadata(dirb)
    
snids = meta_a['SNID'].tolist()
snids = sndata_a['SNID'].tolist()

print('snids',len(snids))
sndata_ap = pd.DataFrame()
sndata_bp = pd.DataFrame()

ccols = ['time','filter','flux','fluxerr','zp','airmass']

df_tot = pd.DataFrame()
mcols = ['SNID','x1','color','z','daymax']

io = -1
for snid in snids:
    
    io +=1
    lc_plus_sn_a = get_info(meta_a,snid,sndata_ap)
    lc_plus_sn_b = get_info(meta_b,snid,sndata_bp)
    
    lc_a = lc_plus_sn_a.lc_plot['lc']
    lc_b = lc_plus_sn_b.lc_plot['lc']
    
    dfa = lc_a[ccols].to_pandas()
    dfb = lc_b[ccols].to_pandas()
    
    
    dfc = dfa.merge(dfb, left_on=['time','filter','airmass'],
                    right_on=['time','filter','airmass'])
    
    
    info_a = get_info_sn(sndata_a,snid)
    info_b = get_info_sn(sndata_b,snid)
    
    bb = info_a.merge(info_b, left_on=mcols, right_on=mcols)
    
    df_tot = pd.concat((df_tot,bb))
    #print('add',len(df_tot),io,io-len(df_tot),len(info_a),len(info_b))
    """
    bands = dfc['filter'].unique()
    
    for b in bands:
        idx = dfc['filter'] == b
        sel = dfc[idx]
        
        fig, ax = plt.subplots()
        fig.suptitle(b)
        ax.hist(sel['zp_y']-sel['zp_x'])
        
        figb, axb = plt.subplots()
        figb.suptitle(b)
        
        axb.plot(sel['airmass'],sel['zp_y']-sel['zp_x'],'ko')
    
    break
    """
  
print('finally',len(df_tot))


"""
idx = df_tot['pull_x1_y']  >= -0.7
idx &= df_tot['pull_x1_y']  <= -0.4
df_tot = df_tot[idx]
"""

"""
plot_hist(df_tot)
plot_hist(df_tot,var=['pull_color_x','pull_color_y'])

plot_hist(df_tot,var=['chisq_red_x','chisq_red_y'])
plot_hist(df_tot,var=['diff_mu_x','diff_mu_y'])
plot_hist(df_tot,var=['sigma_mu_x','sigma_mu_y'])
plot_hist(df_tot,var=['sigma_x1_x','sigma_x1_y'])
plot_hist(df_tot,var=['sigma_color_x','sigma_color_y'])
"""
"""
plot_hist(df_tot,var=['x1_fit_x','x1_fit_y'])
plot_hist(df_tot,var=['sigma_t0_x','sigma_t0_y'])
"""
plt.show()


snids = df_tot['SNID'].to_list()

for snid in snids:
    idc = df_tot['SNID'] == snid
    sel = df_tot[idc]
    print(sel['pull_x1_y'])
    lc_plus_sn_a = get_info(meta_a,snid,sndata_a)
    lc_plus_sn_b = get_info(meta_b,snid,sndata_b)
    
    """
    lc_plus_sn_a.plot_all("time")
    lc_plus_sn_b.plot_all("time")
    """
    lc_a = lc_plus_sn_a.lc_plot['lc']
    lc_b = lc_plus_sn_b.lc_plot['lc']
    fig, ax = plt.subplots()
    plot_lc_feature(lc_a,vary='sigma_5',fig=fig,ax=ax,marker='o',color='k')
    plot_lc_feature(lc_b,vary='sigma_5',fig=fig,ax=ax,marker='*',color='r')
    #plot_lc_feature(lc_a,vary='sigma_shot',fig=fig,ax=ax,marker='*',color='r')
    #compare_lc(lc_plus_sn_a,lc_plus_sn_b)

    plt.show()

plt.show()

