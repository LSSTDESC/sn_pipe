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
from sn_tools.sn_utils import multiproc

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
   
def compare_lc(lc_a,lc_b):
    
    cols = ['filter','time','flux','fluxerr','zp',
            'airmass','sigma_5','sigma_shot','flux_orig','night','snr']
    df_a = lc_a[cols].to_pandas()
    df_b = lc_b[cols].to_pandas()
    
   
    
    df_a = df_a.round({'time':6})
    df_b = df_b.round({'time':6})
    
    
    df_c = df_a.merge(df_b,left_on=['filter','time'],
                      right_on=['filter','time'])
    
    df_c['fluxerr_ratio'] = df_c['fluxerr_x']/df_c['fluxerr_y']
    df_c['flux_orig_ratio'] = df_c['flux_orig_x']/df_c['flux_orig_y']
    df_c['m5_ratio'] = df_c['sigma_5_x']/df_c['sigma_5_y']
    df_c['noise_ratio'] = df_c['sigma_shot_x']/df_c['sigma_shot_y']
    df_c['delta_zp'] = df_c['zp_x']-df_c['zp_y']
    
    return df_c
    sel = df_c.groupby(['filter']).apply(lambda x: calc_lc(x))
    
    #print(sel)
    
    """
    fig, ax = plt.subplots()
    
    var = 'delta_zp'
    idx = df_c['flux_x'] > 0. 
    idx &= df_c['flux_y'] > 0.
    #ax.hist(df_c[idx][var],histtype='step',bins=40)
    
    
    for b in df_c['filter'].unique():
        idx = df_c['filter'] == b
        idx &= df_c['flux_x'] > 0. 
        idx &= df_c['flux_y'] > 0. 
        sel = df_c[idx]
        ax.hist(sel[var],histtype='step',bins=40,label=b)
        print(b,sel['delta_zp'].mean(),
              sel['delta_zp'].median(),sel['delta_zp'].std())
    
    ax.legend()
    """
    
def calc_lc(grp,cols=['delta_zp','fluxerr_ratio',
                      'm5_ratio','noise_ratio',
                      'flux_orig_ratio','flux_x','flux_y','time']):
    
    #compare what can be compared
    """
    idx = grp['flux_x'] > 0. 
    idx &= grp['flux_y'] > 0. 
    sel = grp[idx]
    """
    sel = pd.DataFrame(grp)
    
    sel = sel.sort_values(by=['time'])
    print(grp.name)
    print(sel.columns)
    ccolsb = ['delta_zp','fluxerr_ratio',
              'flux_x','flux_y','time','flux_orig_y','fluxerr_y',
              'flux_orig_x','fluxerr_x']
    print(sel[ccolsb])
    
    
    return sel[cols]
    
    
def plot_lc_feature(lc, varx='flux',vary='sigma_m5',fig=None,ax=None,marker='o',color='r'):
    
    if fig is None:
        fig, ax = plt.subplots()

    ax.plot(lc[varx],lc[vary],marker=marker,color=color,linestyle='None')
    
def process_comp_lc(toproc, params, j=0, output_q=None):

    meta_a = params['meta_a']   
    meta_b = params['meta_b']

    df = pd.DataFrame()
    
    for snid in toproc:
        
        lc_plus_sn_a = get_info(meta_a,snid,pd.DataFrame())
        lc_plus_sn_b = get_info(meta_b,snid,pd.DataFrame())
        
        lc_a = lc_plus_sn_a.lc_plot['lc']
        lc_b = lc_plus_sn_b.lc_plot['lc']
        
        ro = compare_lc(lc_a,lc_b)

        ro['snid'] = snid
        
        df = pd.concat((df,ro))
        
    if output_q is not None:
        return output_q.put({j: df})
    else:
        return df
        
def plot_diff_lc(df):
    
    print(df.columns)
    
    print(df['filter'].unique())
    dfa = df.groupby(['filter','airmass_x'])['delta_zp'].std()
    
    print(dfa)
    
def plot_diff_indiv(grp):
    
    print('toto')

dira = '../test_LC_confe_nocoadd/baseline_v5.3.0_10yrs/DDF_spectroz/'
dirb = '../test_LC_confd_nocoadd/baseline_v5.3.0_10yrs/DDF_spectroz/'

snFile_a = 'SN_SN_DD_baseline_v5.3.0_10yrs_-2.0_0.2_1.hdf5'
snFile_b = 'SN_SN_DD_baseline_v5.3.0_10yrs_-2.0_0.2_1.hdf5'

sellist = selection_criteria()['G10_JLA']
print(sellist)

"""
import operator as op
sellist.append(('chisq_red',op.le,20))
sellist.append(('sigma_t0',op.le,0.5))
"""


meta_a = get_metadata(dira)
meta_b = get_metadata(dirb)
    
snids = meta_a['SNID'].tolist()

print('snids',len(snids))
sndata_ap = pd.DataFrame()
sndata_bp = pd.DataFrame()

ccols = ['time','filter','flux','fluxerr','zp','airmass']

df_tot = pd.DataFrame()
mcols = ['SNID','x1','color','z','daymax']

io = -1

params = {}
params['meta_a'] = meta_a
params['meta_b'] = meta_b

res = multiproc(snids,params,process_comp_lc,nproc=8)

plot_diff_lc(res)

print(res.columns)

print(test)

print(sellist)
sndata_a = get_sndata(dira,snFile_a,sellist)
sndata_b = get_sndata(dirb,snFile_b,sellist)

print('alors',len(sndata_a),len(sndata_b))

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

snids = ['SN_0108958_01_00003_6']
for snid in snids:
    idc = df_tot['SNID'] == snid
    sel = df_tot[idc]
    print(sel['pull_x1_y'])
    lc_plus_sn_a = get_info(meta_a,snid,sndata_a)
    lc_plus_sn_b = get_info(meta_b,snid,sndata_b)
    
    
    lc_plus_sn_a.plot_all("time")
    lc_plus_sn_b.plot_all("time")
    
    lc_a = lc_plus_sn_a.lc_plot['lc']
    lc_b = lc_plus_sn_b.lc_plot['lc']
    fig, ax = plt.subplots()
    plot_lc_feature(lc_a,varx='flux_orig',vary='sigma_5',fig=fig,ax=ax,marker='o',color='k')
    plot_lc_feature(lc_b,varx='flux_orig',vary='sigma_5',fig=fig,ax=ax,marker='*',color='r')
    #plot_lc_feature(lc_a,vary='sigma_shot',fig=fig,ax=ax,marker='*',color='r')
    compare_lc(lc_a,lc_b)
    print('SNID',snid)
    plt.show()

plt.show()

