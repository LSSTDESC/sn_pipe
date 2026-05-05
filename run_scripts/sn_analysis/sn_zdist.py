#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 13:52:03 2026

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import pandas as pd
import glob
from sn_analysis.sn_calc_plot import bin_it
import numpy as np
import matplotlib.pyplot as plt
from sn_analysis.sn_tools import get_spline

def load_data(dbDir,dbName,runType):
    
    search_path = '{}/{}/{}/*.hdf5'.format(dbDir,dbName,runType)
    
    fis = glob.glob(search_path)
    
    dout = pd.DataFrame()
    for fi in fis:
        df = pd.read_hdf(fi)
        dout = pd.concat((dout,df))
        
    dout['dbName'] = dbName
    return dout
    
    
def zdist(grp,norm_factor=30,sel=False):
    
    dz = 0.01
    bins=np.arange(0.01, 1.1+dz,dz)
    if sel:
        idxb = grp['sigmaC'] <= 0.04
        grp = grp[idxb]
    ro = bin_it(grp,xvar='z_fit',bins=bins,norm_factor=norm_factor,outvar='nz')
    
    return ro

def plot_zdist_indiv(df,fields=['COSMOS']):
    
    dbName = df['dbName'].unique()[0]
    
    fig, ax = plt.subplots(figsize=(12,8))
    fig.subplots_adjust(right=0.85)
    ffields = ','.join(fields)
    ffields = '{} \n {}'.format(dbName,ffields)
    fig.suptitle(ffields)
    
    idx = df['field'].isin(fields)
    sel = df[idx]
    
    sel = sel.sort_values(by=['year'])
    years = sel['year'].unique()
    
    years = years[years <=10]
    mmarkers = ['o','s','P']*3+['h']
    llstyle = ['solid']*3+['dashed']*3+['dotted']*3+['dotted']
    ccolors = ['k']*3+['b']*3+['r']*3+['r']
    yyears = range(1,11)
    md = dict(zip(yyears,mmarkers))
    ld = dict(zip(yyears,llstyle))
    cd = dict(zip(yyears,ccolors))
    
    print(cd)
    for year in years:
        idxb = sel['year'] == year
        #idxb &= sel['sigma_c'] <= 0.04
        selb = sel[idxb]
        print('counting',year,len(selb))
        #selb = selb.sort_values(by=['z_fit'])
        #print(year,len(selb),selb['nz_err'])
        ax.plot(selb['z_fit'],selb['nz'].sum()-selb['nz'].cumsum(),
                color=cd[year],marker=md[year],linestyle=ld[year],
                label='year {}'.format(year),mfc='None',markersize=10)
        #ax.hist(selb['z_fit'],histtype='step')
        
    #plt.show()
    
    ax.grid(visible=True)
    ax.set_xlim([0.06,1.06])
    ax.set_ylim([0.0,None])
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$N_{SN}(z\geq)$')
    ax.legend(bbox_to_anchor=(1., 0.8), ncol=1, frameon=False, fontsize=15)
        
def plot_zdist_all(df,config,fig=None,ax=None,figtitle=''):
    
    if fig is None:
        fig, ax = plt.subplots(figsize=(12,8))
        
    fig.subplots_adjust(right=0.8)
    if figtitle != '':
        fig.suptitle(figtitle)
    for i,row in config.iterrows():
        idx = df['dbName'] == row['dbName']
        #idx &= df['field'] == 'COSMOS'
        #idx &= df['sigma_c'] <= 0.04
        selb = df[idx]
        selb = selb.sort_values(by=['z_fit'])
        #ax.plot(selb['z_fit'],1.-selb['nz'].cumsum()/selb['nz'].sum())
        #ax.plot(selb['z_fit'],selb['nz'].sum()-selb['nz'].cumsum())
        #ax.hist(selb['z_fit'],histtype='step')
        
        dbName_plot = row['dbName_plot']
        marker = row['marker']
        color = row['color']
        ls = row['ls']
        print(selb['z_fit'])
        selb['thevar'] = selb['nz'].sum()-selb['nz'].cumsum()
        x,y = get_spline(selb,'z_fit','thevar',nx=100)
        ax.plot(x,y,label=dbName_plot,marker=marker,
                linestyle=ls,color=color,mfc='None',markersize=15,markevery=10)
    
    ax.grid(visible=True)
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$N_{SN}(z\geq)$')
    ax.set_xlim([0.06,1.06])
    ax.set_ylim([0.0,None])
    ax.legend(bbox_to_anchor=(1.3, 0.8), ncol=1, frameon=False, fontsize=13)
    plt.tight_layout()
    
def readIt_deprecated(fName,field='COSMOS'):

    df = pd.read_hdf(fName)
    
    print('before',len(df))
    idx = df['field'] == field
    
    print('after',len(df[idx]))
    return df[idx]
             
    
    
    

parser = OptionParser(description='Script to estimate SN Ia redshift distrib')

parser.add_option('--dbDir', type=str,
                  default='../new_prod_DDF_G10_JLA',
                  help='OS location dir[%default]')
parser.add_option('--config', type=str,
                  default='config_ana_selplot_newprod.csv',
                  help='config file [%default]')
parser.add_option('--runType', type=str,
                  default='DDF_spectroz',
                  help='run type [%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
config = opts.config
runType = opts.runType

df_config = pd.read_csv(config,comment='#')

ccols = ['dbName','field','year']
#ccols = ['year']
ccols=['dbName']
fields = ['COSMOS','XMM-LSS','CDFS','ELAISS1','EDFS_a','EDFS_b']
fields = ['XMM-LSS']

df_z_tot = pd.DataFrame()
df_z_tot_sel = pd.DataFrame()
df_tot = pd.DataFrame()
for i,row in df_config.iterrows():
    # load the data
    df = load_data(dbDir,row['dbName'],runType)    

    print(df)
    df_tot = pd.concat((df_tot,df))
    
    df_z = df.groupby(ccols).apply(lambda x:zdist(x),include_groups=False).reset_index()
    #plot_zdist_indiv(df_z,fields=fields)
    
    df_z_tot = pd.concat((df_z_tot,df_z))
    
    df_z_sel = df.groupby(ccols).apply(lambda x:zdist(x,sel=True),include_groups=False).reset_index()
    #plot_zdist_indiv(df_z,fields=fields)
    
    df_z_tot_sel = pd.concat((df_z_tot_sel,df_z_sel))

#fig, ax = plt.subplots(figsize=(12,8))
fig=None
ax=None
plot_zdist_all(df_z_tot,df_config,fig=fig,ax=ax)
plot_zdist_all(df_z_tot_sel,df_config,fig=fig,ax=ax,figtitle='$\sigma_C \leq 0.04$')

plt.show()
