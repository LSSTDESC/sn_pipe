#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 11:13:59 2026

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import pandas as pd
import glob
from sn_plotter_analysis import plt
import numpy as np

def load_data(dbDir, dbName, fieldType='DD'):
    """
    Function to load the data

    Parameters
    ----------
    dbDir : str
        data main directory.
    dbName : str
        OS to analyze.
    fieldType : str, optional
        type of field (DD/WFD). The default is 'DD'.

    Returns
    -------
    df : pandas df
        output data.

    """

    theDir = '{}/{}'.format(dbDir, dbName)
    prefix = '{}_pixels_{}'.format(fieldType, dbName)

    # grab the list of data
   
    fName = '{}/{}_*.hdf5'.format(theDir, prefix)
    print('path finder',fName)
    
    fis = glob.glob(fName)
    if len(fis) == 0:
        print('data not found', fName)
        
    df = pd.DataFrame()
    for fi in fis:
        dd = pd.read_hdf(fi)
        df = pd.concat((df, dd))
        
    df['dbName'] = dbName
    return df

def add_nvisits_survey(df):
    """
    Function to add the total number of visits/pixel - full survey

    Parameters
    ----------
    df : pandas df
        Data to process.

    Returns
    -------
    df : pandas df
        original df plus nvisits_10_yrs.

    """
    
    ccols = ['healpixID','dbName']
    idx = df['year'] >=0
    selb = df[idx].groupby(ccols)['nvisits'].sum().reset_index()
    selb = selb.rename(columns={'nvisits':'nvisits_10yrs'})
    df = df.merge(selb,left_on=ccols,right_on=ccols)
    
    return df

def add_dust_select(df,nside,ebvofMW_max):
    """
    Function to add dust and select

    Parameters
    ----------
    df : pandas df
        Data to process.
    nside : int
        healpix nside parameter.
    ebvofMW_max : float
        E(B-V) max.

    Returns
    -------
    df : pandas df
        output.

    """
    
    #load the dust_map

    fDust = 'reference_files/dustmap_{}_delta_mag_dust.hdf5'.format(nside)
    df_dust = pd.read_hdf(fDust)

    df = df.merge(df_dust,left_on=['healpixID'],right_on=['healpixID'])

    #select on dust

    idx = df['ebvofMW'] < ebvofMW_max
    df = pd.DataFrame(df[idx])
    
    return df
    
def plot_histo_db(df,xvar,xlabel='ooo',
                  figtit='',fig=None,ax=None,
                  lstyle='solid',color='k',marker='o'):
    
    if xvar == 'nvisits_10yrs':
        dfb = df.groupby(['healpixID','dbName'])[xvar].mean().reset_index()
        
    if fig is None:
        fig,ax = plt.subplots(figsize=(12,8))
    
    if figtit != '':
        fig.suptitle(figtit)
    
    ax.hist(dfb[xvar],bins=20,linestyle=lstyle,color=color,histtype='step')
    
    ax.set_xlabel(r'{}'.format(xlabel))
    
    
def plot_histos_db(df,xvar,xlabel,figtit):
    
    dbNames = df['dbName'].unique()
    
    lstyles = ['solid','dashed','dotted']*2
    colors = ['k','r','b','m','g','darkgrey']
    markers = ['o','s','P','v','1','2']
    
    
    ndis = np.min([len(colors),len(dbNames)])
    
    
    fig, ax = plt.subplots(figsize=(12,8))
    for i, dbName in enumerate(dbNames[:ndis]):
        idx = df['dbName'] == dbName
        sel = df[idx]
        
        plot_histo_db(sel,xvar,xlabel,figtit,
                      fig=fig,ax=ax,lstyle=lstyles[i],
                      color=colors[i],marker=markers[i])

def get_area(grp,nside):
    
    import healpy as hp

    pixArea = hp.nside2pixarea(nside, degrees=True)
    
    res = {}
    
    res['survey_area'] = [len(grp['healpixID'].unique())*pixArea]
    res['nvisits_10yrs'] = [grp['nvisits_10yrs'].median()]
    
    ccols = ['cadence']
    
    for b in 'ugrizy':
        ccols += ['cadence_{}'.format(b)]
        
    for col in ccols:
        res[col] = [grp[col].median()]
    
    return pd.DataFrame.from_dict(res)
    
def plot_vs_db(df,yvar='survey_area',ylabel='survey area [deg2]',
               ref_OS='baseline_v5.3.0_10yrs',plot_mode="diff_abs"):
    
    yvarp = yvar
    if plot_mode=="diff_abs" or plot_mode=="diff_rel":
        idx = df['dbName'] == ref_OS
        sel = df[idx]
        df = df.merge(sel,how='cross')
        yvarp = 'delta_{}'.format(yvar)
        norm = 1
        if plot_mode == "diff_rel":
            norm = df['{}_y'.format(yvar)]/100.
        df[yvarp] = (df['{}_x'.format(yvar)]-df['{}_y'.format(yvar)])/norm
        df['dbName_plot'] = df['dbName_plot_x']
    
    fig, ax = plt.subplots(figsize=(15,10))
    
    df = df.sort_values(by=[yvarp])
    ax.plot(df['dbName_plot'],df[yvarp])
    
    ax.set_ylabel(r'{}'.format(ylabel))
    
    ax.grid(visible=True)
    ax.tick_params(axis='x', labelrotation=20, labelsize=10)
    
    if plot_mode.split('_')[0] != "diff":
        #add a point corresponding to the ref
        idx = df['dbName'] == ref_OS
        
        sel = df[idx]
        ax.plot(sel['dbName_plot'],sel[yvar],marker='*',color='r',
                markersize=15)
    
    
parser = OptionParser(description='Script to plot pixel level OS infos')

parser.add_option('--dbList', type=str, default='list_OS.csv',
                  help='dbList to process [%default]')
parser.add_option('--dbDir', type=str, default='../wfd_pixels_5.3_new',
                  help='dbDir of the OS to process [%default]')
parser.add_option('--fieldType', type=str, default='WFD',
                  help='field type to process [%default]')
parser.add_option('--ref_OS', type=str, default='baseline_v5.3.0_10yrs',
                  help='ref OS [%default]')
parser.add_option('--nvisits_10yrs_min', type=int,
                  default=500,
                  help='min nvisits after 10 yrs  [%default]')
parser.add_option('--nvisits_10yrs_max', type=int,
                  default=10000,
                  help='max nvisits after 10 yrs  [%default]')
parser.add_option('--ebvofMW_max', type=float,
                  default=0.25,
                  help='max E(B-V) [%default]')
parser.add_option('--nside', type=int, default=64,
                  help='healpix nside parameter [%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbList = opts.dbList
fieldType = opts.fieldType
ref_OS = opts.ref_OS
nvisits_10yrs_min=opts.nvisits_10yrs_min
nvisits_10yrs_max=opts.nvisits_10yrs_max
ebvofMW_max = opts.ebvofMW_max
nside = opts.nside

#read dbList
dbNames = pd.read_csv(dbList,comment='#')

df = pd.DataFrame()

for i,row in dbNames.iterrows():
    dfa = load_data(dbDir, row['dbName'],fieldType)
    df = pd.concat((df,dfa))
    
print(df.columns)

df = add_nvisits_survey(df)
print('one',len(df))
# select
idx = df['nvisits_10yrs'] >= nvisits_10yrs_min
idx &= df['nvisits_10yrs'] <= nvisits_10yrs_max

df = pd.DataFrame(df[idx])

#add dust cut
df = add_dust_select(df, nside, ebvofMW_max)

#plot_histos_db(df,'nvisits_10yrs','$\Sigma N_{visits}$','10 years')

#grab survey area
df['dbName_plot'] = df['dbName'].str.split('_10yrs').str.get(0)
dd = df.groupby(['dbName','dbName_plot']).apply(lambda x: get_area(x,nside),
                                  include_groups=False).reset_index()

print(dd)

plot_vs_db(dd,plot_mode='normal')
plot_vs_db(dd,'survey_area','$\Delta$survey area [%]',plot_mode="diff_rel")
plot_vs_db(dd,'nvisits_10yrs','$\Sigma N_{visits}$',plot_mode='diff_abs')
plot_vs_db(dd,'cadence','$\Delta$cadence [night]',plot_mode="diff_abs")
plot_vs_db(dd,'cadence','$\Delta$cadence [%]',plot_mode="diff_rel")
plt.show()