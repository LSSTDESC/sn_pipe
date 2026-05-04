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

def load_data(dbDir,dbName,runType):
    
    search_path = '{}/{}/{}/*.hdf5'.format(dbDir,dbName,runType)
    
    fis = glob.glob(search_path)
    
    dout = pd.DataFrame()
    for fi in fis:
        df = pd.read_hdf(fi)
        dout = pd.concat((dout,df))
        
    return dout
    
    
def zdist(grp,norm_factor=1):
    
    dz = 0.1
    bins=np.arange(0.01, 1.1+dz,dz)
    idxb = grp['sigma_c'] <= 0.04
    ro = bin_it(grp[idxb],xvar='z_fit',bins=bins,norm_factor=norm_factor,outvar='nz')
    
    return ro

def plot_zdist(df,fields=['COSMOS']):
    
    
    
    fig, ax = plt.subplots(figsize=(12,8))
    
    idx = df['field'].isin(fields)
    sel = df[idx]
    
    years = sel['year'].unique()
    
    #years = years[years <=10]
    for year in years:
        idxb = sel['year'] == year
        #idxb &= sel['sigma_c'] <= 0.04
        selb = sel[idxb]
        print('counting',year,len(selb))
        #selb = selb.sort_values(by=['z_fit'])
        #print(year,len(selb),selb['nz_err'])
        ax.plot(selb['z_fit'],1.-selb['nz'].cumsum()/selb['nz'].sum())
        #ax.hist(selb['z_fit'],histtype='step')
        
    #plt.show()
    
    ax.grid(visible=True)
        

        
        
def readIt(fName,field='COSMOS'):

    df = pd.read_hdf(fName)
    
    print('before',len(df))
    idx = df['field'] == field
    
    print('after',len(df[idx]))
    return df[idx]
             
    
    
    

parser = OptionParser(description='Script to estimate SN Ia redshift distrib')

parser.add_option('--dbDir', type=str,
                  default='../new_prod_DDF_G10_JLA',
                  help='OS location dir[%default]')
parser.add_option('--dbList', type=str,
                  default='list_test.csv',
                  help='OS name [%default]')
parser.add_option('--runType', type=str,
                  default='DDF_spectroz',
                  help='run type [%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbList = opts.dbList
runType = opts.runType

dbNames = pd.read_csv(dbList,comment='#')

ccols = ['field','year']
#ccols = ['year']
fields = ['COSMOS','XMM-LSS','CDFS','ELAISS1','EDFS_a','EDFS_b']
fields = ['CDFS']
for i,row in dbNames.iterrows():
    # load the data
    df = load_data(dbDir,row['dbName'],runType)    

    print(df)
    
    
    df_z = df.groupby(ccols).apply(lambda x:zdist(x),include_groups=False).reset_index()
    
    print(df_z)
      
    plot_zdist(df_z,fields=fields)
    #plot_zdist(df_z,field='COSMOS')
    """
    ficha = '../new_prod_DDF_G10_JLA/baseline_v5.0.0_10yrs/DDF_spectroz/SN_DDF_baseline_v5.0.0_10yrs_year_3.hdf5'
    fichb = '../new_prod_DDF_G10_JLA/baseline_v5.0.0_10yrs/DDF_spectroz/SN_DDF_baseline_v5.0.0_10yrs_year_5.hdf5'
    
    tta = readIt(ficha)
    ttb = readIt(fichb)
    
    print(tta['field'].unique(),len(tta))
    print(ttb['field'].unique(),len(ttb))
    plt.hist(tta['z_fit'],histtype='step',linestyle='dotted')
    plt.hist(ttb['z_fit'],histtype='step',linestyle='dotted')
    """
    plt.show()    
    print(test)




