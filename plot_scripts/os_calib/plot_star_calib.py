#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 13:47:17 2026

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import numpy as np
import pandas as pd
from sn_plotter_analysis import plt

def plot_hist(df,xvar='dist [deg]',legx='distance [deg]'):
    
    fig, ax = plt.subplots(figsize=(15,9))
    
    ax.hist(df[xvar],histtype='step',bins=20,lw=2)
    
    ax.grid(visible='True')
    
    ax.set_ylabel(r'Number of entries')
    ax.set_xlabel(r'{}'.format(legx))
    
def ana_night(grp):
    
    dd = grp.groupby(['Star_name']).size().to_frame('n').reset_index()
    
    dd['frac'] = dd['n']/dd['n'].sum()
    return dd
    
def plot_nstars(dfa,xvar='frac',year=1):
    
    #select year
    idx = dfa['night'] >= (year-1)*365. 
    idx &= dfa['night'] < year*365.
    idx &= dfa['frac'] >= 0.1
    df = dfa[idx]
    
    star_names = df['Star_name'].unique()
    
    print('allo',len(star_names))
    
    fig, ax = plt.subplots()
    for sn in star_names:
        idx = df['Star_name'] == sn
        
        sel = df[idx]
        sel = sel.sort_values(by='night')
        ax.plot(sel['night'],sel[xvar])
        
        
  
        
    
        
        

parser = OptionParser(description='figures related to obs and calib stars')

parser.add_option('--dataDir', type=str, default='../DB_Files_star_calib',
                  help='data dir [%default]')
parser.add_option('--dbName', type=str, default='baseline_v5.3.0_10yrs_star_calib',
                  help='OS name [%default]')
parser.add_option('--dbExtens', type=str, default='npy',
                  help='OS extens  [%default]')

opts, args = parser.parse_args()

dataDir = opts.dataDir
dbName = opts.dbName
dbExtens = opts.dbExtens

#load the data
fName = '{}/{}.{}'.format(dataDir,dbName,dbExtens)
obs = np.load(fName,allow_pickle=True)

#plot dist hist

#plot_hist(obs)

#plt.show()

df_obs = pd.DataFrame.from_records(obs)

dfb = df_obs.groupby(['night']).apply(lambda x: ana_night(x),include_groups=False).reset_index()

print(dfb)

plot_nstars(dfb)

plt.show()