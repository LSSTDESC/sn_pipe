#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 15:12:30 2026

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
from sn_plotter_analysis import plt
import pandas as pd

def plot_atmos_bias(df,col='delta_zp [mmag]',fig=None,ax=None):
    
    if fig is None:
        fig, ax = plt.subplots(figsize=(12,8))
    
    params = df[col].unique()
    
    for param in params:
        idx = df[col]==param
        sel = df[idx]
        airmass = sel['airmass'].unique()
        for airm in airmass:
            idxb = sel['airmass'] == airm
            selb = sel[idxb]
            ax.plot(selb['band'],selb['bias_value [%]'])
        

    ax.grid(visible=True)
    
    plt.show()
    


parser = OptionParser(description='summry plots from bias results')

parser.add_option('--dataDir', type=str, default='../zp_atmos_bias_summary',
                  help='data dir [%default]')
parser.add_option('--atmos_params', type=str, default='airmass,ozone,aerosol,pwv',
                  help='bias atmospheric parameters [%default]')
parser.add_option('--obs', type=str, default='zp',
                  help='variable to use as ref [%default]')
parser.add_option('--obs_unit', type=str, default='mmag',
                  help='unit variable to use as ref [%default]')
parser.add_option('--bands', type=str, default='grizy',
                  help='filters to plot [%default]')

opts, args = parser.parse_args()

theDir = opts.dataDir
atmos_params= opts.atmos_params.split(',')
obs=opts.obs
obs_unit=opts.obs_unit
bands = opts.bands

#load the data

df = pd.DataFrame()
for vv in atmos_params:
    fName = '{}/{}_atmos_bias_{}.hdf5'.format(theDir,obs,vv)
    da = pd.read_hdf(fName)
    df = pd.concat((df,da))
    
atmos_param = 'pwv'
#var_obs = 'delta_{} [{}]'.format(obs,obs_unit)
idx = df['atmos_param'] == atmos_param

plot_atmos_bias(df[idx])
print(df)