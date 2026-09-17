#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 14:26:22 2026

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from sn_plotter_analysis import plt
from sn_plotter_tools.plot_tools import plot_grid

def plot(df,b='g',xvar='airmass',yvar='zp',unit='mmag'):
    
    xxvar = 'orig_{}'.format(xvar)
    yyvar = 'mean_{}_{}'.format(yvar,b)
    
    """
    fig, ax = plt.subplots(figsize=(12,10))
    tit = '{} band - {} bias'.format(b,xvar)
    fig.suptitle(tit)
    """
    
    bias_str = 'bias_{}'.format(xvar)
    bias_values = df[bias_str]
    
    #add ref 
    cols = 'orig_{}'.format(xvar)
    dfa = df.groupby([cols]).apply(lambda x: get_var_ref(x,xvar,yvar,b),include_groups=False).reset_index()
    
    df = df.merge(dfa, left_on=cols,right_on=cols)

    yyvar_delta = 'delta_{}_{}'.format(yvar,b)
    yyvar_ref = 'mean_{}_{}_ref'.format(yvar,b)
    df[yyvar_delta] = 1000.*(df[yyvar]-df[yyvar_ref])

    idx = df[cols] >= 1.1
    idx &= df[cols] <= 2.3
    #idx &= np.abs(df[yyvar_delta]) <=5
    df = pd.DataFrame(df[idx])
    
    print(df.to_records().shape)
    
    
    plot_grid(Table.from_pandas(df),varx='orig_airmass',
              vary='bias_airmass',ylabel=yvar,
              varz=yyvar_delta,
              figtitle='{} band'.format(b),smoothIt=True)

    """    
    ax.plot(df[cols],df[bias_str],'ko')


    for bb in bias_values:
        idx = df[bias_str] == bb
      
        sel = pd.DataFrame(df[idx])
        
        
        
        #ax.plot(sel[xxvar],sel[yyvar_delta],'ko')
        
        
    ax.set_xlabel(r'{}'.format(xvar))
    yvar_str = '$\Delta$'+'{}'.format(yvar)+'[{}]'.format(unit)
    ax.set_ylabel(r'{}'.format(yvar_str))
        
    ax.grid(visible=True)
    """
        
def get_var_ref(grp,xvar='airmass',yvar='zp',band='g'):
    
    
    bias_str = 'bias_{}'.format(xvar)
    var_ref = 'mean_{}_{}'.format(yvar,band)    
    
    idx = np.abs(grp[bias_str]) < 1e-5
    
    rr = [grp[idx][var_ref].mean()]
    
    res = pd.DataFrame(rr,columns=['{}_ref'.format(var_ref)])
    
    return res
    
    
    

parser = OptionParser(description='analyze and plot zp and mean wave from bias run')

parser.add_option('--dataDir', type=str, default='../zp_atmos_bias',
                  help='data dir [%default]')
parser.add_option('--atmos_param', type=str, default='airmass',
                  help='atmospheric parameter[%default]')

opts, args = parser.parse_args()

theDir = opts.dataDir
atmos_param= opts.atmos_param
theFile = '{}/zp_atmos_{}.hdf5'.format(theDir,atmos_param)

df = pd.read_hdf(theFile)

print(df)
bands = 'grizy'
#bands = 'y'

for b in bands:
    plot(df,b)
    
plt.show()